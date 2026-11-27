//! Ollaya Cloud from the desktop app: signing in with the OAuth device flow (RFC 8628), an inference
//! key kept in the system's keychain, and decisions on api.ollaya.dev.
//!
//! 1. `POST /device/code` gives a code to show; the browser opens the page that approves it.
//! 2. `POST /device/token` is polled until someone approves it (see [`Poller`]). Its answer is the
//!    device session (7 days, sliding), which goes to the keychain.
//! 3. `GET /token` with the session gives a 5-minute JWT, kept in memory only.
//! 4. With the JWT, `POST /v1/orgs/{org}/keys` on the gateway makes an inference key for the
//!    account's first organization, which goes to the keychain too. Decisions use that key.
//!
//! Every request goes from here, never from the page: api.ollaya.dev refuses any request with an
//! `Origin` header, which the webview adds, and no secret may reach the page. A [`Secret`] has no
//! `Display` or `Serialize`, and its `Debug` prints nothing, so it cannot end up in an error, a log or
//! a command's answer; every message that comes back from a server goes through [`redact`] as well.
//! The page sees only [`CloudStatus`], the catalog's model names, and decide answers or [`UiError`]s.

use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tauri::{AppHandle, Emitter, Manager};

/// Ollaya Cloud's sign-in service.
const AUTH: &str = "https://ollaya.dev/dashboard/api/auth";
/// Ollaya Cloud's gateway.
const API: &str = "https://api.ollaya.dev";
const CLIENT_ID: &str = "ollaya-desktop";
const DEVICE_GRANT: &str = "urn:ietf:params:oauth:grant-type:device_code";
/// Where an organization gets credits.
const DASHBOARD_URL: &str = "https://ollaya.dev/dashboard";

/// The keychain items: the device session and the inference key, under the app's identifier.
const KEYCHAIN_SERVICE: &str = "dev.ollaya.app";
const SESSION_ITEM: &str = "ollaya-cloud-device-session";
const KEY_ITEM: &str = "ollaya-cloud-inference-key";

/// The JWT lasts 5 minutes; after 4 the app fetches a new one rather than send one about to expire.
const JWT_FRESH: Duration = Duration::from_secs(4 * 60);
/// `/device/token` answers that could not be read in a row (network, 5xx) before signing in stops.
const MAX_UNAVAILABLE: u32 = 5;
/// The event that tells the page the sign-in state changed: its payload is a [`CloudStatus`].
pub const STATUS_EVENT: &str = "cloud-status";

// --- secrets and errors ----------------------------------------------------------------------------

/// A device session, an inference key, a JWT or a device code. Its `Debug` hides it, and it has no
/// `Display` or `Serialize`: the only way to its text is [`Secret::expose`], for a request header.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    pub fn new(text: impl Into<String>) -> Self {
        Secret(text.into())
    }

    fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(..)")
    }
}

/// `text` without anything that may be a secret: the `known` ones, API keys (`oly_…`), JWTs
/// (`eyJ…`), the word after `Bearer`, any run of 32 or more token characters, and any run of 20 or
/// more that mixes letters and digits, as a session token does. At most 300 characters, so a server
/// cannot fill the page with an error.
pub fn redact(text: &str, known: &[&Secret]) -> String {
    let mut text = text.to_owned();
    for s in known.iter().filter(|s| !s.0.is_empty()) {
        text = text.replace(&s.0, "[redacted]");
    }
    let token = |c: char| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '+');
    let mut out = String::with_capacity(text.len());
    let mut word = String::new();
    let mut after_bearer = false;
    let flush = |word: &mut String, out: &mut String, after_bearer: &mut bool| {
        if word.is_empty() {
            return;
        }
        let n = word.chars().count();
        let mixed = word.chars().any(|c| c.is_ascii_digit())
            && word.chars().any(|c| c.is_ascii_alphabetic());
        let secret = *after_bearer
            || word.starts_with("oly_")
            || word.starts_with("eyJ")
            || n >= 32
            || (n >= 20 && mixed);
        out.push_str(if secret { "[redacted]" } else { word });
        *after_bearer = word.eq_ignore_ascii_case("bearer");
        word.clear();
    };
    for c in text.chars() {
        if token(c) {
            word.push(c);
        } else {
            flush(&mut word, &mut out, &mut after_bearer);
            out.push(c);
        }
    }
    flush(&mut word, &mut out, &mut after_bearer);
    match out.char_indices().nth(300) {
        Some((cut, _)) => format!("{}…", &out[..cut]),
        None => out,
    }
}

/// What went wrong with Ollaya Cloud. Its messages carry no secret (see [`redact`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CloudError {
    /// No device session, or it has ended: sign in again.
    SignedOut,
    /// Someone refused the code in the browser.
    Denied,
    /// The code expired before anyone approved it.
    Expired,
    /// The sign-in service no longer knows the code.
    InvalidGrant,
    /// Signing in was cancelled, or a new sign-in replaced it.
    Cancelled,
    /// The account has no organization to run requests for.
    NoOrganization,
    /// 402: the organization has no credits left. The gateway's message.
    InsufficientCredits(String),
    /// Any other answer from a server: its status, its `code` and its message.
    Api {
        status: u16,
        code: Option<String>,
        message: String,
    },
    /// The server could not be reached.
    Network(String),
    /// The system's keychain refused or failed.
    Keychain(String),
    /// A request the app does not send to Ollaya Cloud.
    Request(String),
}

impl CloudError {
    pub fn code(&self) -> String {
        match self {
            CloudError::SignedOut => "SIGNED_OUT".into(),
            CloudError::Denied => "ACCESS_DENIED".into(),
            CloudError::Expired => "EXPIRED".into(),
            CloudError::InvalidGrant => "INVALID_GRANT".into(),
            CloudError::Cancelled => "CANCELLED".into(),
            CloudError::NoOrganization => "NO_ORGANIZATION".into(),
            CloudError::InsufficientCredits(_) => "INSUFFICIENT_CREDITS".into(),
            CloudError::Api { code: Some(c), .. } => c.clone(),
            CloudError::Api { status, .. } => format!("HTTP_{status}"),
            CloudError::Network(_) => "NETWORK".into(),
            CloudError::Keychain(_) => "KEYCHAIN".into(),
            CloudError::Request(_) => "INVALID_REQUEST".into(),
        }
    }
}

impl fmt::Display for CloudError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CloudError::SignedOut => {
                f.write_str("You are signed out of Ollaya Cloud. Sign in again.")
            }
            CloudError::Denied => f.write_str("The sign-in was refused in the browser."),
            CloudError::Expired => {
                f.write_str("The sign-in code expired before it was approved. Sign in again.")
            }
            CloudError::InvalidGrant => {
                f.write_str("The sign-in code is no longer valid. Sign in again.")
            }
            CloudError::Cancelled => f.write_str("Signing in was cancelled."),
            CloudError::NoOrganization => write!(
                f,
                "Your Ollaya Cloud account has no organization yet. Create one at {DASHBOARD_URL}."
            ),
            CloudError::InsufficientCredits(m) => write!(
                f,
                "Your Ollaya Cloud organization has no credits left ({m}). Add credits at {DASHBOARD_URL}, or run on Local."
            ),
            CloudError::Api { message, .. } => f.write_str(message),
            CloudError::Network(m) => write!(f, "Ollaya Cloud could not be reached: {m}"),
            CloudError::Keychain(m) => write!(f, "The system keychain failed: {m}"),
            CloudError::Request(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for CloudError {}

/// An error as the page gets it.
#[derive(Debug, Clone, Serialize)]
pub struct UiError {
    code: String,
    message: String,
}

impl From<CloudError> for UiError {
    fn from(e: CloudError) -> Self {
        UiError {
            code: e.code(),
            message: e.to_string(),
        }
    }
}

/// The error a server's answer means: its `code` and its message, redacted. Never the raw body.
fn api_error(status: u16, body: &str, known: &[&Secret]) -> CloudError {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let code = v
        .get("code")
        .and_then(Value::as_str)
        .filter(|c| {
            !c.is_empty()
                && c.len() <= 64
                && c.chars()
                    .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
        })
        .map(str::to_owned);
    let message = ["error_description", "error", "message"]
        .iter()
        .find_map(|k| v.get(*k).and_then(Value::as_str).filter(|m| !m.is_empty()))
        .map(|m| redact(m, known))
        .unwrap_or_else(|| format!("Ollaya Cloud answered HTTP {status}"));
    if status == 402 || code.as_deref() == Some("INSUFFICIENT_CREDITS") {
        return CloudError::InsufficientCredits(message);
    }
    CloudError::Api {
        status,
        code,
        message,
    }
}

fn network(e: reqwest::Error) -> CloudError {
    CloudError::Network(redact(&e.without_url().to_string(), &[]))
}

/// A keychain error, with no part of an item's value.
fn keychain(e: keyring::Error) -> CloudError {
    let what = match &e {
        keyring::Error::NoStorageAccess(inner) => format!(
            "the app may not use it ({})",
            redact(&inner.to_string(), &[])
        ),
        keyring::Error::PlatformFailure(inner) => redact(&inner.to_string(), &[]),
        keyring::Error::BadEncoding(_) => "an item is not text".into(),
        _ => "it refused the item".into(),
    };
    CloudError::Keychain(what)
}

// --- the keychain ----------------------------------------------------------------------------------

/// Where the secrets are kept: the system's keychain, or memory in tests.
pub trait Vault: Send + Sync {
    fn get(&self, item: &str) -> Result<Option<Secret>, CloudError>;
    fn set(&self, item: &str, value: &Secret) -> Result<(), CloudError>;
    fn delete(&self, item: &str) -> Result<(), CloudError>;
}

/// The macOS Keychain, Windows' Credential Manager or the Secret Service, through `keyring`.
pub struct Keychain;

impl Vault for Keychain {
    fn get(&self, item: &str) -> Result<Option<Secret>, CloudError> {
        let entry = keyring::Entry::new(KEYCHAIN_SERVICE, item).map_err(keychain)?;
        match entry.get_password() {
            Ok(text) if !text.is_empty() => Ok(Some(Secret(text))),
            Ok(_) | Err(keyring::Error::NoEntry) => Ok(None),
            Err(e) => Err(keychain(e)),
        }
    }

    fn set(&self, item: &str, value: &Secret) -> Result<(), CloudError> {
        keyring::Entry::new(KEYCHAIN_SERVICE, item)
            .and_then(|entry| entry.set_password(value.expose()))
            .map_err(keychain)
    }

    fn delete(&self, item: &str) -> Result<(), CloudError> {
        let entry = keyring::Entry::new(KEYCHAIN_SERVICE, item).map_err(keychain)?;
        match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(keychain(e)),
        }
    }
}

// --- the device flow -------------------------------------------------------------------------------

/// What one `/device/token` answer means.
#[derive(Debug)]
pub enum TokenAnswer {
    /// Approved: the device session.
    Session(Secret),
    /// `authorization_pending`: nobody has approved the code yet.
    Pending,
    /// `slow_down`: poll 5 s less often from now on.
    SlowDown,
    /// HTTP 429, with Retry-After's delay when it gave one in seconds.
    RateLimited(Option<Duration>),
    /// `access_denied`.
    Denied,
    /// `expired_token`.
    Expired,
    /// `invalid_grant`.
    InvalidGrant,
    /// No answer, or a server error: worth another try or a few.
    Unavailable(CloudError),
    /// Any other answer ends the sign-in.
    Failed(CloudError),
}

/// Read a `/device/token` answer: its status, its Retry-After header and its body.
pub fn token_answer(status: u16, retry_after: Option<&str>, body: &str) -> TokenAnswer {
    if status == 429 {
        let after = retry_after
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(Duration::from_secs);
        return TokenAnswer::RateLimited(after);
    }
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    if (200..300).contains(&status) {
        return match v["access_token"].as_str().filter(|t| !t.is_empty()) {
            Some(t) => TokenAnswer::Session(Secret::new(t)),
            None => TokenAnswer::Failed(CloudError::Api {
                status,
                code: None,
                message: "the sign-in service answered without a session".into(),
            }),
        };
    }
    match v["error"].as_str() {
        Some("authorization_pending") => TokenAnswer::Pending,
        Some("slow_down") => TokenAnswer::SlowDown,
        Some("access_denied") => TokenAnswer::Denied,
        Some("expired_token") => TokenAnswer::Expired,
        Some("invalid_grant") => TokenAnswer::InvalidGrant,
        _ if status >= 500 => TokenAnswer::Unavailable(api_error(status, body, &[])),
        _ => TokenAnswer::Failed(api_error(status, body, &[])),
    }
}

/// What to do after a `/device/token` answer.
#[derive(Debug)]
pub enum Step {
    /// Ask again after this long.
    Wait(Duration),
    /// Signed in.
    Done(Secret),
    /// Stop: signing in failed.
    Fail(CloudError),
}

/// The polling of `/device/token`: every `interval` (5 s more after each `slow_down`), longer when
/// the server rate-limits, until the code expires.
#[derive(Debug)]
pub struct Poller {
    interval: Duration,
    deadline: Instant,
    unavailable: u32,
}

impl Poller {
    /// `interval` and `expires_in` in seconds, as `/device/code` gives them.
    pub fn new(interval: u64, expires_in: u64, now: Instant) -> Self {
        Poller {
            interval: Duration::from_secs(interval.max(1)),
            deadline: now + Duration::from_secs(expires_in),
            unavailable: 0,
        }
    }

    /// How long to wait before the first request.
    pub fn interval(&self) -> Duration {
        self.interval
    }

    pub fn next(&mut self, answer: TokenAnswer, now: Instant) -> Step {
        let wait = match answer {
            TokenAnswer::Session(s) => return Step::Done(s),
            TokenAnswer::Pending => {
                self.unavailable = 0;
                self.interval
            }
            TokenAnswer::SlowDown => {
                self.unavailable = 0;
                self.interval += Duration::from_secs(5);
                self.interval
            }
            TokenAnswer::RateLimited(after) => {
                after.unwrap_or(self.interval * 2).max(self.interval)
            }
            TokenAnswer::Denied => return Step::Fail(CloudError::Denied),
            TokenAnswer::Expired => return Step::Fail(CloudError::Expired),
            TokenAnswer::InvalidGrant => return Step::Fail(CloudError::InvalidGrant),
            TokenAnswer::Unavailable(e) => {
                self.unavailable += 1;
                if self.unavailable >= MAX_UNAVAILABLE {
                    return Step::Fail(e);
                }
                self.interval
            }
            TokenAnswer::Failed(e) => return Step::Fail(e),
        };
        if now + wait >= self.deadline {
            return Step::Fail(CloudError::Expired);
        }
        Step::Wait(wait)
    }
}

// --- HTTP ------------------------------------------------------------------------------------------

fn http() -> &'static reqwest::Client {
    static HTTP: OnceLock<reqwest::Client> = OnceLock::new();
    HTTP.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(concat!("ollaya-desktop/", env!("CARGO_PKG_VERSION")))
            // A redirect would carry a request elsewhere; Ollaya Cloud's endpoints answer directly.
            .redirect(reqwest::redirect::Policy::none())
            .timeout(Duration::from_secs(30))
            .build()
            .expect("an HTTP client")
    })
}

/// A server's answer, read whole.
struct Answer {
    status: u16,
    retry_after: Option<String>,
    body: String,
}

impl Answer {
    fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }

    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

/// Send `req` with `auth` as its Bearer token, if any. reqwest adds no `Origin` header.
async fn send(req: reqwest::RequestBuilder, auth: Option<&Secret>) -> Result<Answer, CloudError> {
    let mut req = req.header(reqwest::header::ACCEPT, "application/json");
    if let Some(auth) = auth {
        req = req.bearer_auth(auth.expose());
    }
    let r = req.send().await.map_err(network)?;
    let status = r.status().as_u16();
    let retry_after = r
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let body = r.text().await.map_err(network)?;
    Ok(Answer {
        status,
        retry_after,
        body,
    })
}

/// This computer's name, for the key's name on the dashboard.
fn hostname() -> String {
    let from_command = |cmd: &str, args: &[&str]| {
        std::process::Command::new(cmd)
            .args(args)
            .stdin(std::process::Stdio::null())
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
    };
    let name = if cfg!(windows) {
        std::env::var("COMPUTERNAME").ok()
    } else if cfg!(target_os = "macos") {
        from_command("scutil", &["--get", "ComputerName"]).or_else(|| from_command("hostname", &[]))
    } else {
        std::fs::read_to_string("/etc/hostname")
            .ok()
            .map(|s| s.trim().to_owned())
            .or_else(|| from_command("hostname", &[]))
    };
    let name: String = name
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_control())
        .take(64)
        .collect();
    if name.is_empty() {
        "this computer".into()
    } else {
        name
    }
}

/// An organization id the app puts in a URL path.
fn valid_id(id: &str) -> bool {
    (1..=64).contains(&id.len())
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

// --- state -----------------------------------------------------------------------------------------

/// Where decisions run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    #[default]
    Local,
    Cloud,
}

/// A sign-in waiting for someone to approve the code in the browser.
#[derive(Debug, Clone, Serialize)]
pub struct Pending {
    pub user_code: String,
    pub verification_uri: String,
    /// The page with the code filled in, which the browser opens.
    #[serde(skip)]
    open: String,
}

/// What the page knows about Ollaya Cloud. Never a secret.
#[derive(Debug, Clone, Serialize)]
pub struct CloudStatus {
    pub signed_in: bool,
    pub email: Option<String>,
    pub mode: Mode,
    /// While signing in: the code to show.
    pub sign_in: Option<Pending>,
    /// Why the last sign-in failed.
    pub error: Option<String>,
}

/// What the app remembers in its config directory (`cloud.json`): no secret.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Settings {
    #[serde(default)]
    mode: Mode,
    #[serde(default)]
    email: Option<String>,
}

#[derive(Default)]
struct Inner {
    loaded: bool,
    session: Option<Secret>,
    key: Option<Secret>,
    jwt: Option<(Secret, Instant)>,
    org: Option<String>,
    settings: Settings,
    pending: Option<Pending>,
    error: Option<String>,
}

/// A started sign-in, for [`Cloud::finish_sign_in`].
pub struct SignIn {
    attempt: u64,
    device_code: Secret,
    interval: u64,
    expires_in: u64,
}

/// The app's Ollaya Cloud state. Locks are held for moments, never across a request.
pub struct Cloud {
    vault: Box<dyn Vault>,
    settings_path: Mutex<Option<PathBuf>>,
    inner: Mutex<Inner>,
    /// Bumped by each sign-in and each cancel: a poll whose attempt is stale stops.
    attempt: AtomicU64,
}

impl Cloud {
    pub fn new(vault: Box<dyn Vault>) -> Self {
        Cloud {
            vault,
            settings_path: Mutex::new(None),
            inner: Mutex::new(Inner::default()),
            attempt: AtomicU64::new(0),
        }
    }

    /// Where `cloud.json` lives; set once at start-up.
    pub fn set_settings_path(&self, path: PathBuf) {
        *self.settings_path.lock().expect("cloud settings path") = Some(path);
    }

    fn inner(&self) -> std::sync::MutexGuard<'_, Inner> {
        let mut inner = self.inner.lock().expect("cloud lock");
        if !inner.loaded {
            inner.loaded = true;
            let path = self
                .settings_path
                .lock()
                .expect("cloud settings path")
                .clone();
            if let Some(text) = path.and_then(|p| std::fs::read_to_string(p).ok()) {
                inner.settings = serde_json::from_str(&text).unwrap_or_default();
            }
            match self.vault.get(SESSION_ITEM) {
                Ok(session) => inner.session = session,
                Err(e) => inner.error = Some(e.to_string()),
            }
        }
        inner
    }

    fn save(&self, settings: &Settings) {
        let Some(path) = self
            .settings_path
            .lock()
            .expect("cloud settings path")
            .clone()
        else {
            return;
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(text) = serde_json::to_string_pretty(settings) {
            let _ = std::fs::write(path, text);
        }
    }

    pub fn status(&self) -> CloudStatus {
        let inner = self.inner();
        let signed_in = inner.session.is_some();
        CloudStatus {
            signed_in,
            email: inner.settings.email.clone().filter(|_| signed_in),
            mode: inner.settings.mode,
            sign_in: inner.pending.clone(),
            error: inner.error.clone(),
        }
    }

    pub fn set_mode(&self, mode: Mode) -> CloudStatus {
        {
            let mut inner = self.inner();
            inner.settings.mode = mode;
            self.save(&inner.settings);
        }
        self.status()
    }

    /// The page to approve the code on, while signing in.
    pub fn verification_page(&self) -> Option<String> {
        self.inner().pending.as_ref().map(|p| p.open.clone())
    }

    /// Step 1: ask for a code. The caller opens [`Cloud::verification_page`] and polls with
    /// [`Cloud::finish_sign_in`].
    pub async fn start_sign_in(&self) -> Result<SignIn, CloudError> {
        let a = send(
            http()
                .post(format!("{AUTH}/device/code"))
                .json(&json!({ "client_id": CLIENT_ID })),
            None,
        )
        .await?;
        if !a.ok() {
            return Err(api_error(a.status, &a.body, &[]));
        }
        let v = a.json();
        let text = |k: &str| v[k].as_str().filter(|s| !s.is_empty()).map(str::to_owned);
        let (Some(device_code), Some(user_code), Some(uri)) = (
            text("device_code"),
            text("user_code"),
            text("verification_uri"),
        ) else {
            return Err(CloudError::Api {
                status: a.status,
                code: None,
                message: "the sign-in service gave no code".into(),
            });
        };
        let open = text("verification_uri_complete").unwrap_or_else(|| uri.clone());
        if !(uri.starts_with("https://") && open.starts_with("https://")) {
            return Err(CloudError::Request(
                "the sign-in page is not an https link".into(),
            ));
        }
        let attempt = self.attempt.fetch_add(1, Ordering::SeqCst) + 1;
        let mut inner = self.inner();
        inner.error = None;
        inner.pending = Some(Pending {
            user_code,
            verification_uri: uri,
            open,
        });
        Ok(SignIn {
            attempt,
            device_code: Secret(device_code),
            interval: v["interval"].as_u64().unwrap_or(5),
            expires_in: v["expires_in"].as_u64().unwrap_or(1800),
        })
    }

    /// Stop waiting for the code.
    pub fn cancel_sign_in(&self) {
        self.attempt.fetch_add(1, Ordering::SeqCst);
        let mut inner = self.inner();
        inner.pending = None;
        inner.error = None;
    }

    /// Step 2: poll `/device/token` until the code is approved, refused or expired, then keep the
    /// session. The outcome is in [`Cloud::status`].
    pub async fn finish_sign_in(&self, s: SignIn) {
        let result = self.poll(&s).await;
        if self.attempt.load(Ordering::SeqCst) != s.attempt {
            return; // cancelled, or another sign-in took over
        }
        let result = match result {
            Ok(session) => self.keep_session(session).await,
            Err(e) => Err(e),
        };
        let mut inner = self.inner();
        inner.pending = None;
        inner.error = match result {
            Ok(()) | Err(CloudError::Cancelled) => None,
            Err(e) => Some(e.to_string()),
        };
    }

    async fn poll(&self, s: &SignIn) -> Result<Secret, CloudError> {
        let mut poller = Poller::new(s.interval, s.expires_in, Instant::now());
        let mut wait = poller.interval();
        let body = json!({
            "grant_type": DEVICE_GRANT,
            "device_code": s.device_code.expose(),
            "client_id": CLIENT_ID,
        });
        loop {
            tokio::time::sleep(wait).await;
            if self.attempt.load(Ordering::SeqCst) != s.attempt {
                return Err(CloudError::Cancelled);
            }
            let answer = match send(
                http().post(format!("{AUTH}/device/token")).json(&body),
                None,
            )
            .await
            {
                Ok(a) => token_answer(a.status, a.retry_after.as_deref(), &a.body),
                Err(e) => TokenAnswer::Unavailable(e),
            };
            match poller.next(answer, Instant::now()) {
                Step::Wait(d) => wait = d,
                Step::Done(session) => return Ok(session),
                Step::Fail(e) => return Err(e),
            }
        }
    }

    /// Keep a new device session: in the keychain, in place of any earlier session and key.
    async fn keep_session(&self, session: Secret) -> Result<(), CloudError> {
        self.vault.delete(KEY_ITEM)?;
        self.vault.set(SESSION_ITEM, &session)?;
        let email = self.email(&session).await;
        let mut inner = self.inner();
        inner.session = Some(session);
        inner.key = None;
        inner.jwt = None;
        inner.org = None;
        inner.settings.email = email;
        self.save(&inner.settings);
        Ok(())
    }

    /// The account's email, when the sign-in service says it.
    async fn email(&self, session: &Secret) -> Option<String> {
        let a = send(http().get(format!("{AUTH}/get-session")), Some(session))
            .await
            .ok()
            .filter(Answer::ok)?;
        a.json()["user"]["email"]
            .as_str()
            .filter(|e| e.contains('@') && e.len() <= 254)
            .map(str::to_owned)
    }

    /// Forget the session and the key, here and in the keychain: the session has ended.
    fn forget(&self) -> Result<(), CloudError> {
        {
            let mut inner = self.inner();
            inner.session = None;
            inner.key = None;
            inner.jwt = None;
            inner.org = None;
            inner.settings.email = None;
            self.save(&inner.settings);
        }
        let key = self.vault.delete(KEY_ITEM);
        self.vault.delete(SESSION_ITEM)?;
        key
    }

    /// Step 6: end the device session, then delete both keychain items.
    pub async fn sign_out(&self) -> Result<(), CloudError> {
        self.cancel_sign_in();
        let session = self.inner().session.clone();
        if let Some(session) = session {
            // Best effort: the session is forgotten here even when the server cannot be told.
            let _ = send(
                http().post(format!("{AUTH}/sign-out")).json(&json!({})),
                Some(&session),
            )
            .await;
        }
        self.forget()
    }

    fn session(&self) -> Result<Secret, CloudError> {
        self.inner().session.clone().ok_or(CloudError::SignedOut)
    }

    /// Step 3: a JWT for the gateway, fetched again once it is 4 minutes old.
    async fn jwt(&self) -> Result<Secret, CloudError> {
        if let Some((jwt, at)) = &self.inner().jwt
            && at.elapsed() < JWT_FRESH
        {
            return Ok(jwt.clone());
        }
        let session = self.session()?;
        let a = send(http().get(format!("{AUTH}/token")), Some(&session)).await?;
        if a.status == 401 {
            self.forget()?;
            return Err(CloudError::SignedOut);
        }
        if !a.ok() {
            return Err(api_error(a.status, &a.body, &[&session]));
        }
        let jwt = a.json()["token"]
            .as_str()
            .filter(|t| !t.is_empty())
            .map(Secret::new)
            .ok_or_else(|| CloudError::Api {
                status: a.status,
                code: None,
                message: "the sign-in service gave no token".into(),
            })?;
        self.inner().jwt = Some((jwt.clone(), Instant::now()));
        Ok(jwt)
    }

    /// The account's first organization.
    async fn org(&self) -> Result<String, CloudError> {
        if let Some(org) = self.inner().org.clone() {
            return Ok(org);
        }
        let session = self.session()?;
        let a = send(
            http().get(format!("{AUTH}/organization/list")),
            Some(&session),
        )
        .await?;
        if a.status == 401 {
            self.forget()?;
            return Err(CloudError::SignedOut);
        }
        if !a.ok() {
            return Err(api_error(a.status, &a.body, &[&session]));
        }
        let v = a.json();
        let first = v.as_array().and_then(|orgs| orgs.first());
        let org = first
            .and_then(|o| o["id"].as_str())
            .ok_or(CloudError::NoOrganization)?;
        if !valid_id(org) {
            return Err(CloudError::Request(
                "Ollaya Cloud gave an organization id the app does not use".into(),
            ));
        }
        self.inner().org = Some(org.to_owned());
        Ok(org.to_owned())
    }

    /// Step 5: a new inference key for the organization, kept in the keychain.
    async fn new_key(&self) -> Result<Secret, CloudError> {
        let org = self.org().await?;
        let body =
            json!({ "name": format!("Ollaya desktop ({})", hostname()), "scope": "inference" });
        let url = format!("{API}/v1/orgs/{org}/keys");
        let mut jwt = self.jwt().await?;
        let mut a = send(http().post(&url).json(&body), Some(&jwt)).await?;
        if a.status == 401 {
            // The JWT may have expired on the way: one more try with a new one.
            self.inner().jwt = None;
            jwt = self.jwt().await?;
            a = send(http().post(&url).json(&body), Some(&jwt)).await?;
        }
        if !a.ok() {
            return Err(api_error(a.status, &a.body, &[&jwt]));
        }
        let key = a.json()["key"]
            .as_str()
            .filter(|k| !k.is_empty())
            .map(Secret::new)
            .ok_or_else(|| CloudError::Api {
                status: a.status,
                code: None,
                message: "the gateway gave no key".into(),
            })?;
        self.vault.set(KEY_ITEM, &key)?;
        self.inner().key = Some(key.clone());
        Ok(key)
    }

    /// The inference key: from memory, else the keychain, else a new one.
    async fn key(&self) -> Result<Secret, CloudError> {
        self.session()?;
        if let Some(key) = self.inner().key.clone() {
            return Ok(key);
        }
        if let Some(key) = self.vault.get(KEY_ITEM)? {
            self.inner().key = Some(key.clone());
            return Ok(key);
        }
        self.new_key().await
    }

    /// `POST /api/decide` on the gateway with `body`, as the local server takes it. A 401 means the
    /// key was revoked or has expired: a new key, once.
    pub async fn decide(&self, body: &Value) -> Result<Value, CloudError> {
        let url = format!("{API}/api/decide");
        let mut key = self.key().await?;
        let mut a = send(http().post(&url).json(body), Some(&key)).await?;
        if a.status == 401 {
            self.inner().key = None;
            self.vault.delete(KEY_ITEM)?;
            key = self.new_key().await?;
            a = send(http().post(&url).json(body), Some(&key)).await?;
        }
        if !a.ok() {
            return Err(api_error(a.status, &a.body, &[&key]));
        }
        serde_json::from_str(&a.body).map_err(|_| CloudError::Api {
            status: a.status,
            code: None,
            message: "Ollaya Cloud's answer is not JSON".into(),
        })
    }
}

/// The models Ollaya Cloud runs (`GET /v1/catalog`, public). Their names differ from local ones.
pub async fn models() -> Result<Vec<String>, CloudError> {
    let a = send(http().get(format!("{API}/v1/catalog")), None).await?;
    if !a.ok() {
        return Err(api_error(a.status, &a.body, &[]));
    }
    let v = a.json();
    Ok(v["data"]
        .as_array()
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m["name"].as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default())
}

// --- commands --------------------------------------------------------------------------------------

fn changed(app: &AppHandle) {
    let _ = app.emit(STATUS_EVENT, app.state::<Cloud>().status());
}

// The commands that may read the keychain are async, so they never block the main thread.

#[tauri::command]
pub async fn cloud_status(app: AppHandle) -> CloudStatus {
    app.state::<Cloud>().status()
}

/// Ask for a code, open the browser on it, and wait for its approval in the background
/// ([`STATUS_EVENT`] tells the page how it ended).
#[tauri::command]
pub async fn cloud_sign_in(app: AppHandle) -> Result<CloudStatus, UiError> {
    let cloud = app.state::<Cloud>();
    let s = cloud.start_sign_in().await?;
    if let Some(page) = cloud.verification_page() {
        crate::open_url(&page);
    }
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        handle.state::<Cloud>().finish_sign_in(s).await;
        changed(&handle);
    });
    Ok(cloud.status())
}

#[tauri::command]
pub async fn cloud_open_sign_in(app: AppHandle) {
    if let Some(page) = app.state::<Cloud>().verification_page() {
        crate::open_url(&page);
    }
}

#[tauri::command]
pub async fn cloud_cancel_sign_in(app: AppHandle) -> CloudStatus {
    let cloud = app.state::<Cloud>();
    cloud.cancel_sign_in();
    cloud.status()
}

#[tauri::command]
pub async fn cloud_sign_out(app: AppHandle) -> Result<CloudStatus, UiError> {
    let cloud = app.state::<Cloud>();
    let result = cloud.sign_out().await;
    changed(&app);
    result?;
    Ok(cloud.status())
}

#[tauri::command]
pub async fn cloud_set_mode(app: AppHandle, mode: Mode) -> CloudStatus {
    app.state::<Cloud>().set_mode(mode)
}

#[tauri::command]
pub async fn cloud_models() -> Result<Vec<String>, UiError> {
    Ok(models().await?)
}

/// The Playground's request, run on Ollaya Cloud: the same body as the local `decide`. Custom
/// presets live on the local server, so only the built-in ones and custom questions go.
#[tauri::command]
pub async fn cloud_decide(
    app: AppHandle,
    model: String,
    state: String,
    preset: Option<String>,
    questions: Option<String>,
) -> Result<Value, UiError> {
    if let Some(p) = preset.as_deref()
        && ollaya_api::presets::get(p).is_none()
    {
        return Err(CloudError::Request(format!(
            "{p:?} is saved on your local server: on Ollaya Cloud, pick a built-in preset or Custom JSON"
        ))
        .into());
    }
    let request =
        crate::decide_request(model, state, preset, questions).map_err(CloudError::Request)?;
    let body = serde_json::to_value(&request).map_err(|e| CloudError::Request(e.to_string()))?;
    let cloud = app.state::<Cloud>();
    let result = cloud.decide(&body).await;
    if matches!(result, Err(CloudError::SignedOut)) {
        changed(&app);
    }
    Ok(result?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    const SESSION: &str = "Xk2pQ9vLm3Rt7Wz1Yb5Nc8Df0Gh4Jk6";
    const KEY: &str = "oly_live_8f3a9c2e1d7b6a5f4e3d2c1b0a9f8e7d";
    const JWT: &str = "eyJhbGciOiJFZERTQSJ9.eyJpZCI6InUxIn0.c2lnbmF0dXJlLWJ5dGVz";

    #[derive(Default)]
    struct Memory(Mutex<HashMap<String, Secret>>);

    impl Vault for Memory {
        fn get(&self, item: &str) -> Result<Option<Secret>, CloudError> {
            Ok(self.0.lock().unwrap().get(item).cloned())
        }
        fn set(&self, item: &str, value: &Secret) -> Result<(), CloudError> {
            self.0.lock().unwrap().insert(item.into(), value.clone());
            Ok(())
        }
        fn delete(&self, item: &str) -> Result<(), CloudError> {
            self.0.lock().unwrap().remove(item);
            Ok(())
        }
    }

    fn pending(body: &str) -> TokenAnswer {
        token_answer(400, None, body)
    }

    #[test]
    fn polling_waits_while_pending_and_slows_down_by_5_s() {
        let t0 = Instant::now();
        let mut p = Poller::new(5, 1800, t0);
        assert_eq!(p.interval(), Duration::from_secs(5));
        let step = p.next(pending(r#"{"error":"authorization_pending"}"#), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(5)));
        let step = p.next(pending(r#"{"error":"slow_down"}"#), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(10)));
        // The slower pace stays.
        let step = p.next(pending(r#"{"error":"authorization_pending"}"#), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(10)));
        let step = p.next(pending(r#"{"error":"slow_down"}"#), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(15)));
    }

    #[test]
    fn polling_backs_off_on_429_by_retry_after() {
        let t0 = Instant::now();
        let mut p = Poller::new(5, 1800, t0);
        let step = p.next(token_answer(429, Some("30"), ""), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(30)));
        // Never sooner than the interval, and twice the interval without a usable Retry-After.
        let step = p.next(token_answer(429, Some("1"), ""), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(5)));
        let step = p.next(
            token_answer(429, Some("Wed, 21 Oct 2026 07:28:00 GMT"), ""),
            t0,
        );
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(10)));
        let step = p.next(
            token_answer(429, None, r#"{"error":"authorization_pending"}"#),
            t0,
        );
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(10)));
        // The rate limit does not change the interval.
        let step = p.next(pending(r#"{"error":"authorization_pending"}"#), t0);
        assert!(matches!(step, Step::Wait(d) if d == Duration::from_secs(5)));
    }

    #[test]
    fn polling_stops_when_denied_expired_or_invalid() {
        let t0 = Instant::now();
        let cases = [
            (r#"{"error":"access_denied"}"#, CloudError::Denied),
            (r#"{"error":"expired_token"}"#, CloudError::Expired),
            (r#"{"error":"invalid_grant"}"#, CloudError::InvalidGrant),
        ];
        for (body, want) in cases {
            let mut p = Poller::new(5, 1800, t0);
            match p.next(pending(body), t0) {
                Step::Fail(e) => assert_eq!(e, want),
                other => panic!("{body}: {other:?}"),
            }
        }
        // Another client error ends it too, with the server's message.
        let mut p = Poller::new(5, 1800, t0);
        match p.next(
            pending(r#"{"error":"invalid_client","error_description":"unknown client"}"#),
            t0,
        ) {
            Step::Fail(CloudError::Api {
                status: 400,
                message,
                ..
            }) => assert_eq!(message, "unknown client"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn polling_ends_when_the_code_expires() {
        let t0 = Instant::now();
        let mut p = Poller::new(5, 12, t0);
        assert!(matches!(
            p.next(pending(r#"{"error":"authorization_pending"}"#), t0),
            Step::Wait(_)
        ));
        let later = t0 + Duration::from_secs(8);
        match p.next(pending(r#"{"error":"authorization_pending"}"#), later) {
            Step::Fail(CloudError::Expired) => {}
            other => panic!("{other:?}"),
        }
        // A Retry-After beyond the code's life ends it now rather than after the wait.
        let mut p = Poller::new(5, 60, t0);
        assert!(matches!(
            p.next(token_answer(429, Some("600"), ""), t0),
            Step::Fail(CloudError::Expired)
        ));
    }

    #[test]
    fn polling_retries_a_few_server_errors_then_stops() {
        let t0 = Instant::now();
        let mut p = Poller::new(5, 1800, t0);
        for _ in 1..MAX_UNAVAILABLE {
            assert!(matches!(
                p.next(token_answer(503, None, "<html>"), t0),
                Step::Wait(_)
            ));
        }
        assert!(matches!(
            p.next(token_answer(503, None, "<html>"), t0),
            Step::Fail(CloudError::Api { status: 503, .. })
        ));
        // A pending answer in between starts the count again.
        let mut p = Poller::new(5, 1800, t0);
        for _ in 0..3 * MAX_UNAVAILABLE {
            assert!(matches!(
                p.next(token_answer(502, None, ""), t0),
                Step::Wait(_)
            ));
            assert!(matches!(
                p.next(pending(r#"{"error":"authorization_pending"}"#), t0),
                Step::Wait(_)
            ));
        }
    }

    #[test]
    fn an_approved_code_gives_the_session() {
        let t0 = Instant::now();
        let mut p = Poller::new(5, 1800, t0);
        let body =
            format!(r#"{{"access_token":"{SESSION}","token_type":"Bearer","expires_in":604800}}"#);
        match p.next(token_answer(200, None, &body), t0) {
            Step::Done(s) => assert_eq!(s.expose(), SESSION),
            other => panic!("{other:?}"),
        }
        assert!(matches!(
            p.next(token_answer(200, None, "{}"), t0),
            Step::Fail(_)
        ));
    }

    #[test]
    fn secrets_never_reach_error_strings() {
        let (session, key, jwt) = (Secret::new(SESSION), Secret::new(KEY), Secret::new(JWT));
        let all = [SESSION, KEY, JWT];
        let clean = |what: &str, text: String| {
            for s in all {
                assert!(!text.contains(s), "{what} leaks a secret: {text}");
            }
        };
        clean("Secret's Debug", format!("{session:?} {key:?} {jwt:?}"));
        clean("a Step", format!("{:?}", Step::Done(session.clone())));
        clean(
            "a TokenAnswer",
            format!("{:?}", TokenAnswer::Session(key.clone())),
        );

        // Servers that echo what they were sent, as an error, a code or a description.
        let bodies = [
            format!(r#"{{"error":"invalid key {KEY}","code":"UNAUTHORIZED"}}"#),
            format!(r#"{{"error":"bad session","code":"{KEY}"}}"#),
            format!(
                r#"{{"error":"invalid_token","error_description":"Bearer {SESSION} is not valid"}}"#
            ),
            format!(r#"{{"message":"jwt {JWT} expired"}}"#),
            format!(r#"{{"error":"no credits for {KEY}","code":"INSUFFICIENT_CREDITS"}}"#),
            format!(r#"{{"error":"session {SESSION} ended"}}"#),
        ];
        for status in [400, 401, 402, 403, 429, 500] {
            for body in &bodies {
                // With the secret named, and with it unknown to the caller.
                for known in [&[&session, &key, &jwt][..], &[]] {
                    let e = api_error(status, body, known);
                    clean("an API error", e.to_string());
                    clean("an API error's Debug", format!("{e:?}"));
                    clean("an API error's code", e.code());
                    clean(
                        "the page's error",
                        serde_json::to_string(&UiError::from(e)).unwrap(),
                    );
                }
                let answer = token_answer(status, None, body);
                clean("a token answer", format!("{answer:?}"));
                if let Step::Fail(e) =
                    Poller::new(5, 1800, Instant::now()).next(answer, Instant::now())
                {
                    clean("a failed sign-in", e.to_string());
                }
            }
        }
        // A keychain error never carries an item's bytes.
        let e = keychain(keyring::Error::BadEncoding(KEY.as_bytes().to_vec()));
        clean("a keychain error", format!("{e} {e:?}"));
        // Nor does what the page gets about the account.
        let cloud = Cloud::new(Box::<Memory>::default());
        cloud.vault.set(SESSION_ITEM, &session).unwrap();
        cloud.vault.set(KEY_ITEM, &key).unwrap();
        cloud.inner().jwt = Some((jwt.clone(), Instant::now()));
        let status = cloud.status();
        assert!(status.signed_in);
        clean("the status", serde_json::to_string(&status).unwrap());
    }

    #[test]
    fn redact_keeps_plain_messages() {
        let plain = "model \"coba:1b\" is not in the catalog; see https://ollaya.dev/docs/api";
        assert_eq!(redact(plain, &[]), plain);
        assert_eq!(redact("Bearer abc", &[]), "Bearer [redacted]");
        assert_eq!(redact(&format!("key={KEY}!"), &[]), "key=[redacted]!");
        assert!(redact(&"x ".repeat(400), &[]).chars().count() <= 301);
    }

    #[test]
    fn credits_and_codes_read_from_the_gateway() {
        let e = api_error(
            402,
            r#"{"error":"insufficient credits","code":"INSUFFICIENT_CREDITS"}"#,
            &[],
        );
        assert_eq!(
            e,
            CloudError::InsufficientCredits("insufficient credits".into())
        );
        assert_eq!(e.code(), "INSUFFICIENT_CREDITS");
        let e = api_error(
            404,
            r#"{"error":"no such model","code":"MODEL_NOT_FOUND"}"#,
            &[],
        );
        assert_eq!(e.code(), "MODEL_NOT_FOUND");
        assert_eq!(e.to_string(), "no such model");
        // Not JSON: the status alone.
        let e = api_error(502, "<html>Bad gateway</html>", &[]);
        assert_eq!(e.to_string(), "Ollaya Cloud answered HTTP 502");
        assert_eq!(e.code(), "HTTP_502");
    }

    #[test]
    fn signing_out_forgets_both_items_and_keeps_the_mode() {
        let dir = std::env::temp_dir().join(format!("ollaya-desktop-cloud-{}", std::process::id()));
        let cloud = Cloud::new(Box::<Memory>::default());
        cloud.set_settings_path(dir.join("cloud.json"));
        assert!(!cloud.status().signed_in);
        assert_eq!(cloud.status().mode, Mode::Local);
        cloud
            .vault
            .set(SESSION_ITEM, &Secret::new(SESSION))
            .unwrap();
        cloud.vault.set(KEY_ITEM, &Secret::new(KEY)).unwrap();
        cloud.inner().session = Some(Secret::new(SESSION));
        cloud.set_mode(Mode::Cloud);
        cloud.forget().unwrap();
        assert_eq!(cloud.vault.get(SESSION_ITEM).unwrap(), None);
        assert_eq!(cloud.vault.get(KEY_ITEM).unwrap(), None);
        assert!(!cloud.status().signed_in);
        // The choice of where decisions run is remembered, in a file without secrets.
        let again = Cloud::new(Box::<Memory>::default());
        again.set_settings_path(dir.join("cloud.json"));
        assert_eq!(again.status().mode, Mode::Cloud);
        let saved = std::fs::read_to_string(dir.join("cloud.json")).unwrap();
        assert!(!saved.contains(SESSION) && !saved.contains(KEY));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn organization_ids_are_url_safe() {
        assert!(valid_id("org_2fG8-x"));
        assert!(!valid_id(""));
        assert!(!valid_id("../keys"));
        assert!(!valid_id("a/b"));
        assert!(!valid_id(&"a".repeat(65)));
    }
}
