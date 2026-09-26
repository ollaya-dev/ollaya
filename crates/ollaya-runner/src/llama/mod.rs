//! The llama.cpp engine: GGUF decision models (`winnow-v1`, `llm-logits-v1`) on libllama, inside
//! the runner process (docs/decisions/0003-llama-cpp-runtime.md).
//!
//! The libraries are llama.cpp's own release build, loaded at run time from the install
//! (`lib/ollaya/llama`, plus `libggml-cuda.so` from the CUDA pack): see [`ffi`]. The layouts in
//! `ollaya_decision` build each question's prompt; this engine tokenizes it with the GGUF's own
//! tokenizer, evaluates it and returns the option labels' logits in wire order, the runner
//! contract every engine shares. Calibration and answers stay in the daemon.
//!
//! Every question is evaluated with one fixed plan, so its numbers never depend on what ran
//! before it or on the other questions of its request: the prompt's state prefix `ids[..p]` as
//! one cold pass (kept while the next question shares it), then the rest `ids[p..]` as one batch,
//! reading the logits at the last token. llama.cpp's numbers depend on how a prompt is split into
//! batches, so the split is part of the model's numerics; the goldens' reference
//! (`convert/ollaya_convert/families/llm_common/plan.py`, on the same build's llama-server) uses
//! the same one.

pub mod ffi;

use std::ffi::{CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;

use ollaya_decision::Questions;
use ollaya_decision::llm_logits::{self, LlmLogitsConfig};
use ollaya_decision::winnow::{self, WinnowConfig};
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{Error, Output, QuestionOutput};
use ffi::{Api, Batch, Token};

/// Layouts the llama engine runs.
pub const LAYOUTS: &[&str] = &[llm_logits::LAYOUT, winnow::LAYOUT];

/// Tokens per `llama_decode` call and per physical batch: llama-server's defaults, which the
/// goldens' reference runs with.
const N_BATCH: usize = 2048;
const N_UBATCH: u32 = 512;

/// `decision.json` → `llama`: how the model runs.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LlamaSettings {
    /// Context size in tokens.
    pub n_ctx: usize,
    /// A full-size sliding-window cache, so a question's cache can be cut back to its prefix.
    #[serde(default)]
    pub swa_full: bool,
    /// `prefix` (the shared-prefix plan) or `cold` (one cold pass per question).
    #[serde(default = "prefix_plan")]
    pub plan: String,
}

fn prefix_plan() -> String {
    "prefix".into()
}

#[derive(Debug, Clone, Deserialize)]
struct Decision {
    engine: String,
    layout: String,
    llama: LlamaSettings,
    #[serde(default)]
    gguf: GgufInfo,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct GgufInfo {
    /// `Q8_0`, `Q4_0`, ... (the GGUF's file type).
    #[serde(default)]
    quantization: String,
}

/// Where llama.cpp is installed.
#[derive(Debug, Clone)]
pub struct Libraries {
    /// libllama, libggml, libggml-base and the CPU backends (Metal too on macOS).
    pub dir: PathBuf,
    /// The CUDA backend (`libggml-cuda.so` in the CUDA pack), when installed.
    pub cuda: Option<PathBuf>,
}

/// Which device to load the model on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Target {
    /// The first GPU llama.cpp finds, else the CPU; a GPU that cannot hold the model falls back
    /// to the CPU.
    Auto,
    Cpu,
    /// A named llama.cpp device (`CUDA0`), with no fallback.
    Device(String),
}

enum Layout {
    LlmLogits {
        cfg: Box<LlmLogitsConfig>,
        /// Tokens of the template pieces and the system message, fixed per model.
        pre: Vec<Token>,
        system: Vec<Token>,
        mid: Vec<Token>,
        post: Vec<Token>,
        bos: Option<Token>,
    },
    Winnow(WinnowConfig),
}

/// One question, ready to evaluate.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The prompt's token ids.
    pub ids: Vec<Token>,
    /// The split point: `ids[..p]` is the state prefix, evaluated as one cold pass.
    pub p: usize,
    /// The option labels' tokens, in prompt order.
    pub candidates: Vec<Token>,
    /// Prompt position of each wire option.
    pub wire_order: Vec<usize>,
}

/// A request's questions as the engine evaluates them.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoded {
    /// `(question id, row)` in request order.
    pub rows: Vec<(String, Row)>,
    /// The rendered state's length in tokens, before truncation.
    pub state_tokens: usize,
    pub state_truncated: bool,
}

struct Context {
    ptr: NonNull<c_void>,
    memory: *mut c_void,
    /// The cold prefix the cache holds at positions `0..len` (possibly followed by the previous
    /// question's tail), or `None` when unknown.
    prefix: Option<Vec<Token>>,
}

/// The model and its context; freed together.
struct Handles {
    api: &'static Api,
    model: NonNull<c_void>,
    context: Mutex<Context>,
}

impl Drop for Handles {
    fn drop(&mut self) {
        let ctx = self.context.get_mut().map_or(None, |c| Some(c.ptr));
        // SAFETY: both were created for this model and are freed once, context first.
        unsafe {
            if let Some(ctx) = ctx {
                (self.api.llama_free)(ctx.as_ptr());
            }
            (self.api.llama_model_free)(self.model.as_ptr());
        }
    }
}

/// The GGUF's tokenizer.
struct Vocab {
    api: &'static Api,
    ptr: *const c_void,
    n_tokens: usize,
}

/// A GGUF model loaded in llama.cpp.
pub struct LlamaModel {
    handles: Handles,
    vocab: Vocab,
    layout: Layout,
    settings: LlamaSettings,
    /// `cuda:0`, `metal` or `cpu`.
    pub device: String,
    /// The GGUF's type (`Q8_0`).
    pub precision: String,
}

// SAFETY: the model and vocabulary are read-only after loading; the context and its memory are
// only used under the `context` mutex.
unsafe impl Send for LlamaModel {}
unsafe impl Sync for LlamaModel {}

fn model_error(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

/// llama.cpp's log, through `tracing` (warnings and errors as such, the rest at debug level).
unsafe extern "C" fn log(level: c_int, text: *const c_char, _: *mut c_void) {
    // SAFETY: llama.cpp passes a NUL-terminated message.
    let text = unsafe { ffi::cstr(text) };
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    match level {
        ffi::LOG_LEVEL_ERROR => tracing::error!(target: "llama.cpp", "{text}"),
        ffi::LOG_LEVEL_WARN => tracing::warn!(target: "llama.cpp", "{text}"),
        _ => tracing::debug!(target: "llama.cpp", "{text}"),
    }
}

/// The process-wide llama.cpp: loaded once, backends registered once.
fn api(libs: &Libraries, want_gpu: bool) -> Result<&'static Api, Error> {
    static API: std::sync::OnceLock<Result<&'static Api, String>> = std::sync::OnceLock::new();
    API.get_or_init(|| {
        let api: &'static Api = Box::leak(Box::new(Api::load(&libs.dir)?));
        // SAFETY: plain calls into the loaded library, before any model exists.
        unsafe {
            (api.llama_log_set)(log, std::ptr::null_mut());
            (api.llama_backend_init)();
            // The CPU backend picks the best build for this CPU from the variants in the
            // directory (Linux, Windows); macOS links its backends into libggml.
            if !cfg!(target_os = "macos") {
                let dir = cpath(&libs.dir)?;
                (api.ggml_backend_load_all_from_path)(dir.as_ptr());
            }
            if want_gpu && let Some(cuda) = &libs.cuda {
                let path = cpath(cuda)?;
                if (api.ggml_backend_load)(path.as_ptr()).is_null() {
                    tracing::warn!("could not load {} (see the llama.cpp log)", cuda.display());
                }
            }
        }
        Ok(api)
    })
    .clone()
    .map_err(Error::Model)
}

fn cpath(p: &Path) -> Result<CString, String> {
    CString::new(p.to_string_lossy().as_bytes()).map_err(|_| format!("bad path {}", p.display()))
}

/// A llama.cpp GPU: its handle, name (`CUDA0`) and description.
struct Gpu {
    handle: ffi::Device,
    name: String,
    description: String,
    free_mib: usize,
}

/// The CUDA backend's file name (in the CUDA pack, next to the CUDA libraries it links).
pub const CUDA_BACKEND: &str = if cfg!(windows) {
    "ggml-cuda.dll"
} else {
    "libggml-cuda.so"
};

/// Load llama.cpp as a runner would and list what it finds: its version and every device,
/// without loading a model. The release smoke test and bug reports use it
/// (`ollaya llama-devices`).
pub fn probe(libs: &Libraries) -> Result<Value, Error> {
    let api = api(libs, true)?;
    // SAFETY: llama_version returns a static string; the registry is initialized and device
    // handles stay valid for the process.
    let (version, devices) = unsafe {
        let version = ffi::cstr((api.llama_version)());
        let devices: Vec<Value> = (0..(api.ggml_backend_dev_count)())
            .map(|i| {
                let d = (api.ggml_backend_dev_get)(i);
                let (mut free, mut total) = (0usize, 0usize);
                (api.ggml_backend_dev_memory)(d, &mut free, &mut total);
                let kind = match (api.ggml_backend_dev_type)(d) {
                    ffi::DEVICE_TYPE_GPU => "gpu",
                    ffi::DEVICE_TYPE_IGPU => "igpu",
                    0 => "cpu",
                    _ => "accel",
                };
                serde_json::json!({
                    "name": ffi::cstr((api.ggml_backend_dev_name)(d)),
                    "description": ffi::cstr((api.ggml_backend_dev_description)(d)),
                    "type": kind,
                    "memory_free_mib": free >> 20,
                    "memory_total_mib": total >> 20,
                })
            })
            .collect();
        (version, devices)
    };
    Ok(serde_json::json!({
        "llama_cpp": version,
        "libraries": libs.dir,
        "cuda_backend": libs.cuda,
        "devices": devices,
    }))
}

fn gpus(api: &Api) -> Vec<Gpu> {
    // SAFETY: the registry is initialized; device handles stay valid for the process.
    unsafe {
        (0..(api.ggml_backend_dev_count)())
            .map(|i| (api.ggml_backend_dev_get)(i))
            .filter(|&d| {
                matches!(
                    (api.ggml_backend_dev_type)(d),
                    ffi::DEVICE_TYPE_GPU | ffi::DEVICE_TYPE_IGPU
                )
            })
            .map(|d| {
                let (mut free, mut total) = (0usize, 0usize);
                (api.ggml_backend_dev_memory)(d, &mut free, &mut total);
                Gpu {
                    handle: d,
                    name: ffi::cstr((api.ggml_backend_dev_name)(d)),
                    description: ffi::cstr((api.ggml_backend_dev_description)(d)),
                    free_mib: free >> 20,
                }
            })
            .collect()
    }
}

/// Ollaya's name for a llama.cpp device: `CUDA0` → `cuda:0`, `MTL0` → `metal`.
pub fn device_name(dev: &str) -> String {
    if let Some(n) = dev.strip_prefix("CUDA") {
        format!("cuda:{n}")
    } else if dev.starts_with("MTL") {
        "metal".into()
    } else {
        dev.to_ascii_lowercase()
    }
}

/// Threads for the CPU backend: about the physical cores, as llama-server's default. SMT
/// siblings slow llama.cpp down, and its numbers do not depend on the thread count.
fn default_threads() -> i32 {
    let logical = std::thread::available_parallelism().map_or(4, |n| n.get());
    let physical = if cfg!(target_arch = "x86_64") {
        (logical / 2).max(1)
    } else {
        logical
    };
    physical as i32
}

impl Vocab {
    /// `llama_tokenize`: `add_special` adds BOS when the vocabulary does; `parse_special` turns
    /// control-token text into control tokens.
    fn tokenize(
        &self,
        text: &str,
        add_special: bool,
        parse_special: bool,
    ) -> Result<Vec<Token>, Error> {
        let len = i32::try_from(text.len()).map_err(|_| model_error("text too long"))?;
        let mut out: Vec<Token> = vec![0; text.len() + 2];
        for _ in 0..2 {
            // SAFETY: `text` is `len` bytes; `out` holds `out.len()` tokens.
            let n = unsafe {
                (self.api.llama_tokenize)(
                    self.ptr,
                    text.as_ptr().cast(),
                    len,
                    out.as_mut_ptr(),
                    out.len() as i32,
                    add_special,
                    parse_special,
                )
            };
            if n >= 0 {
                out.truncate(n as usize);
                return Ok(out);
            }
            if n == i32::MIN {
                break;
            }
            out.resize(n.unsigned_abs() as usize, 0);
        }
        Err(model_error("llama_tokenize failed"))
    }

    fn piece(&self, t: Token, buf: &mut Vec<u8>) -> usize {
        for _ in 0..2 {
            // SAFETY: `buf` holds `buf.len()` bytes; a negative result is the size needed.
            let n = unsafe {
                (self.api.llama_token_to_piece)(
                    self.ptr,
                    t,
                    buf.as_mut_ptr().cast(),
                    buf.len() as i32,
                    0,
                    true,
                )
            };
            if n >= 0 {
                return n as usize;
            }
            buf.resize(n.unsigned_abs() as usize, 0);
        }
        0
    }

    /// The text of `tokens`: their pieces concatenated, special tokens rendered, as
    /// llama-server's `/detokenize` does. Bytes that are not UTF-8 (a character cut in two)
    /// become U+FFFD.
    fn detokenize(&self, tokens: &[Token]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        let mut buf = vec![0u8; 64];
        for &t in tokens {
            let n = self.piece(t, &mut buf);
            bytes.extend_from_slice(&buf[..n]);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Cut a rendered state to `max` tokens (the text of the first `max`), as the references do.
    fn cut(
        &self,
        text: String,
        max: usize,
        parse_special: bool,
    ) -> Result<(String, usize, bool), Error> {
        let ids = self.tokenize(&text, false, parse_special)?;
        if ids.len() <= max {
            return Ok((text, ids.len(), false));
        }
        Ok((self.detokenize(&ids[..max]), ids.len(), true))
    }
}

/// Prepare a layout: tokenize its fixed pieces and check that the label table is this GGUF's.
fn prepare(vocab: &Vocab, decision: &Value) -> Result<Layout, Error> {
    let bad = |e: ollaya_decision::Error| model_error(e.to_string());
    let parse_error = |e: serde_json::Error| model_error(format!("decision.json: {e}"));
    let layout = match decision["layout"].as_str() {
        Some(llm_logits::LAYOUT) => {
            let cfg: LlmLogitsConfig =
                serde_json::from_value(decision.clone()).map_err(parse_error)?;
            cfg.validate().map_err(bad)?;
            let pre = vocab.tokenize(&cfg.template.pre, false, true)?;
            let mid = vocab.tokenize(&cfg.template.mid, false, true)?;
            let post = vocab.tokenize(&cfg.post(), false, true)?;
            let system = vocab.tokenize(&cfg.system, false, false)?;
            let x = vocab.tokenize("x", true, false)?;
            let bos = (x.len() == 2).then(|| x[0]);
            if bos.is_some() != cfg.add_bos {
                return Err(model_error(format!(
                    "the GGUF {} a BOS token, decision.json says the opposite",
                    if bos.is_some() { "adds" } else { "adds no" }
                )));
            }
            Layout::LlmLogits {
                cfg: Box::new(cfg),
                pre,
                system,
                mid,
                post,
                bos,
            }
        }
        Some(winnow::LAYOUT) => {
            let cfg: WinnowConfig =
                serde_json::from_value(decision.clone()).map_err(parse_error)?;
            cfg.validate().map_err(bad)?;
            Layout::Winnow(cfg)
        }
        other => {
            return Err(model_error(format!(
                "the llama engine cannot run layout {other:?}"
            )));
        }
    };
    let tables: Vec<&llm_logits::LabelTable> = match &layout {
        Layout::LlmLogits { cfg, .. } => {
            vec![&cfg.labels.choice, &cfg.labels.score, &cfg.labels.noul]
        }
        Layout::Winnow(cfg) => vec![&cfg.labels],
    };
    for table in tables {
        for (s, &id) in table.strings.iter().zip(&table.ids) {
            let got = vocab.tokenize(s, false, false)?;
            if got != [id as Token] || id as usize >= vocab.n_tokens {
                return Err(model_error(format!(
                    "label {s:?} is {got:?} in this GGUF, decision.json says [{id}]"
                )));
            }
        }
    }
    Ok(layout)
}

impl LlamaModel {
    /// Load `gguf` with the layout of its `decision.json`, on `target`.
    pub fn load(
        gguf: &Path,
        decision: &Path,
        libs: &Libraries,
        target: &Target,
        threads: Option<usize>,
    ) -> Result<Self, Error> {
        let text = std::fs::read_to_string(decision)
            .map_err(|e| model_error(format!("{}: {e}", decision.display())))?;
        let value: Value = serde_json::from_str(&text)
            .map_err(|e| model_error(format!("{}: {e}", decision.display())))?;
        let d: Decision = serde_json::from_value(value.clone())
            .map_err(|e| model_error(format!("decision.json: {e}")))?;
        if d.engine != "llama" {
            return Err(model_error(format!(
                "decision.json: engine {:?} is not llama",
                d.engine
            )));
        }
        if !LAYOUTS.contains(&d.layout.as_str()) {
            return Err(model_error(format!(
                "this version of ollaya cannot run layout {:?} (supported: {}); upgrade ollaya",
                d.layout,
                LAYOUTS.join(", ")
            )));
        }
        if !matches!(d.llama.plan.as_str(), "prefix" | "cold") || d.llama.n_ctx == 0 {
            return Err(model_error(format!(
                "decision.json: bad llama settings {:?}",
                d.llama
            )));
        }
        let api = api(libs, *target != Target::Cpu)?;
        let found = gpus(api);
        let pick = match target {
            Target::Cpu => None,
            Target::Auto => found.first(),
            Target::Device(name) => {
                Some(found.iter().find(|g| &g.name == name).ok_or_else(|| {
                    let names: Vec<&str> = found.iter().map(|g| g.name.as_str()).collect();
                    model_error(format!(
                        "llama.cpp finds no device {name} (devices: {names:?}){}",
                        if libs.cuda.is_none() && name.starts_with("CUDA") {
                            "; the CUDA libraries are not installed"
                        } else {
                            ""
                        }
                    ))
                })?)
            }
        };
        let threads = threads.map_or_else(default_threads, |t| t as i32);
        let (handles, device) = match pick {
            Some(gpu) => {
                tracing::info!(device = %gpu.name, description = %gpu.description,
                    free_mib = gpu.free_mib, "loading on the GPU");
                match load_on(api, gguf, &d.llama, Some(gpu.handle), threads) {
                    Ok(h) => (h, device_name(&gpu.name)),
                    Err(e) if *target == Target::Auto => {
                        tracing::warn!("{e}; loading on the CPU instead");
                        (load_on(api, gguf, &d.llama, None, threads)?, "cpu".into())
                    }
                    Err(e) => return Err(e),
                }
            }
            None => (load_on(api, gguf, &d.llama, None, threads)?, "cpu".into()),
        };
        // SAFETY: the model is loaded; its vocabulary lives as long as it.
        let vocab = unsafe {
            let ptr = (api.llama_model_get_vocab)(handles.model.as_ptr());
            Vocab {
                api,
                ptr,
                n_tokens: (api.llama_vocab_n_tokens)(ptr).max(0) as usize,
            }
        };
        let layout = prepare(&vocab, &value)?;
        let model = LlamaModel {
            handles,
            vocab,
            layout,
            settings: d.llama.clone(),
            device,
            precision: d.gguf.quantization.clone(),
        };
        tracing::info!(model = %model.describe(), device = %model.device, layout = %d.layout, "loaded");
        Ok(model)
    }

    /// `llama_model_desc`: architecture, size and type (`gemma4 12B Q8_0`).
    fn describe(&self) -> String {
        let mut buf = [0 as c_char; 128];
        // SAFETY: the buffer is 128 bytes; llama.cpp NUL-terminates within it.
        unsafe {
            (self.handles.api.llama_model_desc)(
                self.handles.model.as_ptr(),
                buf.as_mut_ptr(),
                buf.len(),
            );
            ffi::cstr(buf.as_ptr())
        }
    }

    /// Build and tokenize every question's prompt; nothing is evaluated. A request the layout
    /// rejects is an error before any question runs.
    pub fn encode(&self, state: &Value, questions: &Value) -> Result<Encoded, Error> {
        let vocab = &self.vocab;
        match &self.layout {
            Layout::LlmLogits {
                cfg,
                pre,
                system,
                mid,
                post,
                bos,
            } => {
                let qs = questions.as_object().ok_or_else(|| {
                    ollaya_decision::Error::invalid("questions must be an object")
                })?;
                let (state_text, state_tokens, state_truncated) =
                    vocab.cut(llm_logits::render_state(state), cfg.max_state_tokens, false)?;
                // Every prompt is built (and validated) before any is evaluated.
                let prompts = qs
                    .iter()
                    .map(|(qid, def)| Ok((qid, cfg.question(qid, def, &state_text)?)))
                    .collect::<Result<Vec<_>, ollaya_decision::Error>>()?;
                // pre ⧺ system ⧺ mid ⧺ user ⧺ post, with BOS first when the GGUF adds one and
                // the template did not emit it.
                let ids = |user: &str| -> Result<Vec<Token>, Error> {
                    let user = vocab.tokenize(user, false, false)?;
                    let mut ids = Vec::with_capacity(pre.len() + system.len() + user.len() + 64);
                    for part in [pre, system, mid, &user, post] {
                        ids.extend_from_slice(part);
                    }
                    if let Some(b) = *bos
                        && ids.first() != Some(&b)
                    {
                        ids.insert(0, b);
                    }
                    Ok(ids)
                };
                let reference = ids(&llm_logits::reference_message(&state_text))?;
                let mut rows = Vec::with_capacity(prompts.len());
                for (qid, q) in prompts {
                    let ids = ids(&q.user)?;
                    let p = split_point(&ids, &reference);
                    rows.push((
                        qid.clone(),
                        Row {
                            ids,
                            p,
                            candidates: q.label_ids.iter().map(|&t| t as Token).collect(),
                            wire_order: q.wire_order,
                        },
                    ));
                }
                Ok(Encoded {
                    rows,
                    state_tokens,
                    state_truncated,
                })
            }
            Layout::Winnow(cfg) => {
                let state_text = winnow::state_text(state)?;
                let prompts = cfg.questions(questions)?;
                let (state_text, state_tokens, state_truncated) =
                    vocab.cut(state_text, cfg.max_state_tokens, true)?;
                let pre = vocab.tokenize(&winnow::prefix(&state_text), true, true)?;
                let mut rows = Vec::with_capacity(prompts.len());
                for (qid, q) in prompts {
                    let mut ids = pre.clone();
                    ids.extend(vocab.tokenize(&q.suffix, false, true)?);
                    rows.push((
                        qid,
                        Row {
                            ids,
                            p: pre.len(),
                            wire_order: (0..q.label_ids.len()).collect(),
                            candidates: q.label_ids.iter().map(|&t| t as Token).collect(),
                        },
                    ));
                }
                Ok(Encoded {
                    rows,
                    state_tokens,
                    state_truncated,
                })
            }
        }
    }

    /// Every question's option logits (wire order), through the fixed evaluation plan.
    pub fn evaluate(&self, enc: &Encoded) -> Result<Vec<Vec<f32>>, Error> {
        let n_ctx = self.settings.n_ctx;
        for (qid, row) in &enc.rows {
            if row.ids.len() >= n_ctx {
                return Err(ollaya_decision::Error::invalid(format!(
                    "question {qid:?}: the prompt is {} tokens, and the model's context holds \
                     {n_ctx}; shorten the question or its options",
                    row.ids.len()
                ))
                .into());
            }
        }
        let mut ctx = self
            .handles
            .context
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let mut out = Vec::with_capacity(enc.rows.len());
        for (_, row) in &enc.rows {
            if row.candidates.len() == 1 {
                out.push(vec![0.0]);
                continue;
            }
            let p = if self.settings.plan == "cold" {
                0
            } else {
                row.p
            };
            let z = match self.ask(&mut ctx, &row.ids, p, &row.candidates) {
                Ok(z) => z,
                Err(e) => {
                    // The cache is in an unknown state: start from nothing next time.
                    ctx.prefix = None;
                    return Err(e);
                }
            };
            out.push(row.wire_order.iter().map(|&j| z[j]).collect());
        }
        Ok(out)
    }

    /// The candidates' logits at the last token of `ids`, evaluated as `ids[..p]` cold, then
    /// `ids[p..]` as one batch.
    fn ask(
        &self,
        ctx: &mut Context,
        ids: &[Token],
        p: usize,
        candidates: &[Token],
    ) -> Result<Vec<f32>, Error> {
        let api = self.handles.api;
        if p == 0 {
            self.clear(ctx);
            self.decode(ctx, ids, 0, true)?;
        } else {
            let pre = &ids[..p];
            if ctx.prefix.as_deref() == Some(pre) {
                // Drop the previous question's tail; the prefix stays.
                // SAFETY: the memory belongs to this context.
                let ok = unsafe { (api.llama_memory_seq_rm)(ctx.memory, 0, p as i32, -1) };
                if !ok {
                    return Err(model_error(
                        "llama.cpp could not cut the cache back to the prefix",
                    ));
                }
            } else {
                self.clear(ctx);
                self.decode(ctx, pre, 0, false)?;
                ctx.prefix = Some(pre.to_vec());
            }
            self.decode(ctx, &ids[p..], p, true)?;
        }
        // SAFETY: the last decode output the logits of its last token: n_vocab floats.
        let logits = unsafe {
            let row = (api.llama_get_logits_ith)(ctx.ptr.as_ptr(), -1);
            if row.is_null() {
                return Err(model_error("llama.cpp returned no logits"));
            }
            std::slice::from_raw_parts(row, self.vocab.n_tokens)
        };
        candidates
            .iter()
            .map(|&c| {
                logits.get(c as usize).copied().ok_or_else(|| {
                    model_error(format!("label token {c} is outside the vocabulary"))
                })
            })
            .collect()
    }

    fn clear(&self, ctx: &mut Context) {
        // SAFETY: the memory belongs to this context.
        unsafe { (self.handles.api.llama_memory_clear)(ctx.memory, true) };
        ctx.prefix = None;
    }

    /// Evaluate `tokens` at positions `start..` of sequence 0, in `N_BATCH`-token calls (as
    /// llama-server does), with the logits of the very last token when `last` is set.
    fn decode(
        &self,
        ctx: &mut Context,
        tokens: &[Token],
        start: usize,
        last: bool,
    ) -> Result<(), Error> {
        let mut seq0: [ffi::SeqId; 1] = [0];
        let n_chunks = tokens.len().div_ceil(N_BATCH);
        for (i, chunk) in tokens.chunks(N_BATCH).enumerate() {
            let n = chunk.len();
            let offset = start + i * N_BATCH;
            let mut token = chunk.to_vec();
            let mut pos: Vec<ffi::Pos> = (0..n).map(|k| (offset + k) as ffi::Pos).collect();
            let mut n_seq_id = vec![1i32; n];
            let mut seq_id: Vec<*mut ffi::SeqId> = vec![seq0.as_mut_ptr(); n];
            let mut logits = vec![0i8; n];
            if last && i + 1 == n_chunks {
                logits[n - 1] = 1;
            }
            let batch = Batch {
                n_tokens: n as i32,
                token: token.as_mut_ptr(),
                embd: std::ptr::null_mut(),
                pos: pos.as_mut_ptr(),
                n_seq_id: n_seq_id.as_mut_ptr(),
                seq_id: seq_id.as_mut_ptr(),
                logits: logits.as_mut_ptr(),
            };
            // SAFETY: every array holds `n` entries and outlives the call.
            let rc = unsafe { (self.handles.api.llama_decode)(ctx.ptr.as_ptr(), batch) };
            if rc != 0 {
                return Err(model_error(match rc {
                    1 => "llama.cpp: the context is full".to_owned(),
                    rc => format!("llama_decode failed ({rc}); see the llama.cpp log"),
                }));
            }
        }
        Ok(())
    }

    /// Encode and evaluate a request: the runner contract's output.
    pub fn run_json(&self, state: &Value, questions: &Value) -> Result<Output, Error> {
        let enc = self.encode(state, questions)?;
        let logits = self.evaluate(&enc)?;
        Ok(Output {
            questions: logits
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: enc.rows.iter().map(|(_, r)| r.ids.len()).sum(),
            state_tokens: enc.state_tokens,
            state_truncated: enc.state_truncated,
        })
    }
}

/// Load the model and create its context: on `device`, or on the CPU (`None`).
fn load_on(
    api: &'static Api,
    gguf: &Path,
    settings: &LlamaSettings,
    device: Option<ffi::Device>,
    threads: i32,
) -> Result<Handles, Error> {
    let path = cpath(gguf).map_err(Error::Model)?;
    // NULL-terminated device list: the one GPU, or none (no offload).
    let mut devices: Vec<ffi::Device> = device.into_iter().chain([std::ptr::null_mut()]).collect();
    // SAFETY: default parameters with the device list, which outlives the call.
    let model = unsafe {
        let mut mp = (api.llama_model_default_params)();
        mp.devices = devices.as_mut_ptr();
        mp.n_gpu_layers = if device.is_some() { -1 } else { 0 };
        (api.llama_model_load_from_file)(path.as_ptr(), mp)
    };
    let model = NonNull::new(model).ok_or_else(|| {
        model_error(format!(
            "llama.cpp could not load {} (see the log{})",
            gguf.display(),
            if device.is_some() {
                "; not enough GPU memory?"
            } else {
                ""
            }
        ))
    })?;
    // SAFETY: the model is loaded; on failure it is freed before returning.
    let ctx = unsafe {
        let mut cp = (api.llama_context_default_params)();
        cp.n_ctx = settings.n_ctx as u32;
        cp.n_batch = N_BATCH as u32;
        cp.n_ubatch = N_UBATCH;
        cp.n_seq_max = 1;
        cp.n_threads = threads;
        cp.n_threads_batch = threads;
        cp.swa_full = settings.swa_full;
        cp.kv_unified = false;
        cp.no_perf = true;
        let ctx = (api.llama_init_from_model)(model.as_ptr(), cp);
        if ctx.is_null() {
            (api.llama_model_free)(model.as_ptr());
        }
        ctx
    };
    let ctx = NonNull::new(ctx).ok_or_else(|| {
        model_error(format!(
            "llama.cpp could not create a {}-token context (see the log)",
            settings.n_ctx
        ))
    })?;
    // SAFETY: the context was just created.
    let (n, memory) = unsafe {
        (
            (api.llama_n_ctx)(ctx.as_ptr()) as usize,
            (api.llama_get_memory)(ctx.as_ptr()),
        )
    };
    if n < settings.n_ctx {
        tracing::warn!(
            want = settings.n_ctx,
            got = n,
            "llama.cpp shortened the context"
        );
    }
    Ok(Handles {
        api,
        model,
        context: Mutex::new(Context {
            ptr: ctx,
            memory,
            prefix: None,
        }),
    })
}

/// A question's split point: the tokens it shares with a reference prompt for the same state,
/// leaving at least one token to evaluate.
pub fn split_point(ids: &[Token], reference: &[Token]) -> usize {
    let lcp = ids
        .iter()
        .zip(reference)
        .take_while(|(a, b)| a == b)
        .count();
    lcp.min(ids.len().saturating_sub(1))
}

impl crate::engine::Engine for LlamaModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        // The llama layouts read the questions as the daemon sent them.
        let raw: Map<String, Value> = questions
            .iter()
            .map(|(k, q)| (k.clone(), q.definition.clone()))
            .collect();
        self.run_json(state, &Value::Object(raw))
    }

    fn run_json(&self, state: &Value, questions: &Value) -> Result<Output, Error> {
        LlamaModel::run_json(self, state, questions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_point_leaves_a_token_to_evaluate() {
        assert_eq!(split_point(&[1, 2, 3], &[1, 2, 9]), 2);
        assert_eq!(split_point(&[1, 2, 3], &[1, 2, 3, 4]), 2);
        assert_eq!(split_point(&[5], &[1]), 0);
    }

    #[test]
    fn device_names() {
        assert_eq!(device_name("CUDA1"), "cuda:1");
        assert_eq!(device_name("MTL0"), "metal");
        assert_eq!(device_name("Vulkan0"), "vulkan0");
    }

    #[test]
    fn settings_default_to_the_prefix_plan() {
        let s: LlamaSettings = serde_json::from_str(r#"{"n_ctx": 8192}"#).unwrap();
        assert_eq!(
            s,
            LlamaSettings {
                n_ctx: 8192,
                swa_full: false,
                plan: "prefix".into()
            }
        );
    }
}
