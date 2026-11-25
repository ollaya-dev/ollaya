//! The model library the Discover page lists, from ollaya.dev, and the newer release the status
//! bar offers. Both are the website's: the app works without them.

use std::time::Duration;

use serde::Deserialize;

/// Where the model library is listed: the website's index for this app (site/src/build.tsx).
/// `OLLAYA_LIBRARY_URL` points at another one, such as a local build of the site.
const LIBRARY_URL: &str = "https://ollaya.dev/data/library.json";

/// Where a newer release is downloaded (the page picks the right installer for this system).
pub const DOWNLOAD_URL: &str = "https://ollaya.dev/download";

/// One family of the library: `kev`, with its versions (`kev:latest`, `kev:9b`…).
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct LibModel {
    pub name: String,
    pub title: String,
    pub description: String,
    pub publisher: Option<String>,
    pub license: String,
    pub caps: Vec<String>,
    /// Editorial order, lower first.
    pub rank: f64,
    /// Measured on the benchmark GPU.
    pub stats: Option<Stats>,
    pub tags: Vec<LibTag>,
    /// The model card as Markdown, its figures filled in (site/src/build.tsx). A site older than
    /// this app has none: the page links to the card on ollaya.dev instead.
    pub card: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct LibTag {
    /// The full name: `kev:9b`.
    pub name: String,
    pub summary: Option<String>,
    pub size: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, rename_all = "camelCase")]
pub struct Stats {
    pub accuracy: Option<f64>,
    pub latency_ms: Option<f64>,
}

impl LibModel {
    /// The versions to offer: `<model>:latest` first (what `ollaya run <model>` uses), then the
    /// main tags. Precision variants (-fp16, -fp32) are for pinning, not for picking.
    pub fn main_tags(&self) -> Vec<String> {
        let latest = format!("{}:latest", self.name);
        let mut tags = vec![latest.clone()];
        tags.extend(
            self.tags
                .iter()
                .map(|t| t.name.clone())
                .filter(|n| *n != latest && !n.ends_with("-fp16") && !n.ends_with("-fp32")),
        );
        tags
    }

    pub fn tag(&self, name: &str) -> Option<&LibTag> {
        self.tags.iter().find(|t| t.name == name)
    }

    /// Whether the search `q` (lower case) finds this family.
    pub fn matches(&self, q: &str) -> bool {
        q.is_empty()
            || [
                self.name.as_str(),
                &self.title,
                &self.description,
                self.publisher.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .chain(self.caps.iter().map(String::as_str))
            .any(|s| s.to_lowercase().contains(q))
    }
}

#[derive(Deserialize)]
struct Index {
    models: Vec<LibModel>,
}

/// The library, in its editorial order.
pub async fn fetch() -> Result<Vec<LibModel>, String> {
    let url = std::env::var("OLLAYA_LIBRARY_URL").unwrap_or_else(|_| LIBRARY_URL.to_owned());
    let client = reqwest::Client::builder()
        .user_agent(concat!("ollaya-desktop/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(20))
        .build()
        .map_err(|e| e.to_string())?;
    let index: Index = client
        .get(url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;
    let mut models = index.models;
    models.sort_by(|a, b| a.rank.total_cmp(&b.rank));
    Ok(models)
}

/// The latest Ollaya release. GitHub redirects `/releases/latest` to `/releases/tag/v<version>`,
/// which `ollaya update` and the install scripts read too: no API call and no rate limit.
pub async fn latest_release() -> Option<String> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("ollaya-desktop/", env!("CARGO_PKG_VERSION")))
        .timeout(Duration::from_secs(10))
        .build()
        .ok()?;
    let resp = client
        .head("https://github.com/ollaya-dev/ollaya/releases/latest")
        .send()
        .await
        .ok()?;
    let location = resp
        .headers()
        .get(reqwest::header::LOCATION)?
        .to_str()
        .ok()?;
    let (_, tag) = location.rsplit_once("/releases/tag/")?;
    Some(tag.trim_start_matches('v').to_owned())
}

/// A link from a model card, as the browser should open it: relative links resolve against the
/// card's page on ollaya.dev, and only web links open (never a file or a custom scheme).
pub fn card_link(model: &str, href: &str) -> Option<String> {
    let page = format!("https://ollaya.dev/library/{model}");
    let url = if href.starts_with("https://") || href.starts_with("http://") {
        href.to_owned()
    } else if let Some(anchor) = href.strip_prefix('#') {
        format!("{page}#{anchor}")
    } else if href.starts_with("//") {
        format!("https:{href}")
    } else if href.starts_with('/') {
        format!("https://ollaya.dev{href}")
    } else if href.contains(':') {
        return None;
    } else {
        format!("https://ollaya.dev/library/{href}")
    };
    Some(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_library_reads_the_sites_index() {
        let json = r##"{"models":[
            {"name":"kev","title":"Kev","description":"Pointer head","publisher":"Jared Palmer",
             "license":"Apache-2.0","caps":["choice"],"rank":2,
             "stats":{"tag":"kev:4b","accuracy":0.81,"latencyMs":42.5},
             "tags":[{"name":"kev:4b","summary":"The default","size":1200,"params":"4B","languages":["en"]},
                     {"name":"kev:4b-fp16","summary":null,"size":600},
                     {"name":"kev:latest","summary":null,"size":1200}],
             "readme":"<p>html</p>"},
            {"name":"von","title":"Von","description":"","publisher":null,"license":"MIT",
             "caps":[],"rank":1,"stats":null,"tags":[],"readme":null,"card":"# Von"}]}"##;
        let mut models = serde_json::from_str::<Index>(json).unwrap().models;
        models.sort_by(|a, b| a.rank.total_cmp(&b.rank));
        assert_eq!(models[0].name, "von");
        assert_eq!(models[0].card.as_deref(), Some("# Von"));
        let kev = &models[1];
        assert_eq!(kev.stats.as_ref().unwrap().latency_ms, Some(42.5));
        assert_eq!(kev.main_tags(), ["kev:latest", "kev:4b"]);
        assert_eq!(kev.tag("kev:4b").unwrap().size, Some(1200));
        assert!(kev.matches("pointer") && kev.matches("jared") && kev.matches("choice"));
        assert!(!kev.matches("guard"));
    }

    #[test]
    fn card_links_open_on_the_website() {
        let l = |h| card_link("kev", h);
        assert_eq!(l("https://hf.co/x").as_deref(), Some("https://hf.co/x"));
        assert_eq!(
            l("/docs/api").as_deref(),
            Some("https://ollaya.dev/docs/api")
        );
        assert_eq!(
            l("#speed").as_deref(),
            Some("https://ollaya.dev/library/kev#speed")
        );
        assert_eq!(
            l("decider").as_deref(),
            Some("https://ollaya.dev/library/decider")
        );
        assert_eq!(l("file:///etc/passwd"), None);
        assert_eq!(l("javascript:alert(1)"), None);
    }
}
