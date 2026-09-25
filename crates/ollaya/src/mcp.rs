//! `ollaya mcp`: the local decision models as Model Context Protocol tools, so an agent (Claude
//! Code, Claude Desktop, Cursor, …) can ask typed questions in milliseconds, with no API key and
//! no data leaving the machine.
//!
//! It is a thin layer over the daemon's HTTP API, reached like every other command (and started
//! when needed): `decide` returns exactly what `POST /v1/systemone` returns. Transport is stdio by
//! default, or streamable HTTP with `--http`.

use std::collections::HashMap;

use futures_util::StreamExt;
use ollaya_api::{Client, PullRequest, SystemOneRequest};
use rmcp::handler::server::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{
    CallToolResult, ContentBlock, Implementation, ListResourcesResult, PaginatedRequestParams,
    ProgressNotificationParam, ReadResourceRequestParams, ReadResourceResponse, ReadResourceResult,
    RequestMetaObject, Resource, ResourceContents, ServerCapabilities, ServerConfig,
};
use rmcp::service::RequestContext;
use rmcp::{
    ErrorData, Peer, RoleServer, ServerHandler, ServiceExt, tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

use ollaya_api::presets;

use crate::daemon;

const INSTRUCTIONS: &str = "Ollaya runs decision models locally. Call `decide` whenever you need a \
typed judgement about a text or JSON state: classify it (choice), rate it (score) or check a \
yes/no statement (noul). It answers in milliseconds with calibrated probabilities, so it is cheap \
enough to run before every routing, triage, moderation or escalation step. Act on high \
confidence; escalate or ask when it is low. `laya` is the fastest general model, `decider` the \
most accurate. Resources list the installed models (ollaya://models) and the built-in question \
presets (ollaya://presets/<name>).";

/// The MCP server. Cheap to clone: every tool call reaches the daemon through its own client.
#[derive(Clone)]
pub struct OllayaMcp {
    tool_router: ToolRouter<Self>,
}

impl Default for OllayaMcp {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct DecideParams {
    /// The model: "laya" (default; routes English to laya:en and other languages to
    /// laya:multilingual), "decider" (most accurate), "nli", "gliclass", or a model created with a
    /// Modelfile.
    #[serde(default)]
    pub model: Option<String>,
    /// What to decide about: a string, or any JSON object or array (a ticket, an email, a message
    /// with metadata).
    pub state: Value,
    /// The questions, keyed by an id you choose. Each is an object with "type" ("choice", "score"
    /// or "noul"), "instructions" (the question, which may refer to fields of the state) and
    /// "criteria": for choice, an object of option → description (or a list of options); for
    /// score, a list of level descriptions from lowest to highest; for noul, optionally
    /// {"true": …, "false": …}. Omit it when you give `preset`, or for a model with built-in
    /// questions.
    #[serde(default)]
    pub questions: Option<Value>,
    /// A built-in question set to use instead of `questions`: "triage", "email", "guard",
    /// "moderation" or "router". Read ollaya://presets/<name> to see what each one asks.
    #[serde(default)]
    pub preset: Option<String>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct ModelParams {
    /// The model name, e.g. "laya", "laya:en" or "decider:0.8b".
    pub model: String,
}

#[tool_router]
impl OllayaMcp {
    pub fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
        }
    }

    /// Answer typed questions about a state with a local decision model.
    #[tool(
        name = "decide",
        description = "Ask typed questions about a text or JSON state and get calibrated answers \
from a local decision model in milliseconds. Returns, per question: for choice, the chosen option, \
its confidence and every option's probability; for score, the expected level and a legend; for \
noul, the probability that the statement is true. The response is exactly TypeSafe's \
/v1/systemone response.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn decide(
        &self,
        Parameters(p): Parameters<DecideParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let questions = match (p.questions, p.preset.as_deref()) {
            (Some(_), Some(_)) => {
                return Ok(tool_error("give either `questions` or `preset`, not both"));
            }
            (Some(q), None) => Some(q),
            (None, Some(name)) => match presets::get(name) {
                Some(q) => Some(q),
                None => {
                    return Ok(tool_error(&format!(
                        "unknown preset {name:?}; the presets are {}",
                        presets::NAMES.join(", ")
                    )));
                }
            },
            (None, None) => None,
        };
        let questions = match questions.map(serde_json::from_value).transpose() {
            Ok(q) => q,
            Err(e) => {
                return Ok(tool_error(&format!(
                    "`questions` is not a question set: {e}"
                )));
            }
        };
        let request = SystemOneRequest {
            model: p.model.unwrap_or_else(|| "laya".to_owned()),
            state: p.state,
            questions,
        };
        let client = connect().await?;
        match client.systemone(&request).await {
            Ok(response) => Ok(structured(&response)),
            Err(e) => Ok(tool_error(&e.to_string())),
        }
    }

    /// The models on this machine.
    #[tool(
        name = "list_models",
        description = "List the decision models installed on this machine, with their size and when they were pulled.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn list_models(&self) -> Result<CallToolResult, ErrorData> {
        let client = connect().await?;
        match client.tags().await {
            Ok(tags) => Ok(structured(&tags)),
            Err(e) => Ok(tool_error(&e.to_string())),
        }
    }

    /// A model's details.
    #[tool(
        name = "show_model",
        description = "Show a model's details: family, parameters, context length, languages, capabilities, license and built-in questions.",
        annotations(read_only_hint = true, open_world_hint = false)
    )]
    async fn show_model(
        &self,
        Parameters(p): Parameters<ModelParams>,
    ) -> Result<CallToolResult, ErrorData> {
        let client = connect().await?;
        match client.show(&p.model).await {
            Ok(show) => Ok(structured(&show)),
            Err(e) => Ok(tool_error(&e.to_string())),
        }
    }

    /// Download a model, reporting progress when the client asked for it.
    #[tool(
        name = "pull_model",
        description = "Download a model from the Ollaya library (for example \"laya\" or \"decider\"), verifying every file. Reports download progress. Models are also pulled on first use by `decide`.",
        annotations(
            read_only_hint = false,
            destructive_hint = false,
            idempotent_hint = true,
            open_world_hint = true
        )
    )]
    async fn pull_model(
        &self,
        Parameters(p): Parameters<ModelParams>,
        meta: RequestMetaObject,
        peer: Peer<RoleServer>,
    ) -> Result<CallToolResult, ErrorData> {
        let client = connect().await?;
        let request = PullRequest {
            model: p.model.clone(),
            insecure: false,
            stream: Some(true),
        };
        let mut stream = match client.pull_stream(&request).await {
            Ok(s) => s,
            Err(e) => return Ok(tool_error(&e.to_string())),
        };
        let token = meta.get_progress_token();
        // Bytes per layer, so progress covers the whole model rather than one file at a time.
        let mut layers: HashMap<String, (u64, u64)> = HashMap::new();
        while let Some(line) = stream.next().await {
            let line = match line {
                Ok(l) => l,
                Err(e) => return Ok(tool_error(&e.to_string())),
            };
            if let (Some(digest), Some(total)) = (&line.digest, line.total) {
                layers.insert(digest.clone(), (line.completed.unwrap_or(0), total));
            }
            if let Some(token) = &token {
                let (done, total) = layers
                    .values()
                    .fold((0, 0), |(d, t), (c, n)| (d + c, t + n));
                let mut progress = ProgressNotificationParam::new(token.clone(), done as f64)
                    .with_message(line.status.clone());
                if total > 0 {
                    progress = progress.with_total(total as f64);
                }
                let _ = peer.notify_progress(progress).await;
            }
        }
        Ok(structured(
            &json!({ "model": p.model, "status": "success" }),
        ))
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for OllayaMcp {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(
            ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_server_info(Implementation::new("ollaya", env!("CARGO_PKG_VERSION")))
        .with_instructions(INSTRUCTIONS)
    }

    async fn list_resources(
        &self,
        _request: Option<PaginatedRequestParams>,
        _context: RequestContext<RoleServer>,
    ) -> Result<ListResourcesResult, ErrorData> {
        let mut resources = vec![
            Resource::new(MODELS_URI, "models")
                .with_description(
                    "The decision models installed on this machine (like `ollaya list`).",
                )
                .with_mime_type("application/json"),
        ];
        for name in presets::NAMES {
            resources.push(
                Resource::new(format!("{PRESETS_URI}{name}"), format!("preset {name}"))
                    .with_description(format!("The questions of the built-in `{name}` preset."))
                    .with_mime_type("application/json"),
            );
        }
        Ok(ListResourcesResult::with_all_items(resources))
    }

    async fn read_resource(
        &self,
        request: ReadResourceRequestParams,
        _context: RequestContext<RoleServer>,
    ) -> Result<ReadResourceResponse, ErrorData> {
        let uri = request.uri.as_str();
        let value = if uri == MODELS_URI {
            let client = connect().await?;
            let tags = client
                .tags()
                .await
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?;
            serde_json::to_value(tags)
                .map_err(|e| ErrorData::internal_error(e.to_string(), None))?
        } else if let Some(q) = uri.strip_prefix(PRESETS_URI).and_then(presets::get) {
            q
        } else {
            return Err(ErrorData::resource_not_found(
                format!("no resource {uri}"),
                None,
            ));
        };
        let text = serde_json::to_string_pretty(&value).unwrap_or_default();
        Ok(ReadResourceResponse::Complete(ReadResourceResult::new(
            vec![ResourceContents::text(text, uri).with_mime_type("application/json")],
        )))
    }
}

const MODELS_URI: &str = "ollaya://models";
const PRESETS_URI: &str = "ollaya://presets/";

/// A client for the daemon, starting it when it is not running (as every CLI command does).
async fn connect() -> Result<Client, ErrorData> {
    daemon::client()
        .await
        .map_err(|e| ErrorData::internal_error(format!("{e:#}"), None))
}

fn structured(value: &impl serde::Serialize) -> CallToolResult {
    CallToolResult::structured(serde_json::to_value(value).unwrap_or(Value::Null))
}

/// A failed call the model should see and can correct (MCP tool errors, not protocol errors).
fn tool_error(message: &str) -> CallToolResult {
    CallToolResult::error(vec![ContentBlock::text(message.to_owned())])
}

/// `ollaya mcp`: serve over stdio until the client closes it.
pub async fn serve_stdio() -> anyhow::Result<()> {
    let service = OllayaMcp::new().serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}

/// `ollaya mcp --http ADDR`: serve streamable HTTP at http://ADDR/mcp until Ctrl-C.
pub async fn serve_http(addr: &str) -> anyhow::Result<()> {
    use rmcp::transport::streamable_http_server::session::local::LocalSessionManager;
    use rmcp::transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService,
    };

    let service: StreamableHttpService<OllayaMcp, LocalSessionManager> = StreamableHttpService::new(
        || Ok(OllayaMcp::new()),
        Default::default(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("Ollaya MCP server at http://{}/mcp", listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
