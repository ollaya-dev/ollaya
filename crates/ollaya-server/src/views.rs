//! Store and service state rendered as the contract's response types (`docs/api.md` §7–§8).

use std::time::SystemTime;

use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use ollaya_api::{
    Answer, DecideAnswer, DecideResponse, DoneReason, LayaExtra, LocalModel, ModelDetails,
    ModelMetadata, RouterInfo, Routing, RunningModel, ShowResponse, Usage,
};
use ollaya_registry::manifest::{ANNOTATION_PRECISION, Manifest};
use ollaya_registry::{Entry, ModelName, Store, media};
use serde_json::{Value, json};

use crate::Error;
use crate::models::{Resolved, resolve};
use crate::scheduler::RunningInfo;
use crate::service::{DecideOutput, ModelInfo};

fn utc(t: SystemTime) -> DateTime<Utc> {
    t.into()
}

/// `sha256:<hex>` -> `<hex>`, as Ollama prints model digests.
fn bare(digest: &str) -> String {
    digest.trim_start_matches("sha256:").to_owned()
}

fn config(store: &Store, manifest: &Manifest) -> Value {
    store
        .read_blob_json(&manifest.config)
        .unwrap_or(Value::Null)
}

fn text(v: &Value) -> String {
    v.as_str().unwrap_or_default().to_owned()
}

/// The precision a pinned model keeps (`PARAMETER precision`), if any.
fn pinned_precision(store: &Store, manifest: &Manifest) -> Option<String> {
    let params: Value = store.read_blob_json(manifest.layer(media::PARAMS)?).ok()?;
    params["precision"].as_str().map(str::to_owned)
}

fn precision_label(p: &str) -> String {
    match p {
        "fp16" => "F16".into(),
        "fp32" => "F32".into(),
        other => other.to_ascii_uppercase(),
    }
}

/// Precisions a model carries, GPU preference first: `F16/F32`, `F32`, ...; `""` for a router.
fn quantization(store: &Store, manifest: &Manifest) -> String {
    if let Some(p) = pinned_precision(store, manifest) {
        return precision_label(&p);
    }
    let mut ps: Vec<String> = manifest
        .layers_of(media::GRAPH_ONNX)
        .map(|g| {
            g.annotations
                .get(ANNOTATION_PRECISION)
                .cloned()
                .unwrap_or_else(|| "fp32".into())
        })
        .collect();
    ps.sort_by_key(|p| (p != "fp16", p.clone()));
    ps.dedup();
    ps.iter()
        .map(|p| precision_label(p))
        .collect::<Vec<_>>()
        .join("/")
}

pub fn details(store: &Store, manifest: &Manifest) -> ModelDetails {
    let c = config(store, manifest);
    let family = text(&c["family"]);
    let format = match text(&c["model_format"]) {
        f if f.is_empty() => "onnx".into(),
        f => f,
    };
    ModelDetails {
        parent_model: text(&c["parent_model"]),
        format,
        families: if family.is_empty() {
            vec![]
        } else {
            vec![family.clone()]
        },
        family,
        parameter_size: text(&c["parameter_size"]),
        quantization_level: quantization(store, manifest),
    }
}

pub fn local_model(store: &Store, e: &Entry) -> LocalModel {
    LocalModel {
        name: e.name.to_string(),
        model: e.name.to_string(),
        modified_at: utc(e.modified),
        size: e.manifest.total_size(),
        digest: bare(&e.digest),
        details: details(store, &e.manifest),
    }
}

pub fn model_metadata(store: &Store, e: &Entry) -> ModelMetadata {
    let c = config(store, &e.manifest);
    let release_date = c["release_date"]
        .as_str()
        .filter(|d| d.len() == 10)
        .map(str::to_owned)
        .unwrap_or_else(|| utc(e.modified).format("%Y-%m-%d").to_string());
    ModelMetadata {
        name: e.name.to_string(),
        description: text(&c["description"]),
        release_date,
    }
}

/// The decision layer (`decision.json`) of a model, if it has one.
fn decision(store: &Store, manifest: &Manifest) -> Option<Value> {
    store.read_blob_json(manifest.layer(media::DECISION)?).ok()
}

fn capabilities(decision: &Value) -> Vec<String> {
    let mut caps: Vec<String> = ["choice", "score", "noul"].map(String::from).to_vec();
    let has_act = decision["outputs"]
        .as_array()
        .is_some_and(|o| o.iter().any(|x| x == "act_logits"));
    if has_act {
        caps.push("act".into());
    }
    caps
}

pub fn show(store: &Store, info: &ModelInfo) -> ShowResponse {
    let e = &info.entry;
    let manifest = &e.manifest;
    let c = config(store, manifest);
    let family = text(&c["family"]);
    let license = manifest
        .layer(media::LICENSE)
        .and_then(|d| store.read_blob(d).ok())
        .map(|b| String::from_utf8_lossy(&b).into_owned())
        .unwrap_or_default();
    let precision = pinned_precision(store, manifest);
    let parameters = precision
        .as_ref()
        .map(|p| format!("precision {p}"))
        .unwrap_or_default();

    let mut model_info = IndexMap::new();
    model_info.insert("general.architecture".into(), json!(family));
    model_info.insert("general.languages".into(), c["languages"].clone());
    model_info.insert("general.source".into(), json!(text(&c["source"])));

    let (questions_json, router, capabilities) = match &info.resolved {
        Resolved::Model(m) => {
            let d = decision(store, manifest).unwrap_or(Value::Null);
            for (key, field) in [
                ("encoder", "encoder"),
                ("layout", "layout"),
                ("context_length", "max_len"),
                ("head_max_len", "head_max_len"),
            ] {
                if !d[field].is_null() {
                    model_info.insert(format!("{family}.{key}"), d[field].clone());
                }
            }
            (m.questions.clone(), None, capabilities(&d))
        }
        Resolved::Router {
            name,
            router,
            questions,
            ..
        } => {
            // Guaranteed whichever route is taken: the intersection of the targets'.
            let mut caps: Option<Vec<String>> = None;
            for target in router.routes.values() {
                let Ok(t) = ModelName::parse_relative(target, name) else {
                    continue;
                };
                if let Ok(Resolved::Model(m)) = resolve(store, &t)
                    && let Ok(Some(te)) = store.read_manifest(&m.name)
                {
                    let d = decision(store, &te.manifest).unwrap_or(Value::Null);
                    let tc = capabilities(&d);
                    caps = Some(match caps {
                        None => tc,
                        Some(prev) => prev.into_iter().filter(|x| tc.contains(x)).collect(),
                    });
                }
            }
            let info = RouterInfo {
                strategy: router.strategy.clone(),
                default: router.default.clone(),
                routes: router
                    .routes
                    .iter()
                    .map(|(k, v)| {
                        let v =
                            ModelName::parse_relative(v, name).map_or(v.clone(), |n| n.to_string());
                        (k.clone(), v)
                    })
                    .collect(),
            };
            (
                questions.clone(),
                Some(info),
                caps.unwrap_or_else(|| ["choice", "score", "noul"].map(String::from).to_vec()),
            )
        }
    };
    let questions = questions_json
        .as_ref()
        .and_then(|q| serde_json::from_value(q.clone()).ok());

    let parent = text(&c["parent_model"]);
    let from = if parent.is_empty() {
        e.name.to_string()
    } else {
        parent
    };
    let mut modelfile = format!(
        "# Modelfile generated by \"ollaya show\"\n# To build a new Modelfile based on this, replace FROM with:\n# FROM {}\n\nFROM {from}\n",
        e.name
    );
    if from != e.name.to_string() {
        if let Some(q) = &questions_json {
            let q = serde_json::to_string_pretty(q).unwrap_or_default();
            modelfile.push_str(&format!("QUESTIONS \"\"\"\n{q}\n\"\"\"\n"));
        }
        if let Some(p) = &precision {
            modelfile.push_str(&format!("PARAMETER precision {p}\n"));
        }
        let description = text(&c["description"]);
        if !description.is_empty() {
            modelfile.push_str(&format!("DESCRIPTION {description}\n"));
        }
    }

    ShowResponse {
        license,
        modelfile,
        parameters,
        questions,
        router,
        details: details(store, manifest),
        model_info,
        capabilities,
        modified_at: utc(e.modified),
    }
}

pub fn running_model(store: &Store, r: &RunningInfo) -> RunningModel {
    let entry = ModelName::parse(&r.name)
        .ok()
        .and_then(|n| store.read_manifest(&n).ok().flatten());
    let (mut details, context_length) = match &entry {
        Some(e) => {
            let c = config(store, &e.manifest);
            let ctx = decision(store, &e.manifest)
                .and_then(|d| d["max_len"].as_u64())
                .or_else(|| c["context_length"].as_u64())
                .unwrap_or(0);
            (details(store, &e.manifest), ctx)
        }
        None => (ModelDetails::default(), 0),
    };
    details.quantization_level = precision_label(&r.precision);
    let on_gpu = r.device.starts_with("cuda");
    RunningModel {
        name: r.name.clone(),
        model: r.name.clone(),
        size: r.size,
        digest: bare(&r.digest),
        details,
        expires_at: r.expires_at.map(utc),
        size_vram: if on_gpu { r.size } else { 0 },
        context_length,
        device: r.device.clone(),
    }
}

/// An `/api/decide` response from the service's output.
pub fn decide_response(out: DecideOutput, laya: bool) -> Result<DecideResponse, Error> {
    let render = |e: serde_json::Error| Error::Runner(format!("rendering answers: {e}"));
    let mut answers = IndexMap::with_capacity(out.answers.len());
    for ((id, q), a) in out.questions.iter().zip(&out.answers) {
        let extra = if laya {
            Some(LayaExtra::from_decision(a, q).map_err(render)?)
        } else {
            None
        };
        answers.insert(
            id.clone(),
            DecideAnswer {
                answer: Answer::from_decision(a, q).map_err(render)?,
                laya: extra,
            },
        );
    }
    let requested = out.requested;
    Ok(DecideResponse {
        model: out.model,
        answers,
        usage: Usage {
            input_tokens: out.input_tokens as u64,
            output_tokens: 0,
        },
        routing: out.routing.map(|r| Routing {
            router: requested,
            model: r.model,
            route: r.route,
            reason: r.reason,
        }),
        state_truncated: out.state_truncated,
        done_reason: DoneReason::Decide,
        created_at: Utc::now(),
        total_duration: out.total_duration.as_nanos() as u64,
        load_duration: out.load_duration.as_nanos() as u64,
        eval_duration: out.eval_duration.as_nanos() as u64,
    })
}
