//! What the daemon does, independent of the HTTP wire format: resolve, route, schedule, decide,
//! and manage the local model store.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ollaya_decision::{Answer, Questions, parse_questions};
use ollaya_registry::{Entry, ModelName, Progress, Puller, Store};
use serde_json::Value;
use tokio::sync::broadcast;

use crate::Error;
use crate::models::{Loadable, Resolved, resolve, resolve_entry};
use crate::scheduler::{KeepAlive, RunningInfo, Scheduler};

/// TypeSafe's limits, enforced for every model.
pub const MAX_CHOICE_OPTIONS: usize = 255;
pub const MAX_SCORE_LEVELS: usize = 10;

pub struct Ollaya {
    pub store: Store,
    puller: Puller,
    pub scheduler: Arc<Scheduler>,
    /// In-flight pulls by model name: a second pull of the same model joins the first.
    pulls: Mutex<HashMap<String, broadcast::Sender<PullEvent>>>,
    /// Model names a create is writing. Lock order: `pulls`, then `creating`.
    creating: Mutex<HashSet<String>>,
}

#[derive(Debug, Clone)]
pub enum PullEvent {
    Progress(Progress),
    Done,
    Failed(Arc<Error>),
}

#[derive(Debug, Clone)]
pub struct DecideInput {
    pub model: String,
    pub state: Value,
    /// Required unless the model has a baked-in question schema.
    pub questions: Option<Value>,
    pub keep_alive: Option<KeepAlive>,
    /// Set when the caller has gone away: once the model is loaded, the decision is skipped.
    pub cancel: Option<Arc<AtomicBool>>,
}

/// Which model answered and why, when the requested model is a router.
#[derive(Debug, Clone, PartialEq)]
pub struct Routing {
    pub model: String,
    /// The router's route key (`english`, `multilingual`).
    pub route: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct DecideOutput {
    /// The name the caller asked for, normalised (`laya` -> `laya:latest`).
    pub requested: String,
    /// The model that answered (differs from `requested` for routers).
    pub model: String,
    pub questions: Questions,
    pub answers: Vec<Answer>,
    pub input_tokens: usize,
    /// Tokens in the serialized state, before truncation.
    pub state_tokens: usize,
    /// The state was cut to fit the model's context for at least one question.
    pub state_truncated: bool,
    pub routing: Option<Routing>,
    pub load_duration: Duration,
    /// Time in the runner: tokenization, forward pass, calibration.
    pub eval_duration: Duration,
    pub total_duration: Duration,
}

/// A derived model: `FROM` plus baked-in layers (what a Modelfile describes).
#[derive(Debug, Clone, Default)]
pub struct CreateSpec {
    pub from: String,
    /// Question schema answered when a request brings none.
    pub questions: Option<Value>,
    /// Temperatures, in the `calibration` layer format.
    pub calibration: Option<Value>,
    /// Pin the graph precision: `fp16` or `fp32`.
    pub precision: Option<String>,
    pub license: Option<String>,
    pub description: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub entry: Entry,
    pub resolved: Resolved,
}

impl Ollaya {
    pub fn new(store: Store, scheduler: Arc<Scheduler>) -> Result<Arc<Self>, Error> {
        let puller = Puller::new(store.clone())?;
        Ok(Arc::new(Ollaya {
            store,
            puller,
            scheduler,
            pulls: Mutex::new(HashMap::new()),
            creating: Mutex::new(HashSet::new()),
        }))
    }

    pub async fn decide(&self, input: DecideInput) -> Result<DecideOutput, Error> {
        let started = Instant::now();
        let name = ModelName::parse(&input.model)?;
        let mut baked = None;
        let (model, routing) = match resolve(&self.store, &name)? {
            Resolved::Model(m) => (m, None),
            Resolved::Router {
                router, questions, ..
            } => {
                baked = questions;
                let default = router.default.as_str();
                let decision = ollaya_lang::route_with_default(&input.state, default);
                let route = decision.target.model(default);
                let target = router.routes.get(route).ok_or_else(|| {
                    Error::Corrupt(format!("{name}: router has no route {route:?}"))
                })?;
                let target = ModelName::parse_relative(target, &name)?;
                let resolved = match resolve(&self.store, &target) {
                    Err(Error::ModelNotFound(_)) => {
                        return Err(Error::RoutedModelNotFound {
                            target: target.to_string(),
                            router: name.to_string(),
                        });
                    }
                    other => other?,
                };
                match resolved {
                    Resolved::Model(m) => {
                        let routing = Routing {
                            model: target.to_string(),
                            route: route.to_owned(),
                            reason: decision.reason,
                        };
                        (m, Some(routing))
                    }
                    Resolved::Router { .. } => {
                        return Err(Error::Corrupt(format!(
                            "{name}: router routes to another router"
                        )));
                    }
                }
            }
        };
        // Request questions win; then the requested model's baked-in set; then the target's.
        let questions_json = match (input.questions, baked.or_else(|| model.questions.clone())) {
            (Some(q), _) => q,
            (None, Some(baked)) => baked,
            (None, None) => return Err(Error::NoQuestions(model.name.to_string())),
        };
        let questions =
            parse_questions(&questions_json).map_err(|e| Error::InvalidRequest(e.to_string()))?;
        check_limits(&questions)?;

        let (lease, load_duration) = self.scheduler.acquire(&model, input.keep_alive).await?;
        if input
            .cancel
            .as_ref()
            .is_some_and(|c| c.load(Ordering::SeqCst))
        {
            return Err(Error::Cancelled);
        }
        let eval_started = Instant::now();
        let raw = lease.decide(&input.state, &questions_json).await?;
        drop(lease);
        let state_tokens = raw["state_tokens"].as_u64().unwrap_or(0) as usize;
        if state_tokens > ollaya_api::MAX_STATE_TOKENS {
            return Err(Error::InputTooLong(state_tokens));
        }
        let answers = answers_from(&model, &questions, &raw, state_tokens)?;
        let eval_duration = eval_started.elapsed();
        Ok(DecideOutput {
            requested: name.to_string(),
            model: model.name.to_string(),
            questions,
            answers,
            input_tokens: raw["input_tokens"].as_u64().unwrap_or(0) as usize,
            state_tokens,
            state_truncated: raw["state_truncated"].as_bool().unwrap_or(false),
            routing,
            load_duration,
            eval_duration,
            total_duration: started.elapsed(),
        })
    }

    /// Load a model without answering anything (`ollaya run` warm-up), or unload it with
    /// `KeepAlive::For(0)`. Returns the time spent loading.
    pub async fn preload(
        &self,
        model: &str,
        keep_alive: Option<KeepAlive>,
    ) -> Result<Duration, Error> {
        let name = ModelName::parse(model)?;
        let targets: Vec<Box<Loadable>> = match resolve(&self.store, &name)? {
            Resolved::Model(m) => vec![m],
            Resolved::Router { router, .. } => router
                .routes
                .values()
                .filter_map(|t| ModelName::parse_relative(t, &name).ok())
                .filter_map(|t| match resolve(&self.store, &t) {
                    Ok(Resolved::Model(m)) => Some(m),
                    _ => None,
                })
                .collect(),
        };
        let mut loading = Duration::ZERO;
        for m in targets {
            if keep_alive == Some(KeepAlive::For(Duration::ZERO)) {
                self.scheduler.unload(&m.digest).await;
            } else {
                let (lease, took) = self.scheduler.acquire(&m, keep_alive).await?;
                drop(lease);
                loading += took;
            }
        }
        Ok(loading)
    }

    /// Pull a model, streaming progress. Concurrent pulls of one model share a single download;
    /// when every receiver is gone, the pull stops (partial blobs stay for the next pull).
    pub fn pull(self: &Arc<Self>, model: &str) -> Result<broadcast::Receiver<PullEvent>, Error> {
        let name = ModelName::parse(model)?;
        let key = name.to_string();
        let mut pulls = self.pulls.lock().unwrap();
        if let Some(tx) = pulls.get(&key) {
            return Ok(tx.subscribe());
        }
        if self.creating.lock().unwrap().contains(&key) {
            return Err(Error::Busy(key));
        }
        let (tx, rx) = broadcast::channel(1024);
        pulls.insert(key.clone(), tx.clone());
        drop(pulls);
        let this = self.clone();
        tokio::spawn(async move {
            let sender = tx.clone();
            let progress = move |p| drop(sender.send(PullEvent::Progress(p)));
            let abandoned = async {
                loop {
                    tokio::time::sleep(Duration::from_millis(250)).await;
                    if tx.receiver_count() == 0 {
                        break;
                    }
                }
            };
            let result = tokio::select! {
                r = this.puller.pull(&name, &progress) => Some(r),
                () = abandoned => None,
            };
            this.pulls.lock().unwrap().remove(&key);
            match result {
                Some(Ok(_)) => drop(tx.send(PullEvent::Done)),
                Some(Err(e)) => drop(tx.send(PullEvent::Failed(Arc::new(e.into())))),
                None => tracing::info!(model = %key, "pull stopped: no client is waiting for it"),
            }
        });
        Ok(rx)
    }

    /// Whether a pull or a create is writing `name` right now.
    pub fn is_busy(&self, name: &ModelName) -> bool {
        let key = name.to_string();
        let pulls = self.pulls.lock().unwrap();
        pulls.contains_key(&key) || self.creating.lock().unwrap().contains(&key)
    }

    pub fn list(&self) -> Result<Vec<Entry>, Error> {
        Ok(self.store.list()?)
    }

    pub fn show(&self, model: &str) -> Result<ModelInfo, Error> {
        let name = ModelName::parse(model)?;
        let entry = self
            .store
            .read_manifest(&name)?
            .ok_or_else(|| Error::ModelNotFound(name.to_string()))?;
        let resolved = resolve_entry(&self.store, entry.clone())?;
        Ok(ModelInfo { entry, resolved })
    }

    /// Remove a model; it is unloaded first. Returns false if it was not present.
    pub async fn delete(&self, model: &str) -> Result<bool, Error> {
        let name = ModelName::parse(model)?;
        if self.is_busy(&name) {
            return Err(Error::Busy(name.to_string()));
        }
        if let Some(entry) = self.store.read_manifest(&name)? {
            self.scheduler.unload(&entry.digest).await;
        }
        Ok(self.store.remove(&name)?)
    }

    /// Create `name` from `spec`. `FROM` must be local (the API never pulls implicitly). Only the
    /// replaced layers are new; everything else (graphs, weights) is shared with `FROM`.
    /// `progress` receives Ollama's create status lines.
    pub async fn create(
        &self,
        name: &str,
        spec: CreateSpec,
        progress: &(dyn Fn(String) + Send + Sync),
    ) -> Result<(), Error> {
        use ollaya_registry::{Descriptor, media};
        let name = ModelName::parse(name)?;
        let from = ModelName::parse(&spec.from)?;
        let _claim = {
            let key = name.to_string();
            let pulls = self.pulls.lock().unwrap();
            let mut creating = self.creating.lock().unwrap();
            if pulls.contains_key(&key) || !creating.insert(key.clone()) {
                return Err(Error::Busy(key));
            }
            Claim {
                set: &self.creating,
                key,
            }
        };
        let entry = self
            .store
            .read_manifest(&from)?
            .ok_or_else(|| Error::ModelNotFound(from.to_string()))?;
        let is_router = entry.manifest.layer(media::ROUTER).is_some();
        let mut manifest = entry.manifest;
        let inherited: HashSet<String> = manifest.layers.iter().map(|l| l.digest.clone()).collect();
        let blob = |media_type: &str, bytes: Vec<u8>| -> Result<Descriptor, Error> {
            let size = bytes.len() as u64;
            let digest = self.store.write_blob(&bytes)?;
            Ok(Descriptor {
                media_type: media_type.into(),
                digest,
                size,
                urls: vec![],
                annotations: Default::default(),
            })
        };
        let mut replace = |media_type: &str, bytes: Vec<u8>| -> Result<(), Error> {
            let d = blob(media_type, bytes)?;
            manifest.layers.retain(|l| l.media_type != media_type);
            manifest.layers.push(d);
            Ok(())
        };
        if let Some(q) = &spec.questions {
            let parsed =
                parse_questions(q).map_err(|e| Error::InvalidRequest(format!("QUESTIONS: {e}")))?;
            check_limits(&parsed)?;
            replace(
                media::QUESTIONS,
                serde_json::to_vec_pretty(q).expect("JSON value serializes"),
            )?;
        }
        if let Some(c) = &spec.calibration {
            if is_router {
                return Err(Error::InvalidRequest(
                    "CALIBRATION applies to a model, not a router".into(),
                ));
            }
            serde_json::from_value::<ollaya_decision::CalibrationFile>(c.clone())
                .map_err(|e| Error::InvalidRequest(format!("CALIBRATION: {e}")))?;
            replace(
                media::CALIBRATION,
                serde_json::to_vec_pretty(c).expect("JSON value serializes"),
            )?;
        }
        if let Some(p) = &spec.precision {
            if is_router || !matches!(p.as_str(), "fp16" | "fp32") {
                return Err(Error::InvalidRequest(format!(
                    "PARAMETER precision {p:?}: use fp16 or fp32 on a model"
                )));
            }
            replace(
                media::PARAMS,
                serde_json::to_vec(&serde_json::json!({"precision": p})).expect("serializes"),
            )?;
        }
        if let Some(l) = &spec.license {
            replace(media::LICENSE, l.clone().into_bytes())?;
        }
        // The config records where the model came from (`details.parent_model`).
        let mut config: Value = self.store.read_blob_json(&manifest.config)?;
        if let Some(obj) = config.as_object_mut() {
            obj.insert("parent_model".into(), Value::String(from.to_string()));
            if let Some(desc) = &spec.description {
                obj.insert("description".into(), Value::String(desc.clone()));
            }
        }
        manifest.config = blob(
            media::CONFIG,
            serde_json::to_vec_pretty(&config).expect("serializes"),
        )?;
        for layer in &manifest.layers {
            let verb = if inherited.contains(&layer.digest) {
                "using existing"
            } else {
                "creating new"
            };
            progress(format!("{verb} layer {}", layer.digest));
        }
        progress("writing manifest".into());
        let bytes = serde_json::to_vec_pretty(&manifest).expect("manifest serializes");
        self.store.write_manifest(&name, &bytes)?;
        Ok(())
    }

    pub fn copy(&self, source: &str, destination: &str) -> Result<(), Error> {
        let (src, dst) = (ModelName::parse(source)?, ModelName::parse(destination)?);
        if self.is_busy(&dst) {
            return Err(Error::Busy(dst.to_string()));
        }
        if self.store.read_manifest(&src)?.is_none() {
            return Err(Error::ModelNotFound(src.to_string()));
        }
        Ok(self.store.copy(&src, &dst)?)
    }

    pub fn running(&self) -> Vec<RunningInfo> {
        self.scheduler.running()
    }
}

/// Releases a create's claim on a model name.
struct Claim<'a> {
    set: &'a Mutex<HashSet<String>>,
    key: String,
}

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.set.lock().unwrap().remove(&self.key);
    }
}

fn check_limits(questions: &Questions) -> Result<(), Error> {
    for (qid, q) in questions {
        let n = q.num_options();
        match q.criteria {
            ollaya_decision::Criteria::Choice(_) if n > MAX_CHOICE_OPTIONS => {
                return Err(Error::InvalidRequest(format!(
                    "question {qid:?}: too many choices ({n}); the limit is {MAX_CHOICE_OPTIONS}"
                )));
            }
            ollaya_decision::Criteria::Score(_) if n > MAX_SCORE_LEVELS => {
                return Err(Error::InvalidRequest(format!(
                    "question {qid:?}: too many score levels ({n}); the limit is {MAX_SCORE_LEVELS}"
                )));
            }
            _ => {}
        }
    }
    Ok(())
}

/// Calibrated answers from the runner's raw logits. `state_tokens` is the runner's count, which
/// an input-conditioned calibration reads.
fn answers_from(
    model: &Loadable,
    questions: &Questions,
    raw: &Value,
    state_tokens: usize,
) -> Result<Vec<Answer>, Error> {
    let rows = raw["questions"]
        .as_array()
        .ok_or_else(|| Error::Runner("malformed runner response".into()))?;
    if rows.len() != questions.len() {
        return Err(Error::Runner(format!(
            "runner answered {} of {} questions",
            rows.len(),
            questions.len()
        )));
    }
    questions
        .values()
        .zip(rows)
        .map(|(q, row)| {
            let logits: Vec<f32> = serde_json::from_value(row["logits"].clone())
                .map_err(|e| Error::Runner(format!("malformed logits: {e}")))?;
            if logits.len() != q.num_options() {
                return Err(Error::Runner(format!(
                    "runner returned {} logits for {} options",
                    logits.len(),
                    q.num_options()
                )));
            }
            let act: Option<Vec<f32>> = serde_json::from_value(row["act_logits"].clone()).ok();
            Ok(Answer::new(
                q,
                &model.calibration,
                &logits,
                act.as_deref(),
                state_tokens,
            ))
        })
        .collect()
}
