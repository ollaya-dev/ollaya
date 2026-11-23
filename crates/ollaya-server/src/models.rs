//! Model resolution: a name in the local store to what a runner needs to load it.

use std::path::PathBuf;

use ollaya_decision::{Calibration, CalibrationFile};
use ollaya_registry::manifest::{
    ANNOTATION_GRAPH, ANNOTATION_PRECISION, ANNOTATION_QUANTIZATION, media,
};
use ollaya_registry::{Descriptor, Entry, ModelConfig, ModelName, Router, Store};
use serde_json::Value;

use crate::Error;

/// Files a runner loads, all inside the blob store.
#[derive(Debug, Clone, PartialEq)]
pub struct RunnerFiles {
    pub graph_fp32: Option<PathBuf>,
    pub graph_fp16: Option<PathBuf>,
    pub tokenizer: PathBuf,
    pub decision: PathBuf,
    /// The arch layer (models the MLX engine can run).
    pub arch: Option<PathBuf>,
    /// The weights files, passed with an arch layer, in the manifest's order.
    pub weights: Vec<PathBuf>,
    /// A vision model's image graph.
    pub vision_graph: Option<PathBuf>,
}

/// Files of a GGUF model, run by a llama.cpp runner.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaFiles {
    pub mmproj: Option<PathBuf>,
    /// The author's `tokenizer.json`, for a layout that renders with it (jev-style).
    pub tokenizer: Option<PathBuf>,
    pub gguf: PathBuf,
    pub decision: PathBuf,
    /// The GGUF's type (`Q4_0`), from the layer's annotation.
    pub quantization: String,
}

/// What loads a model: an ONNX Runtime runner, or a llama.cpp one.
#[derive(Debug, Clone, PartialEq)]
pub enum EngineFiles {
    Onnx(RunnerFiles),
    Llama(LlamaFiles),
}

impl EngineFiles {
    /// The engine's name, as `ollaya show` prints it.
    pub fn engine(&self) -> &'static str {
        match self {
            EngineFiles::Onnx(_) => "onnxruntime",
            EngineFiles::Llama(_) => "llama.cpp",
        }
    }
}

/// The media a model reads besides text, as its files declare them. The runner checks again on
/// load: a projector without an audio encoder rejects audio there.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Inputs {
    pub images: bool,
    pub audio: bool,
}

impl Inputs {
    /// An image graph or projector reads images; audio is opt-in (`"llama": {"audio": true}` in
    /// `decision.json`) and only the `winnow-v1` layout places it, as the runner does.
    fn of(files: &EngineFiles, decision: &Value) -> Self {
        match files {
            EngineFiles::Onnx(f) => Inputs {
                images: f.vision_graph.is_some(),
                audio: false,
            },
            EngineFiles::Llama(f) => Inputs {
                images: f.mmproj.is_some(),
                audio: f.mmproj.is_some()
                    && decision["layout"] == "winnow-v1"
                    && decision["llama"]["audio"] == true,
            },
        }
    }
}

/// What an adapter model adds to its files (decision 0011). Its [`Loadable::files`] are the
/// manifest's own layers: the base's tokenizer, `decision.json`, arch and weights.
#[derive(Debug, Clone, PartialEq)]
pub struct AdapterFiles {
    /// The adapter spec layer (`ollaya_registry::adapter::AdapterSpec`).
    pub spec: PathBuf,
    /// The adapter's safetensors file.
    pub weights: PathBuf,
    /// The key adapter models of one base share a runner under.
    pub key: String,
    /// The base the adapter was created `FROM` (`details.parent_model`).
    pub parent: String,
    /// The base's manifest digest, as the spec records it.
    pub base_manifest: String,
    /// Bytes of the base's weights on disk, until a runner reports what it holds.
    pub base_size: u64,
}

/// A model that answers requests itself.
#[derive(Debug, Clone)]
pub struct Loadable {
    pub name: ModelName,
    /// Manifest digest: runners are keyed by content, so two tags of one model share a runner.
    pub digest: String,
    pub files: EngineFiles,
    pub calibration: Calibration,
    pub inputs: Inputs,
    pub config: ModelConfig,
    /// A question schema baked in with a Modelfile, used when a request brings none.
    pub questions: Option<Value>,
    /// Bytes on disk, for `ps` and scheduling.
    pub size: u64,
    /// Set for an adapter model, which a runner shared with the other adapters of its base serves.
    pub adapter: Option<AdapterFiles>,
}

impl Loadable {
    /// What the scheduler keys this model's runner by: the manifest digest, so two tags of one
    /// model share a runner, except for an adapter model, whose runner is the base's.
    pub fn runner_key(&self) -> &str {
        match &self.adapter {
            Some(a) => &a.key,
            None => &self.digest,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Resolved {
    Model(Box<Loadable>),
    /// A router: the target is chosen per request.
    Router {
        name: Box<ModelName>,
        digest: String,
        router: Box<Router>,
        config: Box<ModelConfig>,
        questions: Option<Value>,
    },
}

pub fn resolve(store: &Store, name: &ModelName) -> Result<Resolved, Error> {
    let entry = store
        .read_manifest(name)?
        .ok_or_else(|| Error::ModelNotFound(name.to_string()))?;
    resolve_entry(store, entry)
}

pub fn resolve_entry(store: &Store, entry: Entry) -> Result<Resolved, Error> {
    let Entry {
        name,
        manifest,
        digest,
        ..
    } = entry;
    let config: ModelConfig = store.read_blob_json(&manifest.config)?;
    let questions = match manifest.layer(media::QUESTIONS) {
        Some(d) => Some(store.read_blob_json::<Value>(d)?),
        None => None,
    };
    if let Some(router) = manifest.layer(media::ROUTER) {
        let router = store.read_blob_json(router)?;
        return Ok(Resolved::Router {
            name: Box::new(name),
            digest,
            router: Box::new(router),
            config: Box::new(config),
            questions,
        });
    }
    if !matches!(
        config.model_format.as_str(),
        "" | "onnx" | "gguf" | ollaya_registry::adapter::MODEL_FORMAT
    ) {
        return Err(Error::Unsupported(format!(
            "{name} is a {} model, which this build of ollaya cannot run",
            config.model_format
        )));
    }
    let blob = |media_type: &str| -> Result<PathBuf, Error> {
        let d = manifest
            .layer(media_type)
            .ok_or_else(|| Error::Corrupt(format!("{name}: manifest has no {media_type} layer")))?;
        Ok(store.blob_path(&d.digest)?)
    };
    let calibration = match manifest.layer(media::CALIBRATION) {
        Some(d) => Calibration::from_file(&store.read_blob_json::<CalibrationFile>(d)?),
        None => Calibration::default(),
    };
    if config.model_format == "gguf" {
        let quantization = manifest
            .layer(media::GGUF)
            .and_then(|d| d.annotations.get(ANNOTATION_QUANTIZATION))
            .cloned()
            .unwrap_or_default();
        let files = EngineFiles::Llama(LlamaFiles {
            mmproj: manifest
                .layer(media::MMPROJ)
                .map(|d| store.blob_path(&d.digest))
                .transpose()?,
            tokenizer: manifest
                .layer(media::TOKENIZER)
                .map(|d| store.blob_path(&d.digest))
                .transpose()?,
            gguf: blob(media::GGUF)?,
            decision: blob(media::DECISION)?,
            quantization,
        });
        let decision = store.read_blob_json::<Value>(
            manifest
                .layer(media::DECISION)
                .expect("the decision layer was found above"),
        )?;
        let inputs = Inputs::of(&files, &decision);
        return Ok(Resolved::Model(Box::new(Loadable {
            name,
            digest,
            files,
            calibration,
            inputs,
            config,
            questions,
            size: manifest.total_size(),
            adapter: None,
        })));
    }
    let mut files = RunnerFiles {
        graph_fp32: None,
        graph_fp16: None,
        tokenizer: blob(media::TOKENIZER)?,
        decision: blob(media::DECISION)?,
        arch: None,
        weights: Vec::new(),
        vision_graph: None,
    };
    // MLX builds its network from the weights files, in the order the arch layer lists them.
    if let Some(arch) = manifest.layer(media::ARCH) {
        files.arch = Some(store.blob_path(&arch.digest)?);
        for w in manifest.layers_of(media::WEIGHTS) {
            files.weights.push(store.blob_path(&w.digest)?);
        }
    }
    for g in manifest.layers_of(media::GRAPH_ONNX) {
        let path = store.blob_path(&g.digest)?;
        match main_graph_precision(g) {
            None => files.vision_graph = Some(path),
            Some("fp16") => files.graph_fp16 = Some(path),
            Some(_) => files.graph_fp32 = Some(path),
        }
    }
    // A precision-pinned tag (`laya:en-fp32`) keeps only the graph it names.
    if let Some(params) = manifest.layer(media::PARAMS) {
        let params: Value = store.read_blob_json(params)?;
        match params["precision"].as_str() {
            Some("fp32") => files.graph_fp16 = None,
            Some("fp16") => files.graph_fp32 = None,
            _ => {}
        }
    }
    let adapter = if config.model_format == ollaya_registry::adapter::MODEL_FORMAT {
        Some(adapter_files(store, &manifest, &name.to_string())?)
    } else {
        None
    };
    // An adapter model runs on MLX only, from the base's arch layer and weights: no graph needed.
    if adapter.is_none() && files.graph_fp32.is_none() && files.graph_fp16.is_none() {
        return Err(Error::Corrupt(format!("{name}: manifest has no graph")));
    }
    let files = EngineFiles::Onnx(files);
    Ok(Resolved::Model(Box::new(Loadable {
        name,
        digest,
        inputs: Inputs::of(&files, &Value::Null),
        files,
        calibration,
        config,
        questions,
        size: manifest.total_size(),
        adapter,
    })))
}

/// The adapter layers of an adapter model, checked against the base layers the manifest keeps.
fn adapter_files(
    store: &Store,
    manifest: &ollaya_registry::Manifest,
    name: &str,
) -> Result<AdapterFiles, Error> {
    use ollaya_registry::adapter::AdapterSpec;
    let layer = |media_type: &str| {
        manifest.layer(media_type).ok_or_else(|| {
            Error::Corrupt(format!("{name}: adapter model has no {media_type} layer"))
        })
    };
    let (spec_layer, weights_layer, arch) = (
        layer(media::ADAPTER)?,
        layer(media::ADAPTER_WEIGHTS)?,
        layer(media::ARCH)?,
    );
    let spec: AdapterSpec = store.read_blob_json(spec_layer)?;
    let base_weights: Vec<&Descriptor> = manifest.layers_of(media::WEIGHTS).collect();
    let digests: Vec<String> = base_weights.iter().map(|w| w.digest.clone()).collect();
    if spec.schema != ollaya_registry::adapter::SCHEMA
        || spec.format != ollaya_registry::adapter::FORMAT
    {
        return Err(Error::Unsupported(format!(
            "{name}: adapter schema {} {:?} is not one this version of ollaya reads; upgrade ollaya",
            spec.schema, spec.format
        )));
    }
    if spec.base.arch != arch.digest || spec.base.weights != digests {
        return Err(Error::Corrupt(format!(
            "{name}: the adapter was checked against another base than the manifest holds"
        )));
    }
    let parent = {
        let c: Value = store.read_blob_json(&manifest.config)?;
        c["parent_model"].as_str().unwrap_or_default().to_owned()
    };
    Ok(AdapterFiles {
        spec: store.blob_path(&spec_layer.digest)?,
        weights: store.blob_path(&weights_layer.digest)?,
        key: AdapterSpec::sharing_key(&arch.digest, &digests),
        parent,
        base_manifest: spec.base.manifest,
        base_size: base_weights.iter().map(|w| w.size).sum(),
    })
}

/// The precision an ONNX graph layer computes in, `fp16` or `fp32` (also when unannotated), or
/// `None` for a vision model's image graph, which is never the main graph.
pub(crate) fn main_graph_precision(graph: &Descriptor) -> Option<&'static str> {
    if graph.annotations.get(ANNOTATION_GRAPH).map(String::as_str) == Some("vision") {
        return None;
    }
    match graph
        .annotations
        .get(ANNOTATION_PRECISION)
        .map(String::as_str)
    {
        Some("fp16") => Some("fp16"),
        _ => Some("fp32"),
    }
}

#[cfg(test)]
mod tests {
    use ollaya_registry::manifest::{Descriptor, MANIFEST_V2, Manifest};

    use super::*;

    fn blob(store: &Store, media_type: &str, bytes: &[u8]) -> Descriptor {
        Descriptor {
            media_type: media_type.into(),
            digest: store.write_blob(bytes).unwrap(),
            size: bytes.len() as u64,
            urls: vec![],
            annotations: Default::default(),
        }
    }

    /// A model with one fp32 graph, `weights` weights files and, with `arch`, an arch layer; with
    /// `vision`, an image graph after the main one.
    fn resolve_with(
        weights: usize,
        arch: bool,
        vision: bool,
    ) -> (RunnerFiles, Store, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::open(dir.path()).unwrap();
        let config = br#"{"model_format": "onnx", "family": "laya"}"#;
        let mut layers = vec![
            blob(&store, media::GRAPH_ONNX, b"graph"),
            blob(&store, media::TOKENIZER, b"{}"),
            blob(&store, media::DECISION, br#"{"layout": "laya-markers-v1"}"#),
        ];
        for i in 0..weights {
            layers.push(blob(
                &store,
                media::WEIGHTS,
                format!("weights {i}").as_bytes(),
            ));
        }
        if arch {
            layers.push(blob(&store, media::ARCH, br#"{"schema": 1}"#));
        }
        if vision {
            let mut g = blob(&store, media::GRAPH_ONNX, b"vision graph");
            g.annotations
                .insert(ANNOTATION_GRAPH.into(), "vision".into());
            layers.push(g);
        }
        let manifest = Manifest {
            schema_version: 2,
            media_type: MANIFEST_V2.into(),
            config: blob(&store, media::CONFIG, config),
            layers,
        };
        let name = ModelName::parse("laya:en").unwrap();
        store
            .write_manifest(&name, &serde_json::to_vec(&manifest).unwrap())
            .unwrap();
        match resolve(&store, &name).unwrap() {
            Resolved::Model(m) => match m.files {
                EngineFiles::Onnx(f) => (f, store, dir),
                EngineFiles::Llama(_) => panic!("not an ONNX model"),
            },
            Resolved::Router { .. } => panic!("not a router"),
        }
    }

    #[test]
    fn gguf_projector_is_optional_and_resolves_to_its_own_blob() {
        for vision in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let store = Store::open(dir.path()).unwrap();
            let mut layers = vec![
                blob(&store, media::GGUF, b"text weights"),
                blob(&store, media::DECISION, br#"{"layout": "winnow-v1"}"#),
            ];
            let projector = blob(&store, media::MMPROJ, b"projector weights");
            if vision {
                layers.push(projector.clone());
            }
            let manifest = Manifest {
                schema_version: 2,
                media_type: MANIFEST_V2.into(),
                config: blob(
                    &store,
                    media::CONFIG,
                    br#"{"model_format":"gguf","family":"winnow"}"#,
                ),
                layers,
            };
            let name = ModelName::parse("winnow:e4b-vision").unwrap();
            store
                .write_manifest(&name, &serde_json::to_vec(&manifest).unwrap())
                .unwrap();
            let Resolved::Model(model) = resolve(&store, &name).unwrap() else {
                panic!("expected model");
            };
            let inputs = Inputs {
                images: vision,
                audio: false,
            };
            assert_eq!(model.inputs, inputs);
            let EngineFiles::Llama(files) = model.files else {
                panic!("expected llama");
            };
            assert_eq!(
                files.mmproj,
                if vision {
                    Some(store.blob_path(&projector.digest).unwrap())
                } else {
                    None
                }
            );
        }
    }

    #[test]
    fn audio_is_read_only_when_the_decision_layer_opts_in() {
        let inputs = |decision: Value, mmproj: bool| {
            let files = EngineFiles::Llama(LlamaFiles {
                mmproj: mmproj.then(|| PathBuf::from("mmproj")),
                tokenizer: None,
                gguf: PathBuf::from("gguf"),
                decision: PathBuf::from("decision.json"),
                quantization: String::new(),
            });
            Inputs::of(&files, &decision)
        };
        let both = Inputs {
            images: true,
            audio: true,
        };
        let images = Inputs {
            images: true,
            audio: false,
        };
        let on = serde_json::json!({"layout": "winnow-v1", "llama": {"audio": true}});
        assert_eq!(inputs(on.clone(), true), both);
        assert_eq!(inputs(on, false), Inputs::default());
        for off in [
            serde_json::json!({"layout": "winnow-v1", "llama": {}}),
            serde_json::json!({"layout": "winnow-v1", "llama": {"audio": "yes"}}),
            serde_json::json!({"layout": "d1-omni-v1", "llama": {"audio": true}}),
        ] {
            assert_eq!(inputs(off, true), images);
        }
        let (files, ..) = resolve_with(1, false, true);
        assert_eq!(Inputs::of(&EngineFiles::Onnx(files), &Value::Null), images);
    }

    #[test]
    fn an_arch_layer_passes_the_weights_files_to_mlx() {
        for n in [1, 2] {
            let (files, ..) = resolve_with(n, true, false);
            assert!(files.arch.is_some() && files.weights.len() == n);
        }
        // Without an arch layer the model stays on ONNX Runtime.
        let (files, ..) = resolve_with(1, false, false);
        assert!(files.arch.is_none() && files.weights.is_empty());
    }

    #[test]
    fn a_vision_graph_is_passed_on_its_own() {
        let (files, store, _dir) = resolve_with(1, false, true);
        let digest = |b: &[u8]| {
            format!(
                "sha256:{}",
                hex::encode(<sha2::Sha256 as sha2::Digest>::digest(b))
            )
        };
        assert_eq!(
            files.graph_fp32,
            Some(store.blob_path(&digest(b"graph")).unwrap())
        );
        assert_eq!(
            files.vision_graph,
            Some(store.blob_path(&digest(b"vision graph")).unwrap())
        );
        assert!(resolve_with(1, false, false).0.vision_graph.is_none());
    }
}
