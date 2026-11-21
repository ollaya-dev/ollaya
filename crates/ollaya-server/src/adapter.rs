//! Creating an adapter model (decision 0011): the checks `create` runs on an adapter against its
//! base, and the two layers it adds to the base's manifest.

use std::collections::HashMap;

use ollaya_registry::Store;
use ollaya_registry::adapter::{self, AdapterBase, AdapterSpec};
use ollaya_registry::manifest::{Descriptor, Manifest, media};
use serde_json::Value;

use crate::Error;

/// Backbones an adapter can run on in this version: ModernBERT only. Decoders are a later slice.
/// DeBERTa-v2 is refused: its relative-position projections (`pos_key`, `pos_query`) are computed
/// once at load without the adapter, so an adapter on `query_proj` or `key_proj` would differ from
/// the merged model; parity was measured on ModernBERT alone.
const ENCODERS: &[&str] = &["modernbert"];

/// Layouts whose engine takes adapters in this version.
pub const LAYOUTS: &[&str] = &["laya-markers-v1"];

/// What the base says about its tensors.
pub struct BaseTensors {
    /// Every tensor's shape by name, from the weights files' headers (safetensors) and the arch
    /// layer's own table (torch zip).
    pub shapes: HashMap<String, Vec<usize>>,
    /// The arch layer's backbone prefix.
    pub backbone_prefix: String,
}

fn refuse(msg: impl Into<String>) -> Error {
    Error::InvalidRequest(msg.into())
}

/// Read the base's arch layer and the shapes of its weights. Refuses what an adapter cannot sit
/// on: no arch layer, a router, an adapter model, a decoder, a layout without adapter support.
pub fn read_base(store: &Store, from: &str, manifest: &Manifest) -> Result<BaseTensors, Error> {
    if manifest.layer(media::ROUTER).is_some() {
        return Err(refuse(format!("ADAPTER: {from} is a router, not a model")));
    }
    let config: Value = store.read_blob_json(&manifest.config)?;
    if config["model_format"] == adapter::MODEL_FORMAT || manifest.layer(media::ADAPTER).is_some() {
        return Err(refuse(format!(
            "ADAPTER: {from} is an adapter model itself; adapters do not stack"
        )));
    }
    let Some(arch_layer) = manifest.layer(media::ARCH) else {
        return Err(refuse(format!(
            "ADAPTER: {from} has no arch layer, so the MLX engine cannot run it"
        )));
    };
    let arch: Value = store.read_blob_json(arch_layer)?;
    let kind = arch["backbone"]["type"].as_str().unwrap_or_default();
    if kind == "deberta-v2" {
        return Err(refuse(format!(
            "ADAPTER: {from} has a \"deberta-v2\" backbone; adapters are not supported on DeBERTa-v2 \
             yet (its relative-position projections are computed without the adapter), only on \
             ModernBERT"
        )));
    }
    if !ENCODERS.contains(&kind) {
        return Err(refuse(format!(
            "ADAPTER: {from} has a {kind:?} backbone; this version runs adapters on ModernBERT encoders only"
        )));
    }
    let layout = manifest
        .layer(media::DECISION)
        .map(|d| store.read_blob_json::<Value>(d))
        .transpose()?
        .and_then(|d| d["layout"].as_str().map(str::to_owned))
        .unwrap_or_default();
    if !LAYOUTS.contains(&layout.as_str()) {
        return Err(refuse(format!(
            "ADAPTER: {from} uses the {layout:?} layout; this version runs adapters on {}",
            LAYOUTS.join(", ")
        )));
    }
    // The arch layer lists the weights files in the manifest's order of weights layers.
    let mut files: Vec<(&str, &Value)> = vec![(
        arch["weights"]["format"].as_str().unwrap_or_default(),
        &arch["weights"]["tensors"],
    )];
    if let Some(more) = arch["weights"]["more"].as_array() {
        files.extend(
            more.iter()
                .map(|w| (w["format"].as_str().unwrap_or_default(), &w["tensors"])),
        );
    }
    let layers: Vec<&Descriptor> = manifest.layers_of(media::WEIGHTS).collect();
    if layers.len() != files.len() {
        return Err(Error::Corrupt(format!(
            "{from}: the arch layer lists {} weights files, the manifest has {}",
            files.len(),
            layers.len()
        )));
    }
    let mut shapes = HashMap::new();
    for ((format, table), layer) in files.into_iter().zip(layers) {
        match format {
            "safetensors" => {
                let header = adapter::read_safetensors_header(&store.blob_path(&layer.digest)?)
                    .map_err(|e| Error::Corrupt(format!("{from}: {e}")))?;
                shapes.extend(header.into_iter().map(|t| (t.name, t.shape)));
            }
            _ => {
                if let Some(table) = table.as_object() {
                    for (name, t) in table {
                        if let Ok(shape) = serde_json::from_value::<Vec<usize>>(t["shape"].clone())
                        {
                            shapes.insert(name.clone(), shape);
                        }
                    }
                }
            }
        }
    }
    Ok(BaseTensors {
        shapes,
        backbone_prefix: arch["backbone"]["prefix"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
    })
}

/// Check an adapter (peft's config and the uploaded weights blob) against its base, and build the
/// spec. Nothing is written.
pub fn spec_for(
    store: &Store,
    from: &str,
    base_digest: &str,
    manifest: &Manifest,
    config: &Value,
    weights: &str,
) -> Result<AdapterSpec, Error> {
    let path = store.blob_path(weights).map_err(Error::from)?;
    if !path.is_file() {
        return Err(refuse(format!(
            "ADAPTER: the weights {weights} are not in the model store; upload them first with \
             POST /api/blobs/{weights}"
        )));
    }
    let base = read_base(store, from, manifest)?;
    let tensors =
        adapter::read_safetensors_header(&path).map_err(|e| refuse(format!("ADAPTER: {e}")))?;
    let checked = adapter::check(config, &tensors, &base.shapes, &base.backbone_prefix)
        .map_err(|e| refuse(format!("ADAPTER: {e}")))?;
    let arch = manifest.layer(media::ARCH).expect("read_base found it");
    let spec = AdapterSpec {
        schema: adapter::SCHEMA,
        format: adapter::FORMAT.into(),
        base: AdapterBase {
            manifest: base_digest.to_owned(),
            arch: arch.digest.clone(),
            weights: manifest
                .layers_of(media::WEIGHTS)
                .map(|w| w.digest.clone())
                .collect(),
        },
        rank: checked.rank,
        alpha: checked.alpha,
        scaling: checked.alpha / f64::from(checked.rank),
        dtype: checked.dtype,
        prefix: adapter::PREFIX.to_owned(),
        targets: checked.targets,
    };
    Ok(spec)
}
