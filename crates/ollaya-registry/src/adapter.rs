//! Adapter models (decision 0011): a LoRA fine-tune that is a model of its own and shares the
//! loaded base with the other adapters of that base.
//!
//! The manifest of an adapter model keeps every layer of its base and adds two:
//! [`media::ADAPTER`] (the [`AdapterSpec`] below, which holds no weights) and
//! [`media::ADAPTER_WEIGHTS`] (the safetensors file as peft wrote it). This module holds what the
//! daemon needs without an engine: the spec, its canonical bytes, the sharing key, and the checks
//! `create` runs on an adapter against its base's tensor shapes.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::manifest::media;

pub const SCHEMA: u32 = 1;
pub const FORMAT: &str = "peft-lora";
/// The name prefix peft gives every tensor of an adapter.
pub const PREFIX: &str = "base_model.model.";
/// `ModelConfig.model_format` of an adapter model.
pub const MODEL_FORMAT: &str = "adapter";

/// What the adapter was checked against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterBase {
    /// The base's manifest digest.
    pub manifest: String,
    /// The base's arch layer digest.
    pub arch: String,
    /// The base's weights layer digests, in the manifest's order.
    pub weights: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterSpec {
    pub schema: u32,
    pub format: String,
    pub base: AdapterBase,
    pub rank: u32,
    pub alpha: f64,
    /// `alpha / rank`.
    pub scaling: f64,
    /// The dtype of the adapter's tensors: `F32`, `F16` or `BF16`.
    pub dtype: String,
    pub prefix: String,
    /// Each adapted module, by the name of its `.weight` tensor in the base, without the suffix.
    pub targets: Vec<String>,
}

impl AdapterSpec {
    /// The spec as bytes: sorted keys, no insignificant whitespace, so that the same adapter on
    /// the same base always gives the same blob.
    pub fn canonical(&self) -> Vec<u8> {
        let value = serde_json::to_value(self).expect("a spec serializes");
        let mut out = Vec::new();
        write_canonical(&value, &mut out);
        out
    }

    /// The key adapter models of one base share a runner under: the base's arch digest and its
    /// weights digests, not its manifest digest (a base with other questions or calibration has
    /// the same tensors). It never equals a manifest digest, which plain models are keyed by.
    pub fn sharing_key(arch: &str, weights: &[String]) -> String {
        format!("adapter-base:{arch}:{}", weights.join(","))
    }
}

fn write_canonical(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, &Value> = map.iter().collect();
            out.push(b'{');
            for (i, (k, v)) in sorted.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                out.extend(serde_json::to_vec(k).expect("a key serializes"));
                out.push(b':');
                write_canonical(v, out);
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(v, out);
            }
            out.push(b']');
        }
        other => out.extend(serde_json::to_vec(other).expect("a scalar serializes")),
    }
}

/// One tensor of a safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// `[begin, end)` offsets into the data section.
    pub offsets: [u64; 2],
}

/// Parse the JSON header of a safetensors file (the bytes after the 8-byte length), without
/// `__metadata__`.
pub fn parse_safetensors_header(header: &[u8]) -> Result<Vec<TensorInfo>, String> {
    let map: serde_json::Map<String, Value> =
        serde_json::from_slice(header).map_err(|e| format!("not a safetensors header: {e}"))?;
    let mut out = Vec::new();
    for (name, v) in map {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v["dtype"]
            .as_str()
            .ok_or_else(|| format!("tensor {name} has no dtype"))?
            .to_owned();
        let shape: Vec<usize> = serde_json::from_value(v["shape"].clone())
            .map_err(|e| format!("tensor {name}: shape: {e}"))?;
        let offsets: [u64; 2] = serde_json::from_value(v["data_offsets"].clone())
            .map_err(|e| format!("tensor {name}: data_offsets: {e}"))?;
        out.push(TensorInfo {
            name,
            dtype,
            shape,
            offsets,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// The tensors of a safetensors file, read from its header alone (the largest header accepted is
/// 100 MiB, as the runner's reader does). Only dtype, shape and offsets: no tensor is read.
pub fn read_safetensors_header(path: &std::path::Path) -> Result<Vec<TensorInfo>, String> {
    use std::io::Read as _;
    let shown = path.display();
    let mut file = std::fs::File::open(path).map_err(|e| format!("{shown}: {e}"))?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)
        .map_err(|e| format!("{shown}: not a safetensors file: {e}"))?;
    let n = u64::from_le_bytes(len);
    if n > 100 << 20 {
        return Err(format!("{shown}: safetensors header of {n} bytes"));
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact(&mut header)
        .map_err(|e| format!("{shown}: not a safetensors file: {e}"))?;
    parse_safetensors_header(&header).map_err(|e| format!("{shown}: {e}"))
}

/// The checked adapter: what the spec records besides its base.
#[derive(Debug, Clone, PartialEq)]
pub struct Checked {
    pub rank: u32,
    pub alpha: f64,
    pub dtype: String,
    pub targets: Vec<String>,
}

fn number(v: &Value) -> Option<f64> {
    v.as_f64().filter(|n| n.is_finite())
}

/// Whether a peft option is switched on: absent, `false`, `null`, `0`, `{}` and `[]` are off.
fn on(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) | Some(Value::Bool(false)) => false,
        Some(Value::Object(m)) => !m.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Number(n)) => n.as_f64() != Some(0.0),
        Some(_) => true,
    }
}

/// The checks of decision 0011, section 2, on an adapter before anything is written. `config` is
/// peft's `adapter_config.json`, `tensors` the adapter's safetensors header, `base` the shape of
/// every tensor of the base by name, `backbone_prefix` the arch layer's backbone prefix. A refusal
/// names the module and both shapes.
pub fn check(
    config: &Value,
    tensors: &[TensorInfo],
    base: &HashMap<String, Vec<usize>>,
    backbone_prefix: &str,
) -> Result<Checked, String> {
    // 2. The adapter config's own checks.
    if config["peft_type"] != "LORA" {
        return Err(format!(
            "adapter config: peft_type is {}, expected \"LORA\"",
            config["peft_type"]
        ));
    }
    let r = config["r"]
        .as_u64()
        .filter(|r| *r > 0 && *r <= u32::MAX as u64)
        .ok_or("adapter config: r must be a positive whole number")?;
    let alpha = number(&config["lora_alpha"])
        .filter(|a| *a > 0.0)
        .ok_or("adapter config: lora_alpha must be a positive number")?;
    for key in ["use_rslora", "use_dora"] {
        if on(config.get(key)) {
            return Err(format!("adapter config: {key} is not supported"));
        }
    }
    for key in ["rank_pattern", "alpha_pattern"] {
        if on(config.get(key)) {
            return Err(format!("adapter config: {key} is not supported"));
        }
    }
    if on(config.get("modules_to_save")) {
        return Err(
            "adapter config: modules_to_save (trained heads and other replaced tensors) is not \
             supported yet"
                .into(),
        );
    }
    let r = r as usize;

    // 4. The dtype.
    let mut dtype: Option<&str> = None;
    for t in tensors {
        if !matches!(t.dtype.as_str(), "F32" | "F16" | "BF16") {
            return Err(format!(
                "tensor {}: dtype {} is not supported (F32, F16 or BF16)",
                t.name, t.dtype
            ));
        }
        match dtype {
            None => dtype = Some(&t.dtype),
            Some(d) if d != t.dtype => {
                return Err(format!(
                    "tensor {}: dtype {} differs from {d}; an adapter has one dtype",
                    t.name, t.dtype
                ));
            }
            Some(_) => {}
        }
    }
    let dtype = dtype.ok_or("the adapter holds no tensors")?;

    // 3. Every tensor is half of a lora_A / lora_B pair on a linear weight of the base.
    let mut pairs: BTreeMap<&str, [Option<&TensorInfo>; 2]> = BTreeMap::new();
    for t in tensors {
        let module = t
            .name
            .strip_prefix(PREFIX)
            .and_then(|n| n.strip_suffix(".lora_A.weight").map(|m| (m, 0)))
            .or_else(|| {
                t.name
                    .strip_prefix(PREFIX)
                    .and_then(|n| n.strip_suffix(".lora_B.weight").map(|m| (m, 1)))
            });
        let Some((module, half)) = module else {
            return Err(format!(
                "tensor {} is not a LoRA pair ({PREFIX}<module>.lora_A.weight or lora_B.weight); \
                 trained heads and other modules_to_save tensors are not supported yet",
                t.name
            ));
        };
        pairs.entry(module).or_default()[half] = Some(t);
    }
    let mut targets = BTreeSet::new();
    for (module, [a, b]) in pairs {
        let (Some(a), Some(b)) = (a, b) else {
            return Err(format!(
                "module {module}: the adapter has only one of lora_A and lora_B"
            ));
        };
        let target = format!("{backbone_prefix}{module}");
        let weight = format!("{target}.weight");
        let Some(shape) = base.get(&weight) else {
            return Err(format!(
                "module {module}: the base has no tensor {weight} (unknown target module)"
            ));
        };
        let &[out, inp] = shape.as_slice() else {
            return Err(format!(
                "module {module}: {weight} has shape {shape:?}, not a linear weight [out, in]"
            ));
        };
        if a.shape != [r, inp] {
            return Err(format!(
                "module {module}: lora_A has shape {:?}, expected [{r}, {inp}] (r = {r}, base in = {inp})",
                a.shape
            ));
        }
        if b.shape != [out, r] {
            return Err(format!(
                "module {module}: lora_B has shape {:?}, expected [{out}, {r}] (base out = {out}, r = {r})",
                b.shape
            ));
        }
        targets.insert(target);
    }
    Ok(Checked {
        rank: r as u32,
        alpha,
        dtype: dtype.to_owned(),
        targets: targets.into_iter().collect(),
    })
}

/// Layer media types an adapter model must have beyond a runnable MLX base.
pub fn is_adapter_layer(media_type: &str) -> bool {
    media_type == media::ADAPTER || media_type == media::ADAPTER_WEIGHTS
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn spec() -> AdapterSpec {
        AdapterSpec {
            schema: SCHEMA,
            format: FORMAT.into(),
            base: AdapterBase {
                manifest: "sha256:m".into(),
                arch: "sha256:a".into(),
                weights: vec!["sha256:w".into()],
            },
            rank: 4,
            alpha: 8.0,
            scaling: 2.0,
            dtype: "F32".into(),
            prefix: PREFIX.into(),
            targets: vec!["encoder.layers.0.attn.Wo".into()],
        }
    }

    #[test]
    fn the_spec_is_written_with_sorted_keys_and_no_whitespace() {
        let text = String::from_utf8(spec().canonical()).unwrap();
        assert_eq!(
            text,
            r#"{"alpha":8.0,"base":{"arch":"sha256:a","manifest":"sha256:m","weights":["sha256:w"]},"dtype":"F32","format":"peft-lora","prefix":"base_model.model.","rank":4,"scaling":2.0,"schema":1,"targets":["encoder.layers.0.attn.Wo"]}"#
        );
        assert_eq!(spec().canonical(), spec().canonical());
        let mut other = spec();
        other.rank = 8;
        assert_ne!(spec().canonical(), other.canonical());
    }

    #[test]
    fn the_sharing_key_is_the_base_tensors_not_the_manifest() {
        let key = AdapterSpec::sharing_key("sha256:a", &["sha256:w1".into(), "sha256:w2".into()]);
        assert_eq!(key, "adapter-base:sha256:a:sha256:w1,sha256:w2");
        // A manifest digest (a plain model's key) is never one.
        assert!(!key.starts_with("sha256:"));
        assert_ne!(
            key,
            AdapterSpec::sharing_key("sha256:a", &["sha256:w2".into(), "sha256:w1".into()])
        );
    }

    fn t(name: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dtype: dtype.into(),
            shape: shape.to_vec(),
            offsets: [0, 0],
        }
    }

    fn base() -> HashMap<String, Vec<usize>> {
        [
            ("encoder.layers.0.attn.Wo.weight".to_owned(), vec![8, 8]),
            ("encoder.layers.0.mlp.Wi.weight".to_owned(), vec![32, 8]),
            ("encoder.norm.weight".to_owned(), vec![8]),
        ]
        .into()
    }

    fn pair(module: &str, r: usize, out: usize, inp: usize, dtype: &str) -> [TensorInfo; 2] {
        [
            t(&format!("{PREFIX}{module}.lora_A.weight"), dtype, &[r, inp]),
            t(&format!("{PREFIX}{module}.lora_B.weight"), dtype, &[out, r]),
        ]
    }

    fn config() -> Value {
        json!({"peft_type": "LORA", "r": 4, "lora_alpha": 8, "use_rslora": false,
               "use_dora": false, "rank_pattern": {}, "alpha_pattern": {}, "modules_to_save": null})
    }

    #[test]
    fn a_good_adapter_is_checked_and_its_targets_are_sorted() {
        let mut ts = Vec::from(pair("layers.0.mlp.Wi", 4, 32, 8, "F32"));
        ts.extend(pair("layers.0.attn.Wo", 4, 8, 8, "F32"));
        let c = check(&config(), &ts, &base(), "encoder.").unwrap();
        assert_eq!(c.rank, 4);
        assert_eq!(c.alpha, 8.0);
        assert_eq!(c.dtype, "F32");
        assert_eq!(
            c.targets,
            ["encoder.layers.0.attn.Wo", "encoder.layers.0.mlp.Wi"]
        );
    }

    fn refused(config: Value, ts: &[TensorInfo]) -> String {
        check(&config, ts, &base(), "encoder.").unwrap_err()
    }

    #[test]
    fn modules_to_save_are_refused() {
        let ts = Vec::from(pair("layers.0.attn.Wo", 4, 8, 8, "F32"));
        let mut c = config();
        c["modules_to_save"] = json!(["classifier"]);
        assert!(refused(c, &ts).contains("modules_to_save"));
        // A tensor that is not half of a pair (a trained head) is refused as well.
        let mut with_head = ts.clone();
        with_head.push(t(
            &format!("{PREFIX}classifier.modules_to_save.default.weight"),
            "F32",
            &[2, 8],
        ));
        let why = refused(config(), &with_head);
        assert!(why.contains("not a LoRA pair"), "{why}");
        assert!(why.contains("classifier.modules_to_save"), "{why}");
    }

    #[test]
    fn unknown_target_modules_and_shape_mismatches_name_the_module() {
        let why = refused(config(), &pair("layers.9.attn.Wo", 4, 8, 8, "F32"));
        assert!(
            why.contains("layers.9.attn.Wo") && why.contains("unknown target"),
            "{why}"
        );
        // A weight that is not a matrix.
        let why = refused(config(), &pair("norm", 4, 8, 8, "F32"));
        assert!(why.contains("not a linear weight"), "{why}");
        // lora_A of another input width, lora_B of another rank.
        let why = refused(config(), &pair("layers.0.attn.Wo", 4, 8, 16, "F32"));
        assert!(why.contains("lora_A") && why.contains("[4, 8]"), "{why}");
        let mut ts = Vec::from(pair("layers.0.attn.Wo", 4, 8, 8, "F32"));
        ts[1].shape = vec![8, 2];
        let why = refused(config(), &ts);
        assert!(why.contains("lora_B") && why.contains("[8, 4]"), "{why}");
        // Half a pair.
        let why = refused(config(), &pair("layers.0.attn.Wo", 4, 8, 8, "F32")[..1]);
        assert!(why.contains("only one of"), "{why}");
        // The adapter's rank is not the config's.
        let why = refused(config(), &pair("layers.0.attn.Wo", 2, 8, 8, "F32"));
        assert!(why.contains("r = 4"), "{why}");
    }

    #[test]
    fn the_configs_own_checks() {
        let ts = Vec::from(pair("layers.0.attn.Wo", 4, 8, 8, "F32"));
        for (key, value, what) in [
            ("peft_type", json!("IA3"), "peft_type"),
            ("r", json!(0), "r must"),
            ("lora_alpha", json!(-1), "lora_alpha"),
            ("use_rslora", json!(true), "use_rslora"),
            ("use_dora", json!(true), "use_dora"),
            ("rank_pattern", json!({"a": 2}), "rank_pattern"),
            ("alpha_pattern", json!({"a": 2}), "alpha_pattern"),
        ] {
            let mut c = config();
            c[key] = value;
            let why = refused(c, &ts);
            assert!(why.contains(what), "{key}: {why}");
        }
    }

    #[test]
    fn the_dtype_is_f32_f16_or_bf16_and_one_for_all() {
        let why = refused(config(), &pair("layers.0.attn.Wo", 4, 8, 8, "F64"));
        assert!(why.contains("F64"), "{why}");
        for ok in ["F32", "F16", "BF16"] {
            check(
                &config(),
                &pair("layers.0.attn.Wo", 4, 8, 8, ok),
                &base(),
                "encoder.",
            )
            .unwrap();
        }
        let mut ts = Vec::from(pair("layers.0.attn.Wo", 4, 8, 8, "F32"));
        ts[1].dtype = "F16".into();
        assert!(refused(config(), &ts).contains("one dtype"));
        assert!(refused(config(), &[]).contains("no tensors"));
    }

    #[test]
    fn a_safetensors_header_is_read() {
        let header = br#"{"__metadata__":{"format":"pt"},"b":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}}"#;
        let ts = parse_safetensors_header(header).unwrap();
        assert_eq!(
            ts,
            vec![TensorInfo {
                name: "b".into(),
                dtype: "F32".into(),
                shape: vec![2, 3],
                offsets: [0, 24]
            }]
        );
        assert!(parse_safetensors_header(b"[1]").is_err());
    }
}
