//! The `arch` layer (`application/vnd.ollaya.arch`): what the MLX engine needs to build a network
//! from the author's unmodified weights file.
//!
//! It is small, derived at build time (`convert/ollaya_convert/arch.py`) from the author's
//! `config.json` and model code, and holds only hyperparameters and tensor names:
//!
//! ```json
//! {"schema": 1,
//!  "backbone": {"type": "modernbert", "prefix": "encoder.", "hidden_size": 1024, ...},
//!  "head": {"type": "laya", ...},
//!  "weights": {"format": "safetensors", "file": "model.safetensors"}}
//! ```
//!
//! A model whose manifest has no arch layer runs on ONNX Runtime only.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

pub const SCHEMA: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Arch {
    pub schema: u32,
    pub backbone: Backbone,
    pub head: Head,
    pub weights: Weights,
    /// A LoRA adapter on the backbone's linear layers, merged into them when the model loads.
    #[serde(default)]
    pub lora: Option<Lora>,
}

/// A peft LoRA adapter: for a backbone weight `<backbone prefix><name>.weight` whose adapter holds
/// `<prefix><name>.lora_A.weight` [r, in] and `<prefix><name>.lora_B.weight` [out, r], the engine
/// runs `W + scaling * B A` (peft's merge, in float32).
#[derive(Debug, Clone, Deserialize)]
pub struct Lora {
    /// Name prefix of the adapter's tensors, e.g. `base_model.model.`.
    pub prefix: String,
    /// `lora_alpha / r`.
    pub scaling: f32,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Backbone {
    Modernbert(ModernBertConfig),
    #[serde(rename = "deberta-v2")]
    DebertaV2(DebertaV2Config),
    /// The dense Qwen3 decoder (qwen3guard).
    Qwen3(Qwen3Config),
    /// The hybrid Qwen3.5 decoder (decider): Gated DeltaNet and gated attention layers.
    #[serde(rename = "qwen3-5")]
    Qwen35(Qwen35Config),
}

impl Backbone {
    pub fn hidden_size(&self) -> usize {
        match self {
            Backbone::Modernbert(c) => c.hidden_size,
            Backbone::DebertaV2(c) => c.hidden_size,
            Backbone::Qwen3(c) => c.hidden_size,
            Backbone::Qwen35(c) => c.hidden_size,
        }
    }

    /// Name prefix of the backbone's tensors in the weights file.
    pub fn prefix(&self) -> &str {
        match self {
            Backbone::Modernbert(c) => &c.prefix,
            Backbone::DebertaV2(c) => &c.prefix,
            Backbone::Qwen3(c) => &c.prefix,
            Backbone::Qwen35(c) => &c.prefix,
        }
    }

    /// A causal decoder (one position per row is read), not an encoder.
    pub fn is_decoder(&self) -> bool {
        matches!(self, Backbone::Qwen3(_) | Backbone::Qwen35(_))
    }
}

/// Hugging Face `Qwen3_5TextConfig` (transformers 5): a hybrid of `linear_attention` (Gated
/// DeltaNet) and `full_attention` layers, with partial RoPE and an attention output gate.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen35Config {
    /// Name prefix of the decoder's tensors in the weights file, e.g. `model.language_model.`.
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    /// `linear_attention` or `full_attention`, one per layer.
    pub layer_types: Vec<String>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    /// RoPE covers the first `head_dim * partial_rotary_factor` channels.
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub attention_bias: bool,
    /// Must be true (`attn_output_gate`).
    pub attn_output_gate: bool,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_eps: f32,
    /// Only `silu`.
    pub hidden_act: String,
    /// Must be true: the LM head is the embedding.
    pub tie_word_embeddings: bool,
}

/// Hugging Face `Qwen3Config` (transformers 5), as the model's `config.json` gives it.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3Config {
    /// Name prefix of the decoder's tensors in the weights file, e.g. `model.`.
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub attention_bias: bool,
    /// Only `silu`.
    pub hidden_act: String,
    /// Must be true: the LM head is the embedding (the heads below read its rows).
    pub tie_word_embeddings: bool,
    /// Must be false: every layer attends to the whole prefix.
    pub use_sliding_window: bool,
}

/// Hugging Face `ModernBertConfig` (transformers 5), as the encoder's `config.json` gives it.
#[derive(Debug, Clone, Deserialize)]
pub struct ModernBertConfig {
    /// Name prefix of the encoder's tensors in the weights file, e.g. `encoder.` or `model.`.
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub norm_eps: f32,
    pub norm_bias: bool,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    /// Only `gelu` (exact, erf) is supported.
    pub hidden_activation: String,
    /// Window of the sliding-attention layers: a token sees `local_attention / 2` on each side.
    pub local_attention: usize,
    /// `full_attention` or `sliding_attention`, one per layer.
    pub layer_types: Vec<String>,
    /// RoPE base per layer type.
    pub rope_theta: BTreeMap<String, f32>,
}

/// Hugging Face `DebertaV2Config` (transformers 5), as the encoder's `config.json` gives it, for
/// DeBERTa-v3: relative attention over log buckets, no absolute positions, no token types, no conv.
#[derive(Debug, Clone, Deserialize)]
pub struct DebertaV2Config {
    /// Name prefix of the encoder's tensors in the weights file, e.g. `deberta.`.
    pub prefix: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub layer_norm_eps: f32,
    /// Only `gelu` (exact, erf) is supported.
    pub hidden_act: String,
    /// `c2p` and/or `p2c`, with `share_att_key`.
    pub pos_att_type: Vec<String>,
    pub share_att_key: bool,
    pub position_buckets: usize,
    /// Resolved: `max_position_embeddings` when the config says -1.
    pub max_relative_positions: usize,
    /// `layer_norm` or `none`.
    pub norm_rel_ebd: String,
    /// Must be false (no absolute position embeddings).
    pub position_biased_input: bool,
    /// Must be 0 (no token type embeddings).
    pub type_vocab_size: usize,
    /// Must be 0 (no conv layer after layer 0).
    pub conv_kernel_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum Head {
    /// Laya's `DecisionModel` head (`laya-markers-v1`).
    Laya(LayaHead),
    /// `ModernBertForSequenceClassification` (`nli-pairs-v1`).
    SequenceClassification(SequenceClassificationHead),
    /// `DebertaV2ForSequenceClassification`: `ContextPooler`, `classifier` (`nli-pairs-v1`).
    ContextPooler(ContextPoolerHead),
    /// Von's `OptionMarkerScorer` (`von-option-marker-v1`).
    OptionMarker(OptionMarkerHead),
    /// GLiClass `GLiClassUniEncoder` (`gliclass-uni-v1`).
    GliclassUni(GliclassHead),
    /// A decoder's tied LM head at one position per row, restricted to the tokens the engine
    /// names at run time: qwen3guard's candidate first tokens (`qwen3guard-gen-v1`), decider's
    /// labels (`decider-slots-v1`).
    TiedRows,
    /// Decision 1.0's `CandidateHead`: each option's last token against the query token
    /// (`decision-endpoint-v1`).
    Endpoint(EndpointHead),
    /// Kev's `PointerHead`: each option's closing token against the decide token
    /// (`kev-pointer-v1`).
    Pointer(PointerHead),
}

impl Head {
    pub fn name(&self) -> &'static str {
        match self {
            Head::Laya(_) => "laya",
            Head::SequenceClassification(_) | Head::ContextPooler(_) => "sequence-classification",
            Head::OptionMarker(_) => "option-marker",
            Head::GliclassUni(_) => "gliclass-uni",
            Head::TiedRows => "tied-rows",
            Head::Endpoint(_) => "endpoint",
            Head::Pointer(_) => "pointer",
        }
    }
}

/// `laya.common.DecisionModel`: a type embedding, `layers` post-encoder `nn.TransformerEncoder`
/// layers (pre-norm, ReLU), a marker scorer and the act head.
#[derive(Debug, Clone, Deserialize)]
pub struct LayaHead {
    /// `type_emb.weight`, [3, hidden].
    pub type_embeddings: String,
    /// `head.layers.`: `nn.TransformerEncoderLayer` tensors (`self_attn.in_proj_weight`, ...).
    pub layers_prefix: String,
    pub layers: usize,
    pub num_heads: usize,
    /// Must be `relu` (PyTorch's default) with `norm_first`.
    pub activation: String,
    pub norm_first: bool,
    pub layer_norm_eps: f32,
    /// `scorer.`: LayerNorm (`0`), Linear (`1`), GELU, Linear (`3`).
    pub scorer_prefix: String,
    pub scorer_norm_eps: f32,
    /// `act_head.`: Linear (`0`), GELU, Linear (`2`), over [CLS state, 4 features].
    pub act_prefix: String,
}

/// `ModernBertForSequenceClassification`: pooling, `head` (dense, activation, norm), `classifier`.
#[derive(Debug, Clone, Deserialize)]
pub struct SequenceClassificationHead {
    /// `head.`: `dense`, `norm`.
    pub head_prefix: String,
    /// `classifier.`
    pub classifier_prefix: String,
    /// `mean` or `cls`.
    pub pooling: String,
    /// Only `gelu`.
    pub activation: String,
    pub classifier_bias: bool,
    pub num_labels: usize,
}

/// `DebertaV2ForSequenceClassification`: `ContextPooler` (dense, activation, on the first token),
/// `classifier`.
#[derive(Debug, Clone, Deserialize)]
pub struct ContextPoolerHead {
    /// `pooler.`: `dense`.
    pub pooler_prefix: String,
    /// `classifier.`
    pub classifier_prefix: String,
    /// Only `gelu`.
    pub activation: String,
    pub num_labels: usize,
}

/// Decision 1.0's `CandidateHead` (`decision/model.py` in the model repo): LayerNorms on the
/// candidate and query states, a bilinear term `key(c) · query(q) / sqrt(head_dim)` and
/// `scalar(gelu(candidate_mlp(c) + query_mlp(q)))`.
#[derive(Debug, Clone, Deserialize)]
pub struct EndpointHead {
    /// Name prefix of the head's tensors (`candidate_norm`, `query_norm`, `key`, `query`,
    /// `candidate_mlp`, `query_mlp`, `scalar`), usually empty.
    pub prefix: String,
    pub head_dim: usize,
    /// `nn.LayerNorm`'s default.
    pub norm_eps: f32,
}

/// Kev's `PointerHead` (`kev/model.py`): `(k(h_opt) · q(h_decide)) / sqrt(dim)`, two Linears with
/// biases. The fitted temperature is the calibration layer's, not the head's.
#[derive(Debug, Clone, Deserialize)]
pub struct PointerHead {
    /// Name prefix of `q` and `k` in the head file.
    pub prefix: String,
    pub dim: usize,
}

/// Von's `OptionMarkerScorer`: LayerNorm, Linear, GELU, LayerNorm, Linear, at every marker.
#[derive(Debug, Clone, Deserialize)]
pub struct OptionMarkerHead {
    /// `scorer.`: `input_norm`, `dense`, `norm`, `out_proj`.
    pub prefix: String,
    pub norm_eps: f32,
}

/// GLiClass `GLiClassUniEncoder`: segment embeddings added before the encoder, label-span mean
/// pooling, two `FeaturesProjector`s and an `MLPScorer`.
#[derive(Debug, Clone, Deserialize)]
pub struct GliclassHead {
    /// `<<SEP>>`: segment 1 starts at its first occurrence (all 1 when there is none).
    pub text_token_id: i32,
    /// `model.segment_embeddings.weight`, [3, hidden].
    pub segment_embeddings: String,
    /// The text embedding: only `first` (the hidden state at 0).
    pub pooling: String,
    /// `model.text_projector.` and `model.classes_projector.`: `linear_1`, activation, `linear_2`.
    pub text_projector: String,
    pub classes_projector: String,
    /// Only `gelu`.
    pub projector_activation: String,
    /// Only `mlp`: `mlp.0`, ReLU, `mlp.2`, ReLU, `mlp.4` over [text, label].
    pub scorer_type: String,
    /// `model.scorer.`
    pub scorer_prefix: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Weights {
    pub format: WeightsFormat,
    /// The upstream file name (a development directory holds it under this name).
    pub file: String,
    /// For `torch-zip`: where each tensor's bytes are (the pickle is never read at run time).
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorAt>,
    /// Further weights files (a head file, an adapter), in the order of the manifest's weights
    /// layers after the first.
    #[serde(default)]
    pub more: Vec<WeightsFile>,
}

/// One more weights file of a model whose tensors are spread over several.
#[derive(Debug, Clone, Deserialize)]
pub struct WeightsFile {
    pub format: WeightsFormat,
    pub file: String,
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorAt>,
}

impl Weights {
    /// Every file, first to last: `(format, file name, torch-zip offsets)`.
    pub fn files(&self) -> Vec<(WeightsFormat, &str, &BTreeMap<String, TensorAt>)> {
        let mut out = vec![(self.format, self.file.as_str(), &self.tensors)];
        out.extend(
            self.more
                .iter()
                .map(|w| (w.format, w.file.as_str(), &w.tensors)),
        );
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightsFormat {
    Safetensors,
    TorchZip,
}

/// One tensor's raw little-endian bytes in the weights file.
#[derive(Debug, Clone, Deserialize)]
pub struct TensorAt {
    /// `F32`, `F16` or `BF16`.
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Absolute byte offset in the file.
    pub offset: u64,
}

impl Arch {
    pub fn read(path: &Path) -> Result<Arch, Error> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        Arch::parse(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
    }

    pub fn parse(text: &str) -> Result<Arch, String> {
        let arch: Arch = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if arch.schema != SCHEMA {
            return Err(format!(
                "arch schema {} (this build reads {SCHEMA}); upgrade ollaya",
                arch.schema
            ));
        }
        match &arch.backbone {
            Backbone::Modernbert(c) => {
                if c.hidden_activation != "gelu" {
                    return Err(format!(
                        "unsupported hidden_activation {:?}",
                        c.hidden_activation
                    ));
                }
                if c.layer_types.len() != c.num_hidden_layers
                    || c.num_attention_heads == 0
                    || c.hidden_size % c.num_attention_heads != 0
                {
                    return Err("inconsistent backbone".into());
                }
                if let Some(t) = c
                    .layer_types
                    .iter()
                    .find(|t| !c.rope_theta.contains_key(*t))
                {
                    return Err(format!("no rope_theta for {t}"));
                }
            }
            Backbone::Qwen3(c) => {
                if c.hidden_act != "silu" {
                    return Err(format!("unsupported hidden_act {:?}", c.hidden_act));
                }
                if !c.tie_word_embeddings || c.use_sliding_window {
                    return Err(
                        "unsupported Qwen3 options (an untied LM head or sliding windows)".into(),
                    );
                }
                if c.num_key_value_heads == 0
                    || c.num_attention_heads % c.num_key_value_heads != 0
                    || c.head_dim % 2 != 0
                {
                    return Err("inconsistent backbone".into());
                }
            }
            Backbone::Qwen35(c) => {
                if c.hidden_act != "silu" {
                    return Err(format!("unsupported hidden_act {:?}", c.hidden_act));
                }
                if !c.tie_word_embeddings || !c.attn_output_gate {
                    return Err(
                        "unsupported Qwen3.5 options (an untied LM head or no attention gate)"
                            .into(),
                    );
                }
                if c.layer_types.len() != c.num_hidden_layers
                    || c.num_key_value_heads == 0
                    || c.num_attention_heads % c.num_key_value_heads != 0
                    || c.linear_num_key_heads == 0
                    || c.linear_num_value_heads % c.linear_num_key_heads != 0
                    || c.linear_conv_kernel_dim == 0
                {
                    return Err("inconsistent backbone".into());
                }
                let rotary = (c.head_dim as f32 * c.partial_rotary_factor) as usize;
                if rotary == 0 || !rotary.is_multiple_of(2) || rotary > c.head_dim {
                    return Err(format!(
                        "unsupported partial_rotary_factor {}",
                        c.partial_rotary_factor
                    ));
                }
            }
            Backbone::DebertaV2(c) => {
                if c.hidden_act != "gelu" {
                    return Err(format!("unsupported hidden_act {:?}", c.hidden_act));
                }
                if !c.share_att_key
                    || c.pos_att_type.is_empty()
                    || c.pos_att_type.iter().any(|a| a != "c2p" && a != "p2c")
                {
                    return Err(format!(
                        "unsupported attention: pos_att_type {:?}, share_att_key {}",
                        c.pos_att_type, c.share_att_key
                    ));
                }
                if c.position_biased_input || c.type_vocab_size != 0 || c.conv_kernel_size != 0 {
                    return Err("unsupported DeBERTa options (absolute positions, token types or a conv layer)".into());
                }
                if !matches!(c.norm_rel_ebd.as_str(), "layer_norm" | "none") {
                    return Err(format!("unsupported norm_rel_ebd {:?}", c.norm_rel_ebd));
                }
                if c.num_attention_heads == 0
                    || c.hidden_size % c.num_attention_heads != 0
                    || (c.position_buckets == 0 && c.max_relative_positions == 0)
                {
                    return Err("inconsistent backbone".into());
                }
            }
        }
        Ok(arch)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn arch(activation: &str, schema: u32) -> String {
        json!({
            "schema": schema,
            "backbone": {
                "type": "modernbert", "prefix": "model.", "vocab_size": 10, "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 2,
                "norm_eps": 1e-5, "norm_bias": false, "attention_bias": false, "mlp_bias": false,
                "hidden_activation": activation, "local_attention": 4,
                "layer_types": ["full_attention", "sliding_attention"],
                "rope_theta": {"full_attention": 160000.0, "sliding_attention": 10000.0}
            },
            "head": {"type": "option-marker", "prefix": "scorer.", "norm_eps": 1e-5},
            "weights": {"format": "torch-zip", "file": "option_marker.pt",
                        "tensors": {"scorer.dense.weight": {"dtype": "F32", "shape": [4, 8], "offset": 64}}}
        })
        .to_string()
    }

    #[test]
    fn reads_what_arch_py_writes() {
        let a = Arch::parse(&arch("gelu", SCHEMA)).unwrap();
        assert_eq!(a.head.name(), "option-marker");
        assert_eq!(a.weights.format, WeightsFormat::TorchZip);
        assert_eq!(a.weights.tensors["scorer.dense.weight"].offset, 64);
    }

    #[test]
    fn reads_a_deberta_arch() {
        let text = json!({
            "schema": SCHEMA,
            "backbone": {
                "type": "deberta-v2", "prefix": "deberta.", "vocab_size": 10, "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 2,
                "layer_norm_eps": 1e-7, "hidden_act": "gelu", "pos_att_type": ["p2c", "c2p"],
                "share_att_key": true, "position_buckets": 4, "max_relative_positions": 8,
                "norm_rel_ebd": "layer_norm", "position_biased_input": false, "type_vocab_size": 0,
                "conv_kernel_size": 0
            },
            "head": {"type": "context-pooler", "pooler_prefix": "pooler.", "classifier_prefix": "classifier.",
                     "activation": "gelu", "num_labels": 2},
            "weights": {"format": "safetensors", "file": "model.safetensors"}
        })
        .to_string();
        let a = Arch::parse(&text).unwrap();
        assert_eq!(a.head.name(), "sequence-classification");
        assert_eq!(a.backbone.hidden_size(), 8);
        let conv = text.replace("\"conv_kernel_size\":0", "\"conv_kernel_size\":3");
        assert!(Arch::parse(&conv).unwrap_err().contains("conv"));
    }

    #[test]
    fn reads_a_qwen3_arch() {
        let text = json!({
            "schema": SCHEMA,
            "backbone": {
                "type": "qwen3", "prefix": "model.", "vocab_size": 10, "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 2, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
                "attention_bias": false, "hidden_act": "silu", "tie_word_embeddings": true,
                "use_sliding_window": false
            },
            "head": {"type": "tied-rows"},
            "weights": {"format": "safetensors", "file": "model.safetensors"}
        })
        .to_string();
        let a = Arch::parse(&text).unwrap();
        assert_eq!(a.head.name(), "tied-rows");
        assert!(a.backbone.is_decoder());
        let untied = text.replace(
            "\"tie_word_embeddings\":true",
            "\"tie_word_embeddings\":false",
        );
        assert!(Arch::parse(&untied).unwrap_err().contains("untied"));
    }

    #[test]
    fn reads_several_weights_files_and_a_lora() {
        let text = json!({
            "schema": SCHEMA,
            "backbone": {
                "type": "qwen3-5", "prefix": "model.language_model.", "vocab_size": 10, "hidden_size": 8,
                "intermediate_size": 12, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "partial_rotary_factor": 0.5,
                "rope_theta": 1e7, "attention_bias": false, "attn_output_gate": true, "linear_num_key_heads": 2,
                "linear_num_value_heads": 2, "linear_key_head_dim": 4, "linear_value_head_dim": 4,
                "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "hidden_act": "silu", "tie_word_embeddings": true
            },
            "head": {"type": "pointer", "prefix": "head.", "dim": 256},
            "weights": {"format": "safetensors", "file": "model.safetensors",
                        "more": [{"format": "safetensors", "file": "adapter_model.safetensors"},
                                 {"format": "torch-zip", "file": "head.pt",
                                  "tensors": {"head.q.weight": {"dtype": "F32", "shape": [256, 8], "offset": 64}}}]},
            "lora": {"prefix": "base_model.model.", "scaling": 2.0}
        })
        .to_string();
        let a = Arch::parse(&text).unwrap();
        assert_eq!(a.head.name(), "pointer");
        assert_eq!(a.backbone.prefix(), "model.language_model.");
        let files = a.weights.files();
        assert_eq!(files.len(), 3);
        assert_eq!(
            (files[2].0, files[2].1),
            (WeightsFormat::TorchZip, "head.pt")
        );
        assert_eq!(files[2].2["head.q.weight"].offset, 64);
        assert_eq!(a.lora.as_ref().map(|l| l.scaling), Some(2.0));
    }

    #[test]
    fn refuses_what_it_cannot_run() {
        assert!(
            Arch::parse(&arch("gelu", SCHEMA + 1))
                .unwrap_err()
                .contains("upgrade")
        );
        assert!(
            Arch::parse(&arch("silu", SCHEMA))
                .unwrap_err()
                .contains("silu")
        );
    }
}
