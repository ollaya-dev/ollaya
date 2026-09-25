//! The decision heads on top of the encoder, one per layout, each written against the author's
//! own module (named on each type). They compute in float32 on the GPU.

use ollaya_mlx::{Array, Dtype};

use super::arch::{GliclassHead, LayaHead, OptionMarkerHead, SequenceClassificationHead};
use super::modernbert::{Linear, Norm, gelu, relu};
use super::weights::{Tensors, mlx};
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

/// Masked marker slots get this logit, as every exported graph does.
const MASKED: f32 = -1e4;

/// `h[r, marker_pos[r, c], :]`: [rows, markers, hidden].
fn gather_markers(h: &Array, marker_pos: &Array) -> Result<Array> {
    let (s, p) = (h.shape(), marker_pos.shape());
    let idx = marker_pos
        .reshape(&[p[0], p[1], 1])
        .and_then(|i| i.broadcast_to(&[p[0], p[1], s[2]]))
        .map_err(mlx)?;
    h.take_along_axis(&idx, 1).map_err(mlx)
}

/// `logits.masked_fill(~mask, -1e4)`.
fn mask_logits(logits: &Array, marker_mask: &Array) -> Result<Array> {
    Array::select(marker_mask, logits, &Array::scalar(MASKED)).map_err(mlx)
}

/// `h[:, 0]`: [rows, hidden].
fn first_token(h: &Array) -> Result<Array> {
    let s = h.shape();
    h.slice(&[0, 0, 0], &[s[0], 1, s[2]])
        .and_then(|a| a.reshape(&[s[0], s[2]]))
        .map_err(mlx)
}

/// One `nn.TransformerEncoderLayer(d, heads, ffn, batch_first=True, norm_first=True)` in eval
/// mode: `x + SA(norm1(x))`, then `x + linear2(relu(linear1(norm2(x))))`.
struct TorchEncoderLayer {
    norm1: Norm,
    in_proj: Linear,
    out_proj: Linear,
    norm2: Norm,
    linear1: Linear,
    linear2: Linear,
}

impl TorchEncoderLayer {
    fn load(t: &Tensors, p: &str, d: usize, eps: f32) -> Result<Self> {
        let ffn = t
            .shape(&format!("{p}linear1.weight"))
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {p}linear1.weight")))?;
        // `in_proj_weight` / `in_proj_bias`: the q, k and v projections, stacked.
        let in_proj = Linear::load_named(
            t,
            &format!("{p}self_attn.in_proj_weight"),
            Some(&format!("{p}self_attn.in_proj_bias")),
            3 * d,
            d,
        )?;
        Ok(TorchEncoderLayer {
            norm1: Norm::load(t, &format!("{p}norm1."), d, true, eps)?,
            in_proj,
            out_proj: Linear::load(t, &format!("{p}self_attn.out_proj."), d, d, true)?,
            norm2: Norm::load(t, &format!("{p}norm2."), d, true, eps)?,
            linear1: Linear::load(t, &format!("{p}linear1."), ffn, d, true)?,
            linear2: Linear::load(t, &format!("{p}linear2."), d, ffn, true)?,
        })
    }

    fn forward(&self, x: &Array, keys: &Array, heads: i32) -> Result<Array> {
        let s = x.shape();
        let (rows, seq, d) = (s[0], s[1], s[2]);
        let hd = d / heads;
        let qkv = self
            .in_proj
            .forward(&self.norm1.forward(x)?)?
            .split(3, -1)
            .map_err(mlx)?;
        let heads_of = |a: &Array| {
            a.reshape(&[rows, seq, heads, hd])
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .map_err(mlx)
        };
        let (q, k, v) = (heads_of(&qkv[0])?, heads_of(&qkv[1])?, heads_of(&qkv[2])?);
        let attn = Array::attention(&q, &k, &v, (hd as f32).powf(-0.5), Some(keys))
            .and_then(|a| a.transpose(&[0, 2, 1, 3]))
            .and_then(|a| a.reshape(&[rows, seq, d]))
            .map_err(mlx)?;
        let x = x.add(&self.out_proj.forward(&attn)?).map_err(mlx)?;
        let ff = self
            .linear2
            .forward(&relu(&self.linear1.forward(&self.norm2.forward(&x)?)?)?)?;
        x.add(&ff).map_err(mlx)
    }
}

/// `laya.common.DecisionModel` after the encoder (laya 0.3.7).
pub struct Laya {
    type_emb: Array,
    layers: Vec<TorchEncoderLayer>,
    heads: i32,
    scorer_norm: Norm,
    scorer_1: Linear,
    scorer_2: Linear,
    act_1: Linear,
    act_2: Linear,
}

impl Laya {
    pub fn load(c: &LayaHead, d: usize, t: &Tensors) -> Result<Laya> {
        if c.activation != "relu" || !c.norm_first {
            return Err(Error::Model(format!(
                "unsupported laya head layers: activation {:?}, norm_first {}",
                c.activation, c.norm_first
            )));
        }
        let layers = (0..c.layers)
            .map(|i| {
                TorchEncoderLayer::load(t, &format!("{}{i}.", c.layers_prefix), d, c.layer_norm_eps)
            })
            .collect::<Result<_>>()?;
        let sp = &c.scorer_prefix;
        let ap = &c.act_prefix;
        let act_hidden = t
            .shape(&format!("{ap}0.weight"))
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {ap}0.weight")))?;
        let act_labels = t
            .shape(&format!("{ap}2.weight"))
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {ap}2.weight")))?;
        Ok(Laya {
            type_emb: t.get(&c.type_embeddings, &[3, d])?,
            layers,
            heads: c.num_heads as i32,
            scorer_norm: Norm::load(t, &format!("{sp}0."), d, true, c.scorer_norm_eps)?,
            scorer_1: Linear::load(t, &format!("{sp}1."), d, d, true)?,
            scorer_2: Linear::load(t, &format!("{sp}3."), 1, d, true)?,
            // [CLS state, top1, top1 - top2, entropy, k / 255]
            act_1: Linear::load(t, &format!("{ap}0."), act_hidden, d + 4, true)?,
            act_2: Linear::load(t, &format!("{ap}2."), act_labels, act_hidden, true)?,
        })
    }

    /// Option logits [rows, markers] and act logits [rows, act labels].
    pub fn forward(
        &self,
        h: &Array,
        attention: &Array,
        qtype: &Array,
        marker_pos: &Array,
        marker_mask: &Array,
    ) -> Result<(Array, Array)> {
        let s = h.shape();
        let (rows, seq) = (s[0], s[1]);
        let mut h = h
            .add(
                &self
                    .type_emb
                    .take(qtype, 0)
                    .and_then(|e| e.expand_dims(1))
                    .map_err(mlx)?,
            )
            .map_err(mlx)?;
        // `src_key_padding_mask`: keys only.
        let keys = attention
            .reshape(&[rows, 1, 1, seq])
            .and_then(|k| k.broadcast_to(&[rows, 1, seq, seq]))
            .map_err(mlx)?;
        for layer in &self.layers {
            h = layer.forward(&h, &keys, self.heads)?;
        }

        let m = gather_markers(&h, marker_pos)?;
        let k = marker_pos.shape()[1];
        let logits = self
            .scorer_2
            .forward(&gelu(
                &self.scorer_1.forward(&self.scorer_norm.forward(&m)?)?,
            )?)?
            .reshape(&[rows, k])
            .map_err(mlx)?;
        let logits = mask_logits(&logits, marker_mask)?;

        // The act head's features, from the softmax over all marker slots (masked ones ~0).
        let p = logits.softmax(-1).map_err(mlx)?;
        let n = marker_mask
            .astype(Dtype::Float32)
            .and_then(|m| m.sum(-1, true))
            .and_then(|m| m.maximum(&Array::scalar(2.0)))
            .map_err(mlx)?;
        let ent = p
            .maximum(&Array::scalar(1e-9))
            .and_then(|q| q.log())
            .and_then(|l| l.multiply(&p))
            .and_then(|pl| pl.sum(-1, true))
            .and_then(|s| s.multiply(&Array::scalar(-1.0)))
            .and_then(|s| s.divide(&n.log()?))
            .map_err(mlx)?;
        let sorted = p.sort(-1).map_err(mlx)?;
        let top1 = sorted.slice(&[0, k - 1], &[rows, k]).map_err(mlx)?;
        let top2 = if k >= 2 {
            sorted.slice(&[0, k - 2], &[rows, k - 1]).map_err(mlx)?
        } else {
            // One slot: laya pads the second value with 0.
            Array::from_slice(&vec![0.0f32; rows as usize], &[rows, 1]).map_err(mlx)?
        };
        let feats = Array::concatenate(
            &[
                &first_token(&h)?,
                &top1,
                &top1.subtract(&top2).map_err(mlx)?,
                &ent,
                &n.multiply(&Array::scalar(1.0 / 255.0)).map_err(mlx)?,
            ],
            -1,
        )
        .map_err(mlx)?;
        let act = self.act_2.forward(&gelu(&self.act_1.forward(&feats)?)?)?;
        Ok((logits, act))
    }
}

/// `ModernBertForSequenceClassification` after the encoder: pooling, `head`, `classifier`.
pub struct SequenceClassification {
    mean: bool,
    dense: Linear,
    norm: Norm,
    classifier: Linear,
}

impl SequenceClassification {
    pub fn load(
        c: &SequenceClassificationHead,
        d: usize,
        norm_bias: bool,
        norm_eps: f32,
        t: &Tensors,
    ) -> Result<Self> {
        if c.activation != "gelu" || !matches!(c.pooling.as_str(), "mean" | "cls") {
            return Err(Error::Model(format!(
                "unsupported classification head: pooling {:?}, activation {:?}",
                c.pooling, c.activation
            )));
        }
        let hp = &c.head_prefix;
        Ok(SequenceClassification {
            mean: c.pooling == "mean",
            dense: Linear::load(t, &format!("{hp}dense."), d, d, c.classifier_bias)?,
            norm: Norm::load(t, &format!("{hp}norm."), d, norm_bias, norm_eps)?,
            classifier: Linear::load(t, &c.classifier_prefix, c.num_labels, d, true)?,
        })
    }

    /// Class logits [rows, labels].
    pub fn forward(&self, h: &Array, attention: &Array) -> Result<Array> {
        let pooled = if self.mean {
            let s = h.shape();
            let m = attention
                .astype(Dtype::Float32)
                .and_then(|m| m.reshape(&[s[0], s[1], 1]))
                .map_err(mlx)?;
            h.multiply(&m)
                .and_then(|x| x.sum(1, false))
                .and_then(|x| x.divide(&m.sum(1, false)?))
                .map_err(mlx)?
        } else {
            first_token(h)?
        };
        let x = self.norm.forward(&gelu(&self.dense.forward(&pooled)?)?)?;
        self.classifier.forward(&x)
    }
}

/// Von's `OptionMarkerScorer` at every marker (von-sdk 1.1.1).
pub struct OptionMarker {
    input_norm: Norm,
    dense: Linear,
    norm: Norm,
    out_proj: Linear,
}

impl OptionMarker {
    pub fn load(c: &OptionMarkerHead, d: usize, t: &Tensors) -> Result<Self> {
        let p = &c.prefix;
        let half = t
            .shape(&format!("{p}dense.weight"))
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {p}dense.weight")))?;
        Ok(OptionMarker {
            input_norm: Norm::load(t, &format!("{p}input_norm."), d, true, c.norm_eps)?,
            dense: Linear::load(t, &format!("{p}dense."), half, d, true)?,
            norm: Norm::load(t, &format!("{p}norm."), half, true, c.norm_eps)?,
            out_proj: Linear::load(t, &format!("{p}out_proj."), 1, half, true)?,
        })
    }

    /// Marker logits [rows, markers].
    pub fn forward(&self, h: &Array, marker_pos: &Array, marker_mask: &Array) -> Result<Array> {
        let m = gather_markers(h, marker_pos)?;
        let x = self
            .norm
            .forward(&gelu(&self.dense.forward(&self.input_norm.forward(&m)?)?)?)?;
        let p = marker_pos.shape();
        let logits = self
            .out_proj
            .forward(&x)?
            .reshape(&[p[0], p[1]])
            .map_err(mlx)?;
        mask_logits(&logits, marker_mask)
    }
}

/// GLiClass `GLiClassUniEncoder` around the encoder (gliclass 0.1.20).
pub struct Gliclass {
    pub text_token: i32,
    pub segments: Array,
    text_1: Linear,
    text_2: Linear,
    classes_1: Linear,
    classes_2: Linear,
    mlp: [Linear; 3],
}

impl Gliclass {
    pub fn load(c: &GliclassHead, d: usize, t: &Tensors) -> Result<Self> {
        if c.pooling != "first" || c.projector_activation != "gelu" || c.scorer_type != "mlp" {
            return Err(Error::Model(format!(
                "unsupported GLiClass head: pooling {:?}, projector {:?}, scorer {:?}",
                c.pooling, c.projector_activation, c.scorer_type
            )));
        }
        let dim = |name: &str| {
            t.shape(name)
                .map(|s| s[0])
                .ok_or_else(|| Error::Model(format!("no tensor {name}")))
        };
        let proj_hidden = dim(&format!("{}linear_1.weight", c.text_projector))?;
        let sp = &c.scorer_prefix;
        let (m0, m1) = (
            dim(&format!("{sp}mlp.0.weight"))?,
            dim(&format!("{sp}mlp.2.weight"))?,
        );
        let segments = t
            .shape(&c.segment_embeddings)
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {}", c.segment_embeddings)))?;
        Ok(Gliclass {
            text_token: c.text_token_id,
            segments: t.get(&c.segment_embeddings, &[segments, d])?,
            text_1: Linear::load(
                t,
                &format!("{}linear_1.", c.text_projector),
                proj_hidden,
                d,
                true,
            )?,
            text_2: Linear::load(
                t,
                &format!("{}linear_2.", c.text_projector),
                d,
                proj_hidden,
                true,
            )?,
            classes_1: Linear::load(
                t,
                &format!("{}linear_1.", c.classes_projector),
                proj_hidden,
                d,
                true,
            )?,
            classes_2: Linear::load(
                t,
                &format!("{}linear_2.", c.classes_projector),
                d,
                proj_hidden,
                true,
            )?,
            mlp: [
                Linear::load(t, &format!("{sp}mlp.0."), m0, 2 * d, true)?,
                Linear::load(t, &format!("{sp}mlp.2."), m1, m0, true)?,
                Linear::load(t, &format!("{sp}mlp.4."), 1, m1, true)?,
            ],
        })
    }

    /// Segment ids [rows, seq]: 1 from the first `<<SEP>>` on, 0 before it (all 1 without one,
    /// upstream's `argmax` semantics). Computed on the host from the ids.
    pub fn segment_ids(&self, ids: &[i32], rows: usize, seq: usize) -> Vec<i32> {
        let mut out = vec![0; rows * seq];
        for r in 0..rows {
            let row = &ids[r * seq..(r + 1) * seq];
            let first = row.iter().position(|&t| t == self.text_token).unwrap_or(0);
            for v in &mut out[r * seq + first..(r + 1) * seq] {
                *v = 1;
            }
        }
        out
    }

    /// Label logits [rows, markers]: label k is the mean of the hidden states from marker k up to
    /// marker k+1 (the last label up to the end of the row), padding excluded.
    pub fn forward(
        &self,
        h: &Array,
        attention: &Array,
        marker_pos: &Array,
        marker_mask: &Array,
    ) -> Result<Array> {
        let (s, p) = (h.shape(), marker_pos.shape());
        let (rows, seq, k) = (s[0], s[1], p[1]);
        let pos = Array::arange(0, seq, Dtype::Int32)
            .and_then(|a| a.reshape(&[1, 1, seq]))
            .map_err(mlx)?;
        // started[r, c, t]: marker c is at or before t (and is a real marker).
        let started = marker_pos
            .reshape(&[rows, k, 1])
            .and_then(|m| m.less_equal(&pos))
            .and_then(|a| a.logical_and(&marker_mask.reshape(&[rows, k, 1])?))
            .map_err(mlx)?;
        // Labels opened at or before t, [rows, 1, seq]; position t belongs to label cum - 1.
        let cum = started
            .astype(Dtype::Int32)
            .and_then(|a| a.sum(1, true))
            .map_err(mlx)?;
        let label = Array::arange(1, k + 1, Dtype::Int32)
            .and_then(|a| a.reshape(&[1, k, 1]))
            .map_err(mlx)?;
        let span = cum
            .equal(&label)
            .and_then(|a| a.logical_and(&attention.reshape(&[rows, 1, seq])?))
            .and_then(|a| a.astype(Dtype::Float32))
            .map_err(mlx)?;
        let count = span
            .sum(-1, true)
            .and_then(|c| c.maximum(&Array::scalar(1.0)))
            .map_err(mlx)?;
        let classes = span.matmul(h).and_then(|c| c.divide(&count)).map_err(mlx)?;

        let text = self
            .text_2
            .forward(&gelu(&self.text_1.forward(&first_token(h)?)?)?)?;
        let classes = self
            .classes_2
            .forward(&gelu(&self.classes_1.forward(&classes)?)?)?;
        let d = s[2];
        let text = text
            .reshape(&[rows, 1, d])
            .and_then(|t| t.broadcast_to(&[rows, k, d]))
            .map_err(mlx)?;
        let x = Array::concatenate(&[&text, &classes], -1).map_err(mlx)?;
        let x = relu(&self.mlp[0].forward(&x)?)?;
        let x = relu(&self.mlp[1].forward(&x)?)?;
        let logits = self.mlp[2].forward(&x)?.reshape(&[rows, k]).map_err(mlx)?;
        mask_logits(&logits, marker_mask)
    }
}
