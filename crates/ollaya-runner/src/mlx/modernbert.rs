//! ModernBERT in fp32 on MLX, written against transformers 5.17's `modeling_modernbert.py`.
//!
//! Parity notes (each was measured; see `docs/decisions/0001-mlx-engine.md`):
//! * GELU is the exact (erf) form, as `ACT2FN["gelu"]`; the tanh approximation that
//!   mlx-embeddings uses moves probabilities by 8e-3.
//! * Rotary embeddings: `mx.fast.rope` with `traditional = false` (the two halves rotate), which
//!   matches Hugging Face's `rotate_half`.
//! * Attention: MLX's fused SDPA with a boolean mask over keys. A query row that may see no key
//!   gives NaN there and, through the values of the next layer, everywhere; so padding query rows
//!   of the sliding-window layers get the full key mask (their outputs are never read).
//! * No `mx.compile`: kernels compiled at run time use fast math on macOS 15+ (mlx#4553).

use ollaya_mlx::{Array, Dtype};

use super::arch::ModernBertConfig;
use super::weights::{Tensors, mlx};
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

/// `y = x W^T (+ b)` with the checkpoint's `[out, in]` weight.
pub struct Linear {
    w_t: Array,
    b: Option<Array>,
}

impl Linear {
    /// `{prefix}weight` [out, in] and, with `bias`, `{prefix}bias` [out].
    pub fn load(t: &Tensors, prefix: &str, out: usize, inp: usize, bias: bool) -> Result<Linear> {
        let w = t.get(&format!("{prefix}weight"), &[out, inp])?;
        let b = if bias {
            Some(t.get(&format!("{prefix}bias"), &[out])?)
        } else {
            None
        };
        Ok(Linear {
            w_t: w.transpose(&[1, 0]).map_err(mlx)?,
            b,
        })
    }

    /// A Linear whose tensors are not named `{prefix}weight` / `{prefix}bias`.
    pub fn load_named(
        t: &Tensors,
        weight: &str,
        bias: Option<&str>,
        out: usize,
        inp: usize,
    ) -> Result<Linear> {
        Ok(Linear {
            w_t: t
                .get(weight, &[out, inp])?
                .transpose(&[1, 0])
                .map_err(mlx)?,
            b: bias.map(|b| t.get(b, &[out])).transpose()?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let y = x.matmul(&self.w_t).map_err(mlx)?;
        match &self.b {
            Some(b) => y.add(b).map_err(mlx),
            None => Ok(y),
        }
    }
}

/// `nn.LayerNorm` over the last axis.
pub struct Norm {
    w: Array,
    b: Option<Array>,
    eps: f32,
}

impl Norm {
    pub fn load(t: &Tensors, prefix: &str, dim: usize, bias: bool, eps: f32) -> Result<Norm> {
        Ok(Norm {
            w: t.get(&format!("{prefix}weight"), &[dim])?,
            b: if bias {
                Some(t.get(&format!("{prefix}bias"), &[dim])?)
            } else {
                None
            },
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        x.layer_norm(Some(&self.w), self.b.as_ref(), self.eps)
            .map_err(mlx)
    }
}

/// Exact GELU, as PyTorch computes it: `x * 0.5 * (1 + erf(x / sqrt(2)))`.
pub fn gelu(x: &Array) -> Result<Array> {
    let inner = x
        .multiply(&Array::scalar(std::f32::consts::FRAC_1_SQRT_2))
        .and_then(|v| v.erf())
        .and_then(|v| v.add(&Array::scalar(1.0)))
        .map_err(mlx)?;
    x.multiply(&Array::scalar(0.5))
        .and_then(|v| v.multiply(&inner))
        .map_err(mlx)
}

pub fn relu(x: &Array) -> Result<Array> {
    x.maximum(&Array::scalar(0.0)).map_err(mlx)
}

struct Layer {
    attn_norm: Option<Norm>,
    wqkv: Linear,
    wo: Linear,
    mlp_norm: Norm,
    wi: Linear,
    mlp_wo: Linear,
    sliding: bool,
    theta: f32,
}

pub struct ModernBert {
    pub hidden: usize,
    heads: usize,
    head_dim: usize,
    half_window: i32,
    tok_embeddings: Array,
    emb_norm: Norm,
    layers: Vec<Layer>,
    final_norm: Norm,
}

impl ModernBert {
    pub fn load(c: &ModernBertConfig, t: &Tensors) -> Result<ModernBert> {
        let (d, i, p) = (c.hidden_size, c.intermediate_size, &c.prefix);
        let norm = |name: &str| Norm::load(t, &format!("{p}{name}."), d, c.norm_bias, c.norm_eps);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (n, kind) in c.layer_types.iter().enumerate() {
            let lp = format!("layers.{n}.");
            let sliding = match kind.as_str() {
                "sliding_attention" => true,
                "full_attention" => false,
                other => return Err(Error::Model(format!("unknown layer type {other:?}"))),
            };
            layers.push(Layer {
                // Layer 0 has no attention norm (`nn.Identity`).
                attn_norm: if n == 0 {
                    None
                } else {
                    Some(norm(&format!("{lp}attn_norm"))?)
                },
                wqkv: Linear::load(t, &format!("{p}{lp}attn.Wqkv."), 3 * d, d, c.attention_bias)?,
                wo: Linear::load(t, &format!("{p}{lp}attn.Wo."), d, d, c.attention_bias)?,
                mlp_norm: norm(&format!("{lp}mlp_norm"))?,
                wi: Linear::load(t, &format!("{p}{lp}mlp.Wi."), 2 * i, d, c.mlp_bias)?,
                mlp_wo: Linear::load(t, &format!("{p}{lp}mlp.Wo."), d, i, c.mlp_bias)?,
                sliding,
                theta: c.rope_theta[kind],
            });
        }
        Ok(ModernBert {
            hidden: d,
            heads: c.num_attention_heads,
            head_dim: d / c.num_attention_heads,
            half_window: (c.local_attention / 2) as i32,
            tok_embeddings: t.get(
                &format!("{p}embeddings.tok_embeddings.weight"),
                &[c.vocab_size, d],
            )?,
            emb_norm: norm("embeddings.norm")?,
            layers,
            final_norm: norm("final_norm")?,
        })
    }

    /// Token embeddings before the embedding norm: [rows, seq, hidden].
    pub fn embed(&self, ids: &Array) -> Result<Array> {
        self.tok_embeddings.take(ids, 0).map_err(mlx)
    }

    /// The encoder over raw embeddings `x` [rows, seq, hidden] (from [`Self::embed`], plus any
    /// extra embedding the head adds), with `attention` [rows, seq] bool. Returns the final
    /// normed hidden states, as `ModernBertModel(...).last_hidden_state`.
    pub fn forward(&self, x: &Array, attention: &Array) -> Result<Array> {
        let shape = x.shape();
        let (rows, seq) = (shape[0], shape[1]);
        let (h, hd, d) = (self.heads as i32, self.head_dim as i32, self.hidden as i32);

        // Keys every query may attend to, [rows, 1, seq, seq].
        let keys = attention
            .reshape(&[rows, 1, 1, seq])
            .and_then(|k| k.broadcast_to(&[rows, 1, seq, seq]))
            .map_err(mlx)?;
        let sliding_mask = if self.layers.iter().any(|l| l.sliding) {
            // |i - j| <= local_attention / 2, and padding queries see every key (never read, but
            // a fully masked row would be NaN).
            let pos = Array::arange(0, seq, Dtype::Int32).map_err(mlx)?;
            let dist = pos
                .reshape(&[seq, 1])
                .and_then(|a| a.subtract(&pos.reshape(&[1, seq])?))
                .and_then(|a| a.abs())
                .map_err(mlx)?;
            let window = dist
                .less_equal(&Array::from_slice(&[self.half_window], &[]).map_err(mlx)?)
                .and_then(|w| w.reshape(&[1, 1, seq, seq]))
                .map_err(mlx)?;
            let pad_query = attention
                .reshape(&[rows, 1, seq, 1])
                .and_then(|q| q.logical_not())
                .map_err(mlx)?;
            Some(
                window
                    .logical_or(&pad_query)
                    .and_then(|w| w.logical_and(&keys))
                    .map_err(mlx)?,
            )
        } else {
            None
        };

        let scale = (self.head_dim as f32).powf(-0.5);
        let mut x = self.emb_norm.forward(x)?;
        for layer in &self.layers {
            let normed = match &layer.attn_norm {
                Some(n) => n.forward(&x)?,
                None => x.clone(),
            };
            // [rows, seq, 3, heads, head_dim] -> 3 x [rows, heads, seq, head_dim]
            let qkv = layer
                .wqkv
                .forward(&normed)?
                .reshape(&[rows, seq, 3, h, hd])
                .and_then(|a| a.transpose(&[2, 0, 3, 1, 4]))
                .and_then(|a| a.split(3, 0))
                .map_err(mlx)?;
            let part = |i: usize| qkv[i].squeeze(0).map_err(mlx);
            let rope = |a: Array| a.rope(hd, false, layer.theta, 1.0, 0).map_err(mlx);
            let (q, k, v) = (rope(part(0)?)?, rope(part(1)?)?, part(2)?);
            let mask = if layer.sliding {
                sliding_mask.as_ref().expect("built when a layer slides")
            } else {
                &keys
            };
            let attn = Array::attention(&q, &k, &v, scale, Some(mask))
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .and_then(|a| a.reshape(&[rows, seq, d]))
                .map_err(mlx)?;
            x = x.add(&layer.wo.forward(&attn)?).map_err(mlx)?;

            let up = layer.wi.forward(&layer.mlp_norm.forward(&x)?)?;
            let halves = up.split(2, -1).map_err(mlx)?;
            let glu = gelu(&halves[0])?.multiply(&halves[1]).map_err(mlx)?;
            x = x.add(&layer.mlp_wo.forward(&glu)?).map_err(mlx)?;
        }
        self.final_norm.forward(&x)
    }
}
