//! Qwen3 (the dense decoder) in fp32 on MLX, written against transformers 5.17's
//! `modeling_qwen3.py` and the export trunk `convert/ollaya_convert/families/llm_common/qwen3.py`:
//! RMSNorm, grouped-query attention with a RMSNorm per head on q and k, RoPE over the whole head,
//! a causal mask, a SwiGLU MLP. Rows are right-padded; every layer is causal, so pads never reach
//! a readout position and no key mask is needed.
//!
//! Parity notes (`docs/decisions/0001-mlx-engine.md`):
//! * RoPE tables are ours (f64 angles from the fp32 `inv_freq`), as for ModernBERT.
//! * Attention is MLX's fused SDPA with the boolean causal mask; k and v are repeated per group as
//!   the reference's `repeat_interleave` does, so the kernel sees plain multi-head inputs.
//! * The RMSNorms are `mlx.fast.rms_norm`, which computes `x * rsqrt(mean(x²) + eps) * w` in
//!   float32, the reference's arithmetic.

use std::cell::RefCell;

use ollaya_mlx::{Array, Dtype};

use super::arch::Qwen3Config;
use super::modernbert::{Linear, Rotary, rotate, silu};
use super::weights::{Tensors, mlx};
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

/// `Qwen3RMSNorm`: `w * x * rsqrt(mean(x²) + eps)`.
pub struct RmsNorm {
    w: Array,
    eps: f32,
}

impl RmsNorm {
    pub fn load(t: &Tensors, name: &str, dim: usize, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            w: t.get(name, &[dim])?,
            eps,
        })
    }

    /// A norm whose stored weight is an offset from one (`Qwen3_5RMSNorm`: `(1 + w)`).
    pub fn load_zero_centered(t: &Tensors, name: &str, dim: usize, eps: f32) -> Result<RmsNorm> {
        Ok(RmsNorm {
            w: t.get(name, &[dim])?.add(&Array::scalar(1.0)).map_err(mlx)?,
            eps,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        x.rms_norm(&self.w, self.eps).map_err(mlx)
    }
}

/// `Qwen3MLP`: `down(silu(gate(x)) * up(x))`.
pub struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn load(t: &Tensors, prefix: &str, hidden: usize, intermediate: usize) -> Result<Mlp> {
        Ok(Mlp {
            gate: Linear::load(
                t,
                &format!("{prefix}gate_proj."),
                intermediate,
                hidden,
                false,
            )?,
            up: Linear::load(t, &format!("{prefix}up_proj."), intermediate, hidden, false)?,
            down: Linear::load(
                t,
                &format!("{prefix}down_proj."),
                hidden,
                intermediate,
                false,
            )?,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let g = silu(&self.gate.forward(x)?)?;
        self.down
            .forward(&g.multiply(&self.up.forward(x)?).map_err(mlx)?)
    }
}

/// The boolean causal mask [1, 1, seq, seq]: query i sees keys 0..=i.
pub fn causal_mask(seq: i32) -> Result<Array> {
    Array::full(&[seq, seq], 1.0)
        .and_then(|m| m.tril(0))
        .and_then(|m| m.astype(Dtype::Bool))
        .and_then(|m| m.reshape(&[1, 1, seq, seq]))
        .map_err(mlx)
}

/// Rotary tables for one base, grown to the longest sequence seen (in powers of two).
pub struct RotaryCache {
    theta: f32,
    dim: usize,
    table: RefCell<Option<Rotary>>,
}

impl RotaryCache {
    pub fn new(theta: f32, dim: usize) -> RotaryCache {
        RotaryCache {
            theta,
            dim,
            table: RefCell::new(None),
        }
    }

    /// cos and sin for positions `0..seq`, [seq, dim] each.
    pub fn get(&self, seq: i32) -> Result<(Array, Array)> {
        let mut table = self.table.borrow_mut();
        if table.as_ref().is_none_or(|t| t.len < seq) {
            let len = (seq.max(128) as u32).next_power_of_two() as i32;
            *table = Some(Rotary::new(self.theta, self.dim, len)?);
        }
        let (t, d) = (table.as_ref().expect("filled above"), self.dim as i32);
        Ok((
            t.cos.slice(&[0, 0], &[seq, d]).map_err(mlx)?,
            t.sin.slice(&[0, 0], &[seq, d]).map_err(mlx)?,
        ))
    }
}

struct Layer {
    input_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_norm: RmsNorm,
    mlp: Mlp,
}

pub struct Qwen3 {
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    /// `embed_tokens.weight` [vocab, hidden]: the token lookup, and the tied LM head.
    pub embed: Array,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    rotary: RotaryCache,
}

impl Qwen3 {
    pub fn load(c: &Qwen3Config, t: &Tensors) -> Result<Qwen3> {
        let (d, p, eps) = (c.hidden_size, &c.prefix, c.rms_norm_eps);
        let (h, kv, hd) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for n in 0..c.num_hidden_layers {
            let lp = format!("{p}layers.{n}.");
            let ap = format!("{lp}self_attn.");
            layers.push(Layer {
                input_norm: RmsNorm::load(t, &format!("{lp}input_layernorm.weight"), d, eps)?,
                q: Linear::load(t, &format!("{ap}q_proj."), h * hd, d, c.attention_bias)?,
                k: Linear::load(t, &format!("{ap}k_proj."), kv * hd, d, c.attention_bias)?,
                v: Linear::load(t, &format!("{ap}v_proj."), kv * hd, d, c.attention_bias)?,
                o: Linear::load(t, &format!("{ap}o_proj."), d, h * hd, c.attention_bias)?,
                q_norm: RmsNorm::load(t, &format!("{ap}q_norm.weight"), hd, eps)?,
                k_norm: RmsNorm::load(t, &format!("{ap}k_norm.weight"), hd, eps)?,
                post_norm: RmsNorm::load(
                    t,
                    &format!("{lp}post_attention_layernorm.weight"),
                    d,
                    eps,
                )?,
                mlp: Mlp::load(t, &format!("{lp}mlp."), d, c.intermediate_size)?,
            });
        }
        Ok(Qwen3 {
            heads: h,
            kv_heads: kv,
            head_dim: hd,
            embed: t.get(&format!("{p}embed_tokens.weight"), &[c.vocab_size, d])?,
            layers,
            final_norm: RmsNorm::load(t, &format!("{p}norm.weight"), d, eps)?,
            rotary: RotaryCache::new(c.rope_theta, hd),
        })
    }

    /// The final normed hidden states of `ids` [rows, seq]: [rows, seq, hidden].
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let shape = ids.shape();
        let (rows, seq) = (shape[0], shape[1]);
        let (h, kv, hd) = (
            self.heads as i32,
            self.kv_heads as i32,
            self.head_dim as i32,
        );
        let rep = h / kv;
        let mask = causal_mask(seq)?;
        let (cos, sin) = self.rotary.get(seq)?;
        let scale = (self.head_dim as f32).powf(-0.5);
        // [rows, seq, n, hd] -> [rows, n, seq, hd], normed per head.
        let heads_of = |a: Array, n: i32, norm: &RmsNorm| -> Result<Array> {
            let a = a.reshape(&[rows, seq, n, hd]).map_err(mlx)?;
            norm.forward(&a)?.transpose(&[0, 2, 1, 3]).map_err(mlx)
        };
        let mut x = self.embed.take(ids, 0).map_err(mlx)?;
        for layer in &self.layers {
            let normed = layer.input_norm.forward(&x)?;
            let q = rotate(
                &heads_of(layer.q.forward(&normed)?, h, &layer.q_norm)?,
                &cos,
                &sin,
            )?;
            let k = rotate(
                &heads_of(layer.k.forward(&normed)?, kv, &layer.k_norm)?,
                &cos,
                &sin,
            )?;
            let v = layer
                .v
                .forward(&normed)?
                .reshape(&[rows, seq, kv, hd])
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .map_err(mlx)?;
            let (k, v) = if rep > 1 {
                (
                    k.repeat(rep, 1).map_err(mlx)?,
                    v.repeat(rep, 1).map_err(mlx)?,
                )
            } else {
                (k, v)
            };
            let attn = Array::attention(&q, &k, &v, scale, Some(&mask))
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .and_then(|a| a.reshape(&[rows, seq, h * hd]))
                .map_err(mlx)?;
            x = x.add(&layer.o.forward(&attn)?).map_err(mlx)?;
            x = x
                .add(&layer.mlp.forward(&layer.post_norm.forward(&x)?)?)
                .map_err(mlx)?;
        }
        self.final_norm.forward(&x)
    }
}

/// `h[r, pos[r], :] · embed[ids]ᵀ`: the tied LM head at one position per row, restricted to the
/// given tokens. `h` [rows, seq, hidden], `pos` [rows], `ids` [k]; returns [rows, k].
pub fn tied_rows(h: &Array, embed: &Array, pos: &Array, ids: &Array) -> Result<Array> {
    let s = h.shape();
    let (rows, hidden) = (s[0], s[2]);
    let at = pos
        .reshape(&[rows, 1, 1])
        .and_then(|p| p.broadcast_to(&[rows, 1, hidden]))
        .map_err(mlx)?;
    let hs = h
        .take_along_axis(&at, 1)
        .and_then(|a| a.reshape(&[rows, hidden]))
        .map_err(mlx)?;
    let w = embed
        .take(ids, 0)
        .and_then(|w| w.transpose(&[1, 0]))
        .map_err(mlx)?;
    hs.matmul(&w).map_err(mlx)
}
