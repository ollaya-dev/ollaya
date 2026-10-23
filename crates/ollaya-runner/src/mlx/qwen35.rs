//! Qwen3.5 (the hybrid decoder) in fp32 on MLX, written against transformers 5.17's
//! `modeling_qwen3_5.py` and the export trunk `convert/ollaya_convert/families/llm_common/qwen35.py`:
//! Gated DeltaNet layers (a depthwise causal conv, the chunked gated delta rule, a gated RMSNorm)
//! and gated full-attention layers (a RMSNorm per head on q and k, RoPE over the first
//! `rotary_dim` channels, a sigmoid output gate), each followed by a SwiGLU MLP. The RMSNorms are
//! zero-centred (`(1 + w)`). Rows are right-padded to a multiple of [`CHUNK`]; every layer is
//! causal, so pads never reach a readout position.
//!
//! Parity notes (`docs/decisions/0001-mlx-engine.md`):
//! * The gated delta rule is the export trunk's chunked form in plain MLX ops, no custom kernel:
//!   per 64-token chunk the UT transform (the unit lower-triangular inverse by recursive block
//!   doubling, exact in real arithmetic), then the inter-chunk recurrence as a host loop over the
//!   chunks (three batched matmuls each). A custom Metal kernel would be compiled at run time and
//!   use fast math on macOS 15 and later, which the engine avoids.
//! * `exp` of the cumulative decays is taken where the mask keeps a pair only: a masked pair's
//!   exponent can overflow, and `inf * 0` is NaN.
//! * Softplus is `max(x, 0) + log1p(exp(-|x|))`, the stable form, within rounding of torch's
//!   thresholded one.

use ollaya_mlx::Array;

use super::arch::Qwen35Config;
use super::modernbert::{Linear, rotate, silu};
use super::qwen3::{Mlp, RmsNorm, RotaryCache, causal_mask};
use super::weights::{Tensors, mlx};
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

/// The gated delta rule's chunk: rows are padded to a multiple of it.
pub const CHUNK: i32 = 64;

/// `x * rsqrt(sum(x²) + eps)` over the last axis (`_l2norm`).
fn l2norm(x: &Array) -> Result<Array> {
    x.square()
        .and_then(|s| s.sum(-1, true))
        .and_then(|s| s.add(&Array::scalar(1e-6)))
        .and_then(|s| s.rsqrt())
        .and_then(|r| x.multiply(&r))
        .map_err(mlx)
}

/// `log(1 + exp(x))`, as `max(x, 0) + log1p(exp(-|x|))`.
fn softplus(x: &Array) -> Result<Array> {
    let tail = x
        .abs()
        .and_then(|a| a.negative())
        .and_then(|a| a.exp())
        .and_then(|a| a.log1p())
        .map_err(mlx)?;
    x.maximum(&Array::scalar(0.0))
        .and_then(|m| m.add(&tail))
        .map_err(mlx)
}

/// The masks of the chunk: `lower` keeps j <= i, `strict_lower` j < i (both [CHUNK, CHUNK] bool),
/// and `doubling[l]` selects, inside every diagonal block of size 2s (s = 2^l), its lower-left
/// s × s block (`_doubling_masks`), as float32 [levels, CHUNK, CHUNK].
struct ChunkMasks {
    lower: Array,
    strict_lower: Array,
    doubling: Vec<Array>,
}

impl ChunkMasks {
    fn new() -> Result<ChunkMasks> {
        let c = CHUNK as usize;
        let mut lower = Vec::with_capacity(c * c);
        let mut strict = Vec::with_capacity(c * c);
        for i in 0..c {
            for j in 0..c {
                lower.push(j <= i);
                strict.push(j < i);
            }
        }
        let mut doubling = Vec::new();
        let mut s = 1;
        while s < c {
            let m: Vec<f32> = (0..c * c)
                .map(|n| {
                    let (i, j) = (n / c, n % c);
                    let same = i / (2 * s) == j / (2 * s);
                    let lower_left = i % (2 * s) >= s && j % (2 * s) < s;
                    if same && lower_left { 1.0 } else { 0.0 }
                })
                .collect();
            doubling.push(Array::from_slice(&m, &[CHUNK, CHUNK]).map_err(mlx)?);
            s *= 2;
        }
        Ok(ChunkMasks {
            lower: Array::from_slice(&lower, &[CHUNK, CHUNK]).map_err(mlx)?,
            strict_lower: Array::from_slice(&strict, &[CHUNK, CHUNK]).map_err(mlx)?,
            doubling,
        })
    }

    /// `(I + A)^-1` for strictly lower-triangular `A` [.., CHUNK, CHUNK] (`unit_lower_inverse`).
    fn unit_lower_inverse(&self, a: &Array) -> Result<Array> {
        let shape = a.shape();
        let mut x = Array::eye(CHUNK)
            .and_then(|e| e.broadcast_to(&shape))
            .map_err(mlx)?;
        for m in &self.doubling {
            let am = a.multiply(m).map_err(mlx)?;
            x = x
                .matmul(&am)
                .and_then(|t| t.matmul(&x))
                .and_then(|t| x.subtract(&t))
                .map_err(mlx)?;
        }
        Ok(x)
    }
}

/// `Qwen3_5GatedDeltaNet`.
struct DeltaNet {
    qkv: Linear,
    /// The depthwise conv's taps, [conv_dim, kernel, 1] as `mlx.core.conv1d` takes them.
    conv: Array,
    kernel: i32,
    z: Linear,
    b: Linear,
    a: Linear,
    /// `-exp(A_log)`, [v_heads].
    neg_a: Array,
    dt_bias: Array,
    /// The gated norm's weight, [dv].
    norm_w: Array,
    out: Linear,
    k_heads: i32,
    v_heads: i32,
    dk: i32,
    dv: i32,
    eps: f32,
}

impl DeltaNet {
    fn load(t: &Tensors, p: &str, c: &Qwen35Config) -> Result<DeltaNet> {
        let d = c.hidden_size;
        let (hk, hv, dk, dv) = (
            c.linear_num_key_heads,
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let (key_dim, value_dim) = (hk * dk, hv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        let kernel = c.linear_conv_kernel_dim;
        Ok(DeltaNet {
            qkv: Linear::load(t, &format!("{p}in_proj_qkv."), conv_dim, d, false)?,
            // torch: [channels, 1, kernel]; MLX: [channels, kernel, 1].
            conv: t
                .get(&format!("{p}conv1d.weight"), &[conv_dim, 1, kernel])?
                .transpose(&[0, 2, 1])
                .map_err(mlx)?,
            kernel: kernel as i32,
            z: Linear::load(t, &format!("{p}in_proj_z."), value_dim, d, false)?,
            b: Linear::load(t, &format!("{p}in_proj_b."), hv, d, false)?,
            a: Linear::load(t, &format!("{p}in_proj_a."), hv, d, false)?,
            neg_a: t
                .get(&format!("{p}A_log"), &[hv])?
                .exp()
                .and_then(|a| a.negative())
                .map_err(mlx)?,
            dt_bias: t.get(&format!("{p}dt_bias"), &[hv])?,
            norm_w: t.get(&format!("{p}norm.weight"), &[dv])?,
            out: Linear::load(t, &format!("{p}out_proj."), d, value_dim, false)?,
            k_heads: hk as i32,
            v_heads: hv as i32,
            dk: dk as i32,
            dv: dv as i32,
            eps: c.rms_norm_eps,
        })
    }

    /// `h` [rows, seq, hidden] with `seq` a multiple of [`CHUNK`]: [rows, seq, hidden].
    fn forward(&self, h: &Array, masks: &ChunkMasks) -> Result<Array> {
        let shape = h.shape();
        let (rows, seq) = (shape[0], shape[1]);
        let (hk, hv, dk, dv) = (self.k_heads, self.v_heads, self.dk, self.dv);
        let conv_dim = 2 * hk * dk + hv * dv;
        // The depthwise causal conv: pad both sides by kernel - 1, keep the first seq outputs.
        let mixed = self
            .qkv
            .forward(h)?
            .conv1d(&self.conv, self.kernel - 1, conv_dim)
            .and_then(|m| m.slice(&[0, 0, 0], &[rows, seq, conv_dim]))
            .map_err(mlx)?;
        let mixed = silu(&mixed)?;
        let part = |from: i32, to: i32, heads: i32, dim: i32| {
            mixed
                .slice(&[0, 0, from], &[rows, seq, to])
                .and_then(|a| a.reshape(&[rows, seq, heads, dim]))
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .map_err(mlx)
        };
        let q = part(0, hk * dk, hk, dk)?;
        let k = part(hk * dk, 2 * hk * dk, hk, dk)?;
        let v = part(2 * hk * dk, conv_dim, hv, dv)?;
        let rep = hv / hk;
        let (q, k) = if rep > 1 {
            (
                q.repeat(rep, 1).map_err(mlx)?,
                k.repeat(rep, 1).map_err(mlx)?,
            )
        } else {
            (q, k)
        };
        // beta and g: [rows, v_heads, seq].
        let beta = self
            .b
            .forward(h)?
            .sigmoid()
            .and_then(|b| b.transpose(&[0, 2, 1]))
            .map_err(mlx)?;
        let g = softplus(&self.a.forward(h)?.add(&self.dt_bias).map_err(mlx)?)?
            .multiply(&self.neg_a)
            .and_then(|g| g.transpose(&[0, 2, 1]))
            .map_err(mlx)?;
        let core = gated_delta_rule(&q, &k, &v, &g, &beta, masks)?; // [rows, v_heads, seq, dv]
        let core = core
            .transpose(&[0, 2, 1, 3])
            .and_then(|a| a.reshape(&[rows, seq, hv, dv]))
            .map_err(mlx)?;
        // `Qwen3_5RMSNormGated`: norm, weight, then times silu(z).
        let z = self
            .z
            .forward(h)?
            .reshape(&[rows, seq, hv, dv])
            .map_err(mlx)?;
        let gate = silu(&z)?;
        let normed = core
            .rms_norm(&self.norm_w, self.eps)
            .and_then(|n| n.multiply(&gate))
            .and_then(|n| n.reshape(&[rows, seq, hv * dv]))
            .map_err(mlx)?;
        self.out.forward(&normed)
    }
}

/// `gated_delta_rule` of the export trunk: q, k [B, H, T, dk], v [B, H, T, dv], g and beta
/// [B, H, T], T a multiple of [`CHUNK`]. Returns [B, H, T, dv].
fn gated_delta_rule(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    masks: &ChunkMasks,
) -> Result<Array> {
    let shape = k.shape();
    let (b, h, t, dk) = (shape[0], shape[1], shape[2], shape[3]);
    let dv = v.shape()[3];
    let n = t / CHUNK;
    let q = l2norm(q)?
        .multiply(&Array::scalar((dk as f32).powf(-0.5)))
        .map_err(mlx)?;
    let k = l2norm(k)?;
    let beta = beta.expand_dims(-1).map_err(mlx)?;
    let v_beta = v.multiply(&beta).map_err(mlx)?;
    let k_beta = k.multiply(&beta).map_err(mlx)?;
    let chunks = |a: &Array, d: i32| a.reshape(&[b, h, n, CHUNK, d]).map_err(mlx);
    let (q, k, k_beta, v_beta) = (
        chunks(&q, dk)?,
        chunks(&k, dk)?,
        chunks(&k_beta, dk)?,
        chunks(&v_beta, dv)?,
    );
    let g = g.reshape(&[b, h, n, CHUNK]).map_err(mlx)?;
    let cum = g.cumsum(-1).map_err(mlx)?;
    // pair[i, j] = exp(cum_i - cum_j) for j <= i, else 0.
    let neg_inf = Array::scalar(f32::NEG_INFINITY);
    let pair = cum
        .expand_dims(-1)
        .and_then(|ci| ci.subtract(&cum.expand_dims(-2)?))
        .and_then(|d| Array::select(&masks.lower, &d, &neg_inf))
        .and_then(|d| d.exp())
        .map_err(mlx)?;
    let k_t = k.transpose(&[0, 1, 2, 4, 3]).map_err(mlx)?;
    let ut = k_beta
        .matmul(&k_t)
        .and_then(|a| a.multiply(&pair))
        .map_err(mlx)?;
    let intra = q
        .matmul(&k_t)
        .and_then(|a| a.multiply(&pair))
        .map_err(mlx)?;
    let exp_cum = cum.exp().and_then(|e| e.expand_dims(-1)).map_err(mlx)?;
    let decayed_k_beta = k_beta.multiply(&exp_cum).map_err(mlx)?;
    let zero = Array::scalar(0.0);
    let strict = Array::select(&masks.strict_lower, &ut, &zero).map_err(mlx)?;
    let inv = masks.unit_lower_inverse(&strict)?;
    let new_v = inv.matmul(&v_beta).map_err(mlx)?;
    let k_cumdecay = inv.matmul(&decayed_k_beta).map_err(mlx)?;
    let q = q.multiply(&exp_cum).map_err(mlx)?;
    // cum[..., -1:]: the chunk's total decay.
    let last = cum
        .slice(&[0, 0, 0, CHUNK - 1], &[b, h, n, CHUNK])
        .map_err(mlx)?;
    let k = last
        .subtract(&cum)
        .and_then(|d| d.exp())
        .and_then(|e| e.expand_dims(-1))
        .and_then(|e| k.multiply(&e))
        .map_err(mlx)?;
    let chunk_decay = last
        .exp()
        .and_then(|e| e.reshape(&[b, h, n, 1, 1]))
        .map_err(mlx)?;

    // The recurrence over chunks, [B, H, dk, dv] state.
    let mut state = Array::full(&[b, h, dk, dv], 0.0).map_err(mlx)?;
    let mut outs = Vec::with_capacity(n as usize);
    let at = |a: &Array, i: i32, d: i32| {
        a.slice(&[0, 0, i, 0, 0], &[b, h, i + 1, CHUNK, d])
            .and_then(|c| c.reshape(&[b, h, CHUNK, d]))
            .map_err(mlx)
    };
    for i in 0..n {
        let (new_v_i, k_cumdecay_i, q_i, intra_i, k_i) = (
            at(&new_v, i, dv)?,
            at(&k_cumdecay, i, dk)?,
            at(&q, i, dk)?,
            at(&intra, i, CHUNK)?,
            at(&k, i, dk)?,
        );
        let v_new = k_cumdecay_i
            .matmul(&state)
            .and_then(|m| new_v_i.subtract(&m))
            .map_err(mlx)?;
        let out = q_i
            .matmul(&state)
            .and_then(|o| o.add(&intra_i.matmul(&v_new)?))
            .map_err(mlx)?;
        let decay = chunk_decay
            .slice(&[0, 0, i, 0, 0], &[b, h, i + 1, 1, 1])
            .and_then(|d| d.reshape(&[b, h, 1, 1]))
            .map_err(mlx)?;
        state = state
            .multiply(&decay)
            .and_then(|s| s.add(&k_i.transpose(&[0, 1, 3, 2])?.matmul(&v_new)?))
            .map_err(mlx)?;
        // Materialise the state so the lazy graph stays one chunk deep.
        state.eval().map_err(mlx)?;
        outs.push(out);
    }
    let refs: Vec<&Array> = outs.iter().collect();
    Array::concatenate(&refs, 2).map_err(mlx)
}

/// `Qwen3_5Attention`: gated full attention with partial RoPE.
struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Attention {
    fn load(t: &Tensors, p: &str, c: &Qwen35Config) -> Result<Attention> {
        let (d, h, kv, hd, eps) = (
            c.hidden_size,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.rms_norm_eps,
        );
        Ok(Attention {
            q: Linear::load(t, &format!("{p}q_proj."), h * hd * 2, d, c.attention_bias)?,
            k: Linear::load(t, &format!("{p}k_proj."), kv * hd, d, c.attention_bias)?,
            v: Linear::load(t, &format!("{p}v_proj."), kv * hd, d, c.attention_bias)?,
            o: Linear::load(t, &format!("{p}o_proj."), d, h * hd, c.attention_bias)?,
            q_norm: RmsNorm::load_zero_centered(t, &format!("{p}q_norm.weight"), hd, eps)?,
            k_norm: RmsNorm::load_zero_centered(t, &format!("{p}k_norm.weight"), hd, eps)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &Array,
        heads: i32,
        kv_heads: i32,
        hd: i32,
        cos: &Array,
        sin: &Array,
        mask: &Array,
    ) -> Result<Array> {
        let shape = h.shape();
        let (rows, seq) = (shape[0], shape[1]);
        let rd = cos.shape()[1];
        // q_proj gives, per head, the query then its gate.
        let qg = self
            .q
            .forward(h)?
            .reshape(&[rows, seq, heads, 2 * hd])
            .map_err(mlx)?;
        let q = qg
            .slice(&[0, 0, 0, 0], &[rows, seq, heads, hd])
            .map_err(mlx)?;
        let gate = qg
            .slice(&[0, 0, 0, hd], &[rows, seq, heads, 2 * hd])
            .and_then(|g| g.reshape(&[rows, seq, heads * hd]))
            .map_err(mlx)?;
        let heads_of = |a: Array, n: i32, norm: &RmsNorm| -> Result<Array> {
            norm.forward(&a.reshape(&[rows, seq, n, hd]).map_err(mlx)?)?
                .transpose(&[0, 2, 1, 3])
                .map_err(mlx)
        };
        // RoPE over the first rd channels; the rest pass through.
        let partial = |a: Array, n: i32| -> Result<Array> {
            let rot = rotate(
                &a.slice(&[0, 0, 0, 0], &[rows, n, seq, rd]).map_err(mlx)?,
                cos,
                sin,
            )?;
            let rest = a.slice(&[0, 0, 0, rd], &[rows, n, seq, hd]).map_err(mlx)?;
            Array::concatenate(&[&rot, &rest], -1).map_err(mlx)
        };
        let q = partial(heads_of(q, heads, &self.q_norm)?, heads)?;
        let k = partial(
            heads_of(self.k.forward(h)?, kv_heads, &self.k_norm)?,
            kv_heads,
        )?;
        let v = self
            .v
            .forward(h)?
            .reshape(&[rows, seq, kv_heads, hd])
            .and_then(|a| a.transpose(&[0, 2, 1, 3]))
            .map_err(mlx)?;
        let rep = heads / kv_heads;
        let (k, v) = if rep > 1 {
            (
                k.repeat(rep, 1).map_err(mlx)?,
                v.repeat(rep, 1).map_err(mlx)?,
            )
        } else {
            (k, v)
        };
        let attn = Array::attention(&q, &k, &v, (hd as f32).powf(-0.5), Some(mask))
            .and_then(|a| a.transpose(&[0, 2, 1, 3]))
            .and_then(|a| a.reshape(&[rows, seq, heads * hd]))
            .and_then(|a| a.multiply(&gate.sigmoid()?))
            .map_err(mlx)?;
        self.o.forward(&attn)
    }
}

enum Mixer {
    Linear(DeltaNet),
    Full(Attention),
}

struct Layer {
    input_norm: RmsNorm,
    mixer: Mixer,
    post_norm: RmsNorm,
    mlp: Mlp,
}

pub struct Qwen35 {
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    /// `embed_tokens.weight` [vocab, hidden]: the token lookup, and the tied LM head.
    pub embed: Array,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    rotary: RotaryCache,
    masks: ChunkMasks,
}

impl Qwen35 {
    pub fn load(c: &Qwen35Config, t: &Tensors) -> Result<Qwen35> {
        let (d, p, eps) = (c.hidden_size, &c.prefix, c.rms_norm_eps);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for (n, kind) in c.layer_types.iter().enumerate() {
            let lp = format!("{p}layers.{n}.");
            let mixer = match kind.as_str() {
                "linear_attention" => {
                    Mixer::Linear(DeltaNet::load(t, &format!("{lp}linear_attn."), c)?)
                }
                "full_attention" => Mixer::Full(Attention::load(t, &format!("{lp}self_attn."), c)?),
                other => return Err(Error::Model(format!("unknown layer type {other:?}"))),
            };
            layers.push(Layer {
                input_norm: RmsNorm::load_zero_centered(
                    t,
                    &format!("{lp}input_layernorm.weight"),
                    d,
                    eps,
                )?,
                mixer,
                post_norm: RmsNorm::load_zero_centered(
                    t,
                    &format!("{lp}post_attention_layernorm.weight"),
                    d,
                    eps,
                )?,
                mlp: Mlp::load(t, &format!("{lp}mlp."), d, c.intermediate_size)?,
            });
        }
        let rotary_dim = (c.head_dim as f32 * c.partial_rotary_factor) as usize;
        Ok(Qwen35 {
            heads: c.num_attention_heads as i32,
            kv_heads: c.num_key_value_heads as i32,
            head_dim: c.head_dim as i32,
            embed: t.get(&format!("{p}embed_tokens.weight"), &[c.vocab_size, d])?,
            layers,
            final_norm: RmsNorm::load_zero_centered(t, &format!("{p}norm.weight"), d, eps)?,
            rotary: RotaryCache::new(c.rope_theta, rotary_dim),
            masks: ChunkMasks::new()?,
        })
    }

    /// The final normed hidden states of `ids` [rows, seq], `seq` a multiple of [`CHUNK`]:
    /// [rows, seq, hidden].
    pub fn forward(&self, ids: &Array) -> Result<Array> {
        let shape = ids.shape();
        let seq = shape[1];
        if seq % CHUNK != 0 {
            return Err(Error::Model(format!(
                "Qwen3.5 rows must be a multiple of {CHUNK} tokens, got {seq}"
            )));
        }
        let mask = causal_mask(seq)?;
        let (cos, sin) = self.rotary.get(seq)?;
        let mut x = self.embed.take(ids, 0).map_err(mlx)?;
        for layer in &self.layers {
            let normed = layer.input_norm.forward(&x)?;
            let mixed = match &layer.mixer {
                Mixer::Linear(m) => m.forward(&normed, &self.masks)?,
                Mixer::Full(a) => a.forward(
                    &normed,
                    self.heads,
                    self.kv_heads,
                    self.head_dim,
                    &cos,
                    &sin,
                    &mask,
                )?,
            };
            x = x.add(&mixed).map_err(mlx)?;
            x = x
                .add(&layer.mlp.forward(&layer.post_norm.forward(&x)?)?)
                .map_err(mlx)?;
        }
        self.final_norm.forward(&x)
    }
}
