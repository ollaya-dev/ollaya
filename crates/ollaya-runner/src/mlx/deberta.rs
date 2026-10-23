//! DeBERTa-v2/v3 in fp32 on MLX, written against transformers 5.17's `modeling_deberta_v2.py`
//! (`DebertaV2Model`: embeddings, encoder with disentangled attention; no conv layer, no absolute
//! position or token-type embeddings, which DeBERTa-v3 does not use).
//!
//! Parity notes (`docs/decisions/0001-mlx-engine.md`):
//! * Attention is written out (scores, bias, mask, softmax, values), not MLX's fused SDPA: the
//!   relative-position bias is an additive term per head, and the reference masks with
//!   `finfo.min` rather than removing keys, so a fully padded query row comes out uniform, not NaN.
//! * The log buckets of the relative positions are computed on the CPU in f32, as
//!   `make_log_bucket_position` computes them (`ln`, one division, `ceil`); a unit test pins the
//!   table to torch's.
//! * The relative embeddings' LayerNorm and their query and key projections (`share_att_key`) do
//!   not depend on the input, so each layer computes them once, when the model loads.
//! * GELU is the exact (erf) form, as `ACT2FN["gelu"]`.

use std::cell::RefCell;

use ollaya_mlx::{Array, Dtype};

use super::arch::DebertaV2Config;
use super::modernbert::{Linear, Norm, gelu};
use super::weights::{Tensors, mlx};
use crate::Error;

type Result<T> = std::result::Result<T, Error>;

/// What the reference fills masked scores with: `torch.finfo(float32).min`.
const MASKED: f32 = f32::MIN;

/// `make_log_bucket_position(rel, bucket_size, max_position)` for one relative position, in the
/// f32 arithmetic torch uses on its long tensor (true division gives float32).
fn log_bucket(rel: i32, bucket_size: i32, max_position: i32) -> i32 {
    let mid = bucket_size / 2;
    let abs_pos = if rel < mid && rel > -mid {
        mid - 1
    } else {
        rel.abs()
    };
    if abs_pos <= mid {
        return rel;
    }
    let log_pos = ((abs_pos as f32 / mid as f32).ln()
        / ((max_position - 1) as f32 / mid as f32).ln()
        * (mid - 1) as f32)
        .ceil()
        + mid as f32;
    (log_pos * rel.signum() as f32) as i32
}

struct Layer {
    query: Linear,
    key: Linear,
    value: Linear,
    attn_out: Linear,
    attn_norm: Norm,
    intermediate: Linear,
    output: Linear,
    out_norm: Norm,
    /// `key_proj(rel_embeddings)` per head, [heads, 2 * span, head_dim] (content → position).
    pos_key: Option<Array>,
    /// `query_proj(rel_embeddings)` per head, [heads, 2 * span, head_dim] (position → content).
    pos_query: Option<Array>,
}

/// The gather indices of one sequence length: `c2p_pos` and `p2c_pos`, [seq, seq] each.
struct RelPos {
    seq: i32,
    c2p: Array,
    p2c: Array,
}

pub struct DebertaV2 {
    pub hidden: usize,
    heads: usize,
    head_dim: usize,
    /// `position_buckets`, or `max_relative_positions` without buckets: the embedding table holds
    /// `2 * span` relative positions.
    span: i32,
    buckets: i32,
    max_relative: i32,
    word_embeddings: Array,
    emb_norm: Norm,
    layers: Vec<Layer>,
    /// `1 / sqrt(head_dim * scale_factor)`, scale_factor = 1 + c2p + p2c.
    scale: f32,
    rel_pos: RefCell<Option<RelPos>>,
}

impl DebertaV2 {
    pub fn load(c: &DebertaV2Config, t: &Tensors) -> Result<DebertaV2> {
        let (d, i, p) = (c.hidden_size, c.intermediate_size, &c.prefix);
        let eps = c.layer_norm_eps;
        let heads = c.num_attention_heads;
        let c2p = c.pos_att_type.iter().any(|a| a == "c2p");
        let p2c = c.pos_att_type.iter().any(|a| a == "p2c");
        let span = if c.position_buckets > 0 {
            c.position_buckets
        } else {
            c.max_relative_positions
        } as i32;
        // `get_rel_embedding`: the table, normed once (`norm_rel_ebd` = layer_norm).
        let mut rel = t.get(
            &format!("{p}encoder.rel_embeddings.weight"),
            &[2 * span as usize, d],
        )?;
        if c.norm_rel_ebd.split('|').any(|n| n.trim() == "layer_norm") {
            rel = Norm::load(t, &format!("{p}encoder.LayerNorm."), d, true, eps)?.forward(&rel)?;
        }
        let per_head = |a: &Array| {
            a.reshape(&[2 * span, heads as i32, (d / heads) as i32])
                .and_then(|a| a.transpose(&[1, 0, 2]))
                .map_err(mlx)
        };
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for n in 0..c.num_hidden_layers {
            let lp = format!("{p}encoder.layer.{n}.");
            let query = Linear::load(t, &format!("{lp}attention.self.query_proj."), d, d, true)?;
            let key = Linear::load(t, &format!("{lp}attention.self.key_proj."), d, d, true)?;
            // `share_att_key`: the content projections project the positions too.
            let pos_key = if c2p {
                Some(per_head(&key.forward(&rel)?)?)
            } else {
                None
            };
            let pos_query = if p2c {
                Some(per_head(&query.forward(&rel)?)?)
            } else {
                None
            };
            layers.push(Layer {
                query,
                key,
                value: Linear::load(t, &format!("{lp}attention.self.value_proj."), d, d, true)?,
                attn_out: Linear::load(t, &format!("{lp}attention.output.dense."), d, d, true)?,
                attn_norm: Norm::load(
                    t,
                    &format!("{lp}attention.output.LayerNorm."),
                    d,
                    true,
                    eps,
                )?,
                intermediate: Linear::load(t, &format!("{lp}intermediate.dense."), i, d, true)?,
                output: Linear::load(t, &format!("{lp}output.dense."), d, i, true)?,
                out_norm: Norm::load(t, &format!("{lp}output.LayerNorm."), d, true, eps)?,
                pos_key,
                pos_query,
            });
        }
        let scale_factor = 1 + usize::from(c2p) + usize::from(p2c);
        Ok(DebertaV2 {
            hidden: d,
            heads,
            head_dim: d / heads,
            span,
            buckets: c.position_buckets as i32,
            max_relative: c.max_relative_positions as i32,
            word_embeddings: t.get(
                &format!("{p}embeddings.word_embeddings.weight"),
                &[c.vocab_size, d],
            )?,
            emb_norm: Norm::load(t, &format!("{p}embeddings.LayerNorm."), d, true, eps)?,
            layers,
            scale: ((d / heads * scale_factor) as f32).sqrt().recip(),
            rel_pos: RefCell::new(None),
        })
    }

    /// Word embeddings before the embedding norm: [rows, seq, hidden].
    pub fn embed(&self, ids: &Array) -> Result<Array> {
        self.word_embeddings.take(ids, 0).map_err(mlx)
    }

    /// `c2p_pos` and `p2c_pos` for `seq` positions: `clamp(±(i - j) + span, 0, 2 span - 1)` of
    /// the bucketed relative position `i - j` (`build_relative_position`).
    fn rel_pos(&self, seq: i32) -> Result<(Array, Array)> {
        let mut cache = self.rel_pos.borrow_mut();
        if cache.as_ref().is_none_or(|r| r.seq != seq) {
            let n = (seq * seq) as usize;
            let (mut c2p, mut p2c) = (Vec::with_capacity(n), Vec::with_capacity(n));
            for i in 0..seq {
                for j in 0..seq {
                    let rel = if self.buckets > 0 && self.max_relative > 0 {
                        log_bucket(i - j, self.buckets, self.max_relative)
                    } else {
                        i - j
                    };
                    c2p.push((rel + self.span).clamp(0, 2 * self.span - 1));
                    p2c.push((-rel + self.span).clamp(0, 2 * self.span - 1));
                }
            }
            *cache = Some(RelPos {
                seq,
                c2p: Array::from_slice(&c2p, &[seq, seq]).map_err(mlx)?,
                p2c: Array::from_slice(&p2c, &[seq, seq]).map_err(mlx)?,
            });
        }
        let r = cache.as_ref().expect("filled above");
        Ok((r.c2p.clone(), r.p2c.clone()))
    }

    /// The encoder over raw embeddings `x` [rows, seq, hidden] (from [`Self::embed`], plus any
    /// extra embedding the head adds), with `attention` [rows, seq] bool. Returns the last hidden
    /// state, as `DebertaV2Model(...).last_hidden_state`.
    pub fn forward(&self, x: &Array, attention: &Array) -> Result<Array> {
        let shape = x.shape();
        let (rows, seq) = (shape[0], shape[1]);
        let (h, hd, d) = (self.heads as i32, self.head_dim as i32, self.hidden as i32);
        let mask_f = attention.astype(Dtype::Float32).map_err(mlx)?;
        // `DebertaV2Embeddings`: LayerNorm, then zero at padding.
        let mut x = self
            .emb_norm
            .forward(x)?
            .multiply(&mask_f.reshape(&[rows, seq, 1]).map_err(mlx)?)
            .map_err(mlx)?;
        // `get_attention_mask`: mask[i, j] = attention[i] & attention[j], [rows, 1, seq, seq].
        let keep = attention
            .reshape(&[rows, 1, 1, seq])
            .and_then(|k| k.logical_and(&attention.reshape(&[rows, 1, seq, 1])?))
            .map_err(mlx)?;
        let (c2p_pos, p2c_pos) = self.rel_pos(seq)?;
        let gather_index = |pos: &Array| {
            pos.reshape(&[1, 1, seq, seq])
                .and_then(|i| i.broadcast_to(&[rows, h, seq, seq]))
                .map_err(mlx)
        };
        let (c2p_idx, p2c_idx) = (gather_index(&c2p_pos)?, gather_index(&p2c_pos)?);
        let scale = Array::scalar(self.scale);
        let heads_of = |a: Array| {
            a.reshape(&[rows, seq, h, hd])
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .map_err(mlx)
        };
        for layer in &self.layers {
            let q = heads_of(layer.query.forward(&x)?)?;
            let k = heads_of(layer.key.forward(&x)?)?;
            let v = heads_of(layer.value.forward(&x)?)?;
            // `bmm(query, key^T / scale)`.
            let k_t = k
                .transpose(&[0, 1, 3, 2])
                .and_then(|k| k.multiply(&scale))
                .map_err(mlx)?;
            let mut scores = q.matmul(&k_t).map_err(mlx)?;
            // `disentangled_attention_bias`: content→position and position→content, each a gather
            // over the projected relative positions, divided by the same scale.
            if let Some(pos_key) = &layer.pos_key {
                let c2p = q
                    .matmul(&pos_key.transpose(&[0, 2, 1]).map_err(mlx)?)
                    .and_then(|a| a.take_along_axis(&c2p_idx, -1))
                    .and_then(|a| a.multiply(&scale))
                    .map_err(mlx)?;
                scores = scores.add(&c2p).map_err(mlx)?;
            }
            if let Some(pos_query) = &layer.pos_query {
                let p2c = k
                    .matmul(&pos_query.transpose(&[0, 2, 1]).map_err(mlx)?)
                    .and_then(|a| a.take_along_axis(&p2c_idx, -1))
                    .and_then(|a| a.transpose(&[0, 1, 3, 2]))
                    .and_then(|a| a.multiply(&scale))
                    .map_err(mlx)?;
                scores = scores.add(&p2c).map_err(mlx)?;
            }
            let probs = Array::select(&keep, &scores, &Array::scalar(MASKED))
                .and_then(|s| s.softmax(-1))
                .map_err(mlx)?;
            let context = probs
                .matmul(&v)
                .and_then(|a| a.transpose(&[0, 2, 1, 3]))
                .and_then(|a| a.reshape(&[rows, seq, d]))
                .map_err(mlx)?;
            // `DebertaV2SelfOutput`, then `DebertaV2Intermediate` and `DebertaV2Output`.
            let a = layer
                .attn_norm
                .forward(&layer.attn_out.forward(&context)?.add(&x).map_err(mlx)?)?;
            let ff = layer
                .output
                .forward(&gelu(&layer.intermediate.forward(&a)?)?)?;
            x = layer.out_norm.forward(&ff.add(&a).map_err(mlx)?)?;
        }
        Ok(x)
    }

    /// Rows per forward pass so the [rows, heads, seq, seq] scores and the two bias gathers stay
    /// around a quarter of a GiB each: the unified memory is everyone's.
    pub fn rows_per_pass(&self, seq: usize) -> usize {
        ((1 << 26) / (self.heads * seq * seq).max(1)).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::log_bucket;

    /// `make_log_bucket_position(arange(-511, 512), 256, 512)` from torch 2.x: the identity up to
    /// |rel| 128, then the log buckets, 255 at the far end.
    #[test]
    fn buckets_match_torch() {
        let far: [i32; 384] = [
            128, 129, 130, 131, 131, 132, 133, 133, 134, 135, 135, 136, 137, 137, 138, 139, 139,
            140, 141, 141, 142, 142, 143, 144, 144, 145, 145, 146, 147, 147, 148, 148, 149, 150,
            150, 151, 151, 152, 152, 153, 153, 154, 155, 155, 156, 156, 157, 157, 158, 158, 159,
            159, 160, 160, 161, 161, 162, 162, 163, 163, 164, 164, 165, 165, 166, 166, 167, 167,
            168, 168, 169, 169, 169, 170, 170, 171, 171, 172, 172, 173, 173, 173, 174, 174, 175,
            175, 176, 176, 177, 177, 177, 178, 178, 179, 179, 179, 180, 180, 181, 181, 181, 182,
            182, 183, 183, 183, 184, 184, 185, 185, 185, 186, 186, 187, 187, 187, 188, 188, 188,
            189, 189, 190, 190, 190, 191, 191, 191, 192, 192, 192, 193, 193, 194, 194, 194, 195,
            195, 195, 196, 196, 196, 197, 197, 197, 198, 198, 198, 199, 199, 199, 200, 200, 200,
            201, 201, 201, 202, 202, 202, 203, 203, 203, 204, 204, 204, 204, 205, 205, 205, 206,
            206, 206, 207, 207, 207, 208, 208, 208, 208, 209, 209, 209, 210, 210, 210, 211, 211,
            211, 211, 212, 212, 212, 213, 213, 213, 213, 214, 214, 214, 215, 215, 215, 215, 216,
            216, 216, 216, 217, 217, 217, 218, 218, 218, 218, 219, 219, 219, 219, 220, 220, 220,
            221, 221, 221, 221, 222, 222, 222, 222, 223, 223, 223, 223, 224, 224, 224, 224, 225,
            225, 225, 225, 226, 226, 226, 226, 227, 227, 227, 227, 228, 228, 228, 228, 229, 229,
            229, 229, 230, 230, 230, 230, 230, 231, 231, 231, 231, 232, 232, 232, 232, 233, 233,
            233, 233, 233, 234, 234, 234, 234, 235, 235, 235, 235, 236, 236, 236, 236, 236, 237,
            237, 237, 237, 238, 238, 238, 238, 238, 239, 239, 239, 239, 239, 240, 240, 240, 240,
            241, 241, 241, 241, 241, 242, 242, 242, 242, 242, 243, 243, 243, 243, 243, 244, 244,
            244, 244, 244, 245, 245, 245, 245, 245, 246, 246, 246, 246, 246, 247, 247, 247, 247,
            247, 248, 248, 248, 248, 248, 249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 251,
            251, 251, 251, 251, 251, 252, 252, 252, 252, 252, 253, 253, 253, 253, 253, 254, 254,
            254, 254, 254, 254, 255, 255, 255, 255, 255, 255,
        ];
        for rel in -127..=127 {
            assert_eq!(log_bucket(rel, 256, 512), rel);
        }
        for (i, &want) in far.iter().enumerate() {
            let rel = 128 + i as i32;
            assert_eq!(log_bucket(rel, 256, 512), want, "rel {rel}");
            assert_eq!(log_bucket(-rel, 256, 512), -want, "rel -{rel}");
        }
    }
}
