//! LoRA adapters served unmerged (decision 0011): `y = x Wᵀ + scaling · (x Aᵀ) Bᵀ` in float32.
//!
//! An adapter's tensors are loaded once per adapter and held next to the shared base. A forward
//! pass runs for one adapter at a time (the runner's one MLX thread serves requests in order): it
//! makes the adapter *active* for the pass, and every [`Linear`](super::modernbert::Linear) the
//! adapter names adds its delta. With no adapter active, nothing changes: the base's matmuls are
//! the ones it always ran.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::rc::Rc;

use ollaya_mlx::Array;
use serde::Deserialize;

use super::weights::{Tensors, mlx};
use crate::Error;

/// The adapter layer (`ollaya_registry::adapter::AdapterSpec`), as far as the engine reads it.
#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    pub format: String,
    pub rank: usize,
    pub scaling: f64,
    pub prefix: String,
    /// Each adapted module by the name of its `.weight` tensor in the base, without the suffix.
    pub targets: Vec<String>,
}

impl Spec {
    pub fn read(path: &Path) -> Result<Spec, Error> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        let spec: Spec = serde_json::from_str(&text)
            .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
        if spec.format != "peft-lora" {
            return Err(Error::Model(format!(
                "{}: adapter format {:?}; this version of ollaya reads peft-lora; upgrade ollaya",
                path.display(),
                spec.format
            )));
        }
        Ok(spec)
    }
}

/// `Aᵀ` [in, r] and `Bᵀ` [r, out] of one adapted linear, plus `Aᵀ` chunked along `in` for
/// compensated `x Aᵀ` summation when `in > KAHAN_CHUNK`.
struct Pair {
    a_t: Array,
    b_t: Array,
    /// `(n, pad, a_batched [n, KAHAN_CHUNK, r])` when `in > KAHAN_CHUNK`.
    a_batched: Option<(i32, i32, Array)>,
}

/// One adapter's tensors, on the MLX thread.
pub struct AdapterTensors {
    scaling: Array,
    pairs: HashMap<String, Pair>,
    /// Bytes its float32 tensors occupy.
    pub bytes: u64,
}

impl AdapterTensors {
    /// Read the tensors `spec` names from the adapter's safetensors file. `backbone_prefix` is the
    /// arch layer's: peft names a module without it.
    pub fn load(
        weights: &Path,
        spec: &Spec,
        backbone_prefix: &str,
    ) -> Result<AdapterTensors, Error> {
        let t = Tensors::open_safetensors(weights)?;
        let mut pairs = HashMap::with_capacity(spec.targets.len());
        for target in &spec.targets {
            let module = target.strip_prefix(backbone_prefix).ok_or_else(|| {
                Error::Model(format!(
                    "adapter target {target} is not under the backbone prefix {backbone_prefix:?}"
                ))
            })?;
            let (a, b) = (
                format!("{}{module}.lora_A.weight", spec.prefix),
                format!("{}{module}.lora_B.weight", spec.prefix),
            );
            let matrix = |n: &str| -> Result<(usize, usize), Error> {
                match t.shape(n) {
                    Some(&[rows, cols]) => Ok((rows, cols)),
                    Some(other) => Err(Error::Model(format!(
                        "{n} has shape {other:?}, not a matrix"
                    ))),
                    None => Err(Error::Model(format!("the adapter has no tensor {n}"))),
                }
            };
            let ((r, inp), (out, rb)) = (matrix(&a)?, matrix(&b)?);
            if r != spec.rank || rb != spec.rank {
                return Err(Error::Model(format!(
                    "{target}: lora_A has rank {r} and lora_B {rb}, the spec says {}",
                    spec.rank
                )));
            }
            let a_t = t.raw(&a, &[r, inp])?.transpose(&[1, 0]).map_err(mlx)?;
            let b_t = t.raw(&b, &[out, r])?.transpose(&[1, 0]).map_err(mlx)?;
            a_t.eval().map_err(mlx)?;
            b_t.eval().map_err(mlx)?;
            let a_batched = match batched_plan(inp as i32, KAHAN_CHUNK) {
                Some((n, pad)) => {
                    let r_dim = r as i32;
                    let a_pad = if pad > 0 {
                        let zeros = Array::full(&[pad, r_dim], 0.0).map_err(mlx)?;
                        Array::concatenate(&[&a_t, &zeros], 0).map_err(mlx)?
                    } else {
                        a_t.clone()
                    };
                    let ab = a_pad.reshape(&[n, KAHAN_CHUNK, r_dim]).map_err(mlx)?;
                    ab.eval().map_err(mlx)?;
                    Some((n, pad, ab))
                }
                None => None,
            };
            pairs.insert(
                target.clone(),
                Pair {
                    a_t,
                    b_t,
                    a_batched,
                },
            );
        }
        Ok(AdapterTensors {
            scaling: Array::scalar(spec.scaling as f32),
            pairs,
            bytes: t.bytes_read(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

thread_local! {
    /// The adapter the forward pass on this (MLX) thread runs, if any.
    static ACTIVE: RefCell<Option<Rc<AdapterTensors>>> = const { RefCell::new(None) };
}

/// Makes an adapter active until dropped.
pub struct Active(Option<Rc<AdapterTensors>>);

impl Active {
    pub fn set(adapter: Option<Rc<AdapterTensors>>) -> Active {
        Active(ACTIVE.with(|a| a.replace(adapter)))
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        ACTIVE.with(|a| *a.borrow_mut() = self.0.take());
    }
}

/// Reduction-axis chunk width for `x Aᵀ`: computing `KAHAN_CHUNK`-wide partial matmuls in one
/// batched matmul and reducing them with Kahan summation keeps the unmerged branch within 1e-3
/// of both the merged model and the fp64 reference on `laya:en`.
const KAHAN_CHUNK: i32 = 128;

/// Chunk count `n` and zero-padding width `pad` for splitting `k_dim` into `chunk`-wide slices,
/// or `None` when a single matmul fits inside `chunk`.
fn batched_plan(k_dim: i32, chunk: i32) -> Option<(i32, i32)> {
    if k_dim <= chunk {
        return None;
    }
    let n = (k_dim as usize).div_ceil(chunk as usize) as i32;
    Some((n, n * chunk - k_dim))
}

/// `x Aᵀ` via one batched matmul over `KAHAN_CHUNK`-wide `k` slices, reduced with Kahan summation.
fn matmul_kahan(x: &Array, pair: &Pair) -> Result<Array, ollaya_mlx::Error> {
    let Some((n, pad, a_batched)) = &pair.a_batched else {
        return x.matmul(&pair.a_t);
    };
    let (n, pad) = (*n, *pad);
    let xs = x.shape();
    let nd = xs.len();
    let k_dim = xs[nd - 1];
    let m: i32 = xs[..nd - 1].iter().product();
    let r_dim = pair.b_t.shape()[0];
    let x2d = x.reshape(&[m, k_dim])?;
    let x_pad = if pad > 0 {
        Array::concatenate(&[&x2d, &Array::full(&[m, pad], 0.0)?], 1)?
    } else {
        x2d
    };
    let x_batched = x_pad.reshape(&[m, n, KAHAN_CHUNK])?.transpose(&[1, 0, 2])?;
    let partials = x_batched.matmul(a_batched)?;
    let chunks = partials.split(n, 0)?;
    let mut sum: Option<Array> = None;
    let mut comp: Option<Array> = None;
    for p in &chunks {
        match (&sum, &comp) {
            (None, _) => {
                sum = Some(p.clone());
            }
            (Some(s), None) => {
                let t = s.add(p)?;
                let c = t.subtract(s)?.subtract(p)?;
                sum = Some(t);
                comp = Some(c);
            }
            (Some(s), Some(c)) => {
                let y = p.subtract(c)?;
                let t = s.add(&y)?;
                let next_c = t.subtract(s)?.subtract(&y)?;
                sum = Some(t);
                comp = Some(next_c);
            }
        }
    }
    let reduced = sum.expect("batched_plan ensures n >= 2").squeeze(0)?;
    let mut out_shape = xs[..nd - 1].to_vec();
    out_shape.push(r_dim);
    reduced.reshape(&out_shape)
}

/// `scaling · (x Aᵀ) Bᵀ` for the linear named `name`, when the active adapter adapts it.
pub fn delta(name: &str, x: &Array) -> Result<Option<Array>, Error> {
    ACTIVE.with(|a| {
        let a = a.borrow();
        let Some(set) = a.as_ref() else {
            return Ok(None);
        };
        let Some(pair) = set.pairs.get(name) else {
            return Ok(None);
        };
        matmul_kahan(x, pair)
            .and_then(|h| h.matmul(&pair.b_t))
            .and_then(|d| d.multiply(&set.scaling))
            .map(Some)
            .map_err(mlx)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_plan_handles_multiples_remainders_and_small_dims() {
        assert_eq!(batched_plan(1024, KAHAN_CHUNK), Some((8, 0)));
        // ModernBERT mlp.Wo has K = 2624, requiring 64 zeros of padding across 21 chunks.
        assert_eq!(batched_plan(2624, KAHAN_CHUNK), Some((21, 64)));
        for k in [129, 256, 1024, 2624, 3072] {
            let (n, pad) = batched_plan(k, KAHAN_CHUNK).unwrap();
            assert_eq!(n * KAHAN_CHUNK, k + pad);
            assert!((0..KAHAN_CHUNK).contains(&pad));
        }
        assert_eq!(batched_plan(128, KAHAN_CHUNK), None);
        assert_eq!(batched_plan(64, KAHAN_CHUNK), None);
        assert_eq!(batched_plan(0, KAHAN_CHUNK), None);
    }
}
