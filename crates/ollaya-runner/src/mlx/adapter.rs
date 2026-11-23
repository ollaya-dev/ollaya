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

/// `Aᵀ` [in, r] and `Bᵀ` [r, out] of one adapted linear.
struct Pair {
    a_t: Array,
    b_t: Array,
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
            pairs.insert(target.clone(), Pair { a_t, b_t });
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
        x.matmul(&pair.a_t)
            .and_then(|h| h.matmul(&pair.b_t))
            .and_then(|d| d.multiply(&set.scaling))
            .map(Some)
            .map_err(mlx)
    })
}
