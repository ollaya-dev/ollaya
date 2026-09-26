//! The MLX engine: encoder networks on the Metal GPU, from the author's weights file.
//!
//! A family engine (`OnnxModel`, `NliModel`, `VonModel`, `GliclassModel`) keeps its tokenizer,
//! layout and readout, and swaps its ONNX Runtime session for an [`MlxNet`] on
//! [`Device::Metal`](crate::Device::Metal). The network is our own fp32 code per backbone
//! ([`modernbert`]) and per head ([`heads`]), built from the model's `arch` layer; every op runs on
//! the runner's one MLX thread ([`ollaya_mlx::Worker`]).
//!
//! Design and parity evidence: `docs/decisions/0001-mlx-engine.md`.

pub mod arch;
mod heads;
mod modernbert;
mod weights;

use std::path::{Path, PathBuf};

use ndarray::Array2;
use ollaya_mlx::{Array, MetalInfo, Worker};

use self::arch::{Arch, Backbone, Head};
use self::weights::{Tensors, mlx};
use crate::Error;
use crate::net::{Batch, Head as HeadKind};
use crate::onnx::ModelFiles;

/// Layouts `auto` runs on MLX: those whose models pass their parity gate on Metal. A model also
/// needs an `arch` layer with a supported backbone; without one it runs on ONNX Runtime.
///
/// von (`von-option-marker-v1`) is implemented (`heads::OptionMarker`) but stays on ONNX
/// Runtime: one golden row (`td/agent_trace_observability_000000` `urgency`) amplifies fp32
/// rounding through the encoder, and on Metal its logits land 2e-3 from the float64 goldens,
/// over von's 1e-3 gate (ONNX Runtime on this Mac's CPU: 1.1e-3). See the decision record.
pub const LAYOUTS: &[&str] = &["laya-markers-v1", "nli-pairs-v1", "gliclass-uni-v1"];

/// Where `mlx.metallib` is, first match wins:
/// 1. `$OLLAYA_LIBRARY_PATH/mlx_metal/`;
/// 2. `<exe dir>/../lib/ollaya/mlx_metal/` (the tarball layout);
/// 3. `<exe dir>/../Resources/mlx_metal/` (the macOS app bundle);
/// 4. `<exe dir>/` (development builds: the build script copies it next to the binaries);
/// 5. where this build installed it (development: examples and tests).
pub fn metallib() -> Result<PathBuf, String> {
    let exe = std::env::current_exe()
        .and_then(|e| e.canonicalize())
        .map_err(|e| format!("cannot find the executable: {e}"))?;
    let dir = exe.parent().unwrap_or(Path::new("/"));
    let candidates = [
        std::env::var_os("OLLAYA_LIBRARY_PATH")
            .map(|p| PathBuf::from(p).join("mlx_metal/mlx.metallib")),
        Some(dir.join("../lib/ollaya/mlx_metal/mlx.metallib")),
        Some(dir.join("../Resources/mlx_metal/mlx.metallib")),
        Some(dir.join("mlx.metallib")),
        Some(PathBuf::from(ollaya_mlx::BUILD_METALLIB)),
    ];
    candidates
        .iter()
        .flatten()
        .find(|p| p.is_file())
        .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
        .ok_or_else(|| {
            format!(
                "no mlx.metallib (the MLX pack, lib/ollaya/mlx_metal, is not installed next to {})",
                exe.display()
            )
        })
}

/// Why a model cannot run on MLX here, checked without touching the GPU: `Ok` means try.
pub fn usable(arch: Option<&Path>, layout: &str) -> Result<PathBuf, String> {
    if !LAYOUTS.contains(&layout) {
        return Err(format!("the MLX engine does not run layout {layout}"));
    }
    if arch.is_none() {
        return Err("the model has no arch layer".into());
    }
    metallib()
}

struct Model {
    backbone: modernbert::ModernBert,
    head: HeadModel,
}

enum HeadModel {
    Laya(heads::Laya),
    SequenceClassification(heads::SequenceClassification),
    OptionMarker(heads::OptionMarker),
    Gliclass(heads::Gliclass),
}

/// An encoder network on the MLX thread.
pub struct MlxNet {
    worker: Worker<Model>,
}

impl MlxNet {
    /// Build the network the arch layer describes from the weights file, and check that its head
    /// is the one the calling family expects.
    pub fn load(files: &ModelFiles, expect: HeadKind) -> Result<MlxNet, Error> {
        let arch_path = files
            .arch
            .as_ref()
            .ok_or_else(|| Error::Model("the model has no arch layer; MLX cannot run it".into()))?;
        let arch = Arch::read(arch_path)?;
        if arch.head.name() != expect.name() {
            return Err(Error::Model(format!(
                "{}: head {} does not match this layout's {}",
                arch_path.display(),
                arch.head.name(),
                expect.name()
            )));
        }
        let weights = match &files.weights {
            Some(w) => w.clone(),
            None => arch_path
                .parent()
                .unwrap_or(Path::new("."))
                .join(&arch.weights.file),
        };
        let metallib = metallib().map_err(Error::Model)?;
        let worker = Worker::start("ollaya-mlx", &metallib, move || {
            Model::load(&arch, &weights).map_err(|e| ollaya_mlx::Error(e.to_string()))
        })
        .map_err(mlx)?;
        Ok(MlxNet { worker })
    }

    pub fn info(&self) -> &MetalInfo {
        self.worker.info()
    }

    /// Run one padded batch; `outputs` names what an ONNX graph of this layout would return.
    pub fn run(&self, batch: &Batch, outputs: &[&str]) -> Result<Vec<Array2<f32>>, Error> {
        let (rows, seq) = batch.input_ids.dim();
        let ids: Vec<i32> = batch.input_ids.iter().map(|&v| v as i32).collect();
        let attention: Vec<bool> = batch.attention.iter().map(|&v| v != 0).collect();
        let markers = batch.markers.as_ref().map(|(pos, mask)| {
            (
                pos.dim().1,
                pos.iter().map(|&v| v.max(0) as i32).collect::<Vec<i32>>(),
                mask.iter().copied().collect::<Vec<bool>>(),
            )
        });
        let qtype: Option<Vec<i32>> = batch
            .qtype
            .as_ref()
            .map(|q| q.iter().map(|&v| v as i32).collect());
        let (main, act) = self
            .worker
            .run(move |m| {
                m.forward(rows, seq, ids, attention, markers, qtype)
                    .map_err(|e| ollaya_mlx::Error(e.to_string()))
            })
            .map_err(mlx)?;
        let mut main = Some(main);
        let mut act = act;
        outputs
            .iter()
            .map(|name| {
                let (width, data) = match *name {
                    "act_logits" => act.take(),
                    _ => main.take(),
                }
                .ok_or_else(|| Error::Model(format!("the MLX network has no output {name}")))?;
                Array2::from_shape_vec((rows, width), data)
                    .map_err(|e| Error::Model(format!("{name}: {e}")))
            })
            .collect()
    }
}

type Output = (usize, Vec<f32>);

impl Model {
    fn load(arch: &Arch, weights: &Path) -> Result<Model, Error> {
        let t = Tensors::open(weights, &arch.weights)?;
        let Backbone::Modernbert(c) = &arch.backbone;
        let backbone = modernbert::ModernBert::load(c, &t)?;
        let d = c.hidden_size;
        let head = match &arch.head {
            Head::Laya(h) => HeadModel::Laya(heads::Laya::load(h, d, &t)?),
            Head::SequenceClassification(h) => HeadModel::SequenceClassification(
                heads::SequenceClassification::load(h, d, c.norm_bias, c.norm_eps, &t)?,
            ),
            Head::OptionMarker(h) => HeadModel::OptionMarker(heads::OptionMarker::load(h, d, &t)?),
            Head::GliclassUni(h) => HeadModel::Gliclass(heads::Gliclass::load(h, d, &t)?),
        };
        Ok(Model { backbone, head })
    }

    fn forward(
        &self,
        rows: usize,
        seq: usize,
        ids: Vec<i32>,
        attention: Vec<bool>,
        markers: Option<(usize, Vec<i32>, Vec<bool>)>,
        qtype: Option<Vec<i32>>,
    ) -> Result<(Output, Option<Output>), Error> {
        let (r, s) = (rows as i32, seq as i32);
        let ids_a = Array::from_slice(&ids, &[r, s]).map_err(mlx)?;
        let att = Array::from_slice(&attention, &[r, s]).map_err(mlx)?;
        let mut x = self.backbone.embed(&ids_a)?;
        if let HeadModel::Gliclass(g) = &self.head {
            let seg = Array::from_slice(&g.segment_ids(&ids, rows, seq), &[r, s]).map_err(mlx)?;
            x = x
                .add(&g.segments.take(&seg, 0).map_err(mlx)?)
                .map_err(mlx)?;
        }
        let h = self.backbone.forward(&x, &att)?;
        let marker_arrays = || -> Result<(Array, Array), Error> {
            let (k, pos, mask) = markers
                .as_ref()
                .ok_or_else(|| Error::Model("this head needs marker positions".into()))?;
            Ok((
                Array::from_slice(pos, &[r, *k as i32]).map_err(mlx)?,
                Array::from_slice(mask, &[r, *k as i32]).map_err(mlx)?,
            ))
        };
        let read = |a: &Array| -> Result<Output, Error> {
            let width = *a.shape().last().unwrap_or(&1) as usize;
            Ok((width, a.to_f32_vec().map_err(mlx)?))
        };
        match &self.head {
            HeadModel::Laya(head) => {
                let (pos, mask) = marker_arrays()?;
                let qt = qtype
                    .as_ref()
                    .ok_or_else(|| Error::Model("laya needs question types".into()))?;
                let qt = Array::from_slice(qt, &[r]).map_err(mlx)?;
                let (logits, act) = head.forward(&h, &att, &qt, &pos, &mask)?;
                Ok((read(&logits)?, Some(read(&act)?)))
            }
            HeadModel::SequenceClassification(head) => Ok((read(&head.forward(&h, &att)?)?, None)),
            HeadModel::OptionMarker(head) => {
                let (pos, mask) = marker_arrays()?;
                Ok((read(&head.forward(&h, &pos, &mask)?)?, None))
            }
            HeadModel::Gliclass(head) => {
                let (pos, mask) = marker_arrays()?;
                Ok((read(&head.forward(&h, &att, &pos, &mask)?)?, None))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::usable;

    #[test]
    fn only_models_with_an_arch_layer_and_a_known_layout_try_mlx() {
        let arch = Some(Path::new("arch.json"));
        assert!(
            usable(arch, "kev-pointer-v1")
                .unwrap_err()
                .contains("layout")
        );
        assert!(
            usable(None, "laya-markers-v1")
                .unwrap_err()
                .contains("arch layer")
        );
        // A development build finds the Metal library it built.
        assert!(usable(arch, "laya-markers-v1").unwrap().is_file());
    }
}
