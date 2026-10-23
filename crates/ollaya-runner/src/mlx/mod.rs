//! The MLX engine: encoder networks on the Metal GPU, from the author's weights file.
//!
//! A family engine (`OnnxModel`, `NliModel`, `VonModel`, `GliclassModel`) keeps its tokenizer,
//! layout and readout, and swaps its ONNX Runtime session for an [`MlxNet`] on
//! [`Device::Metal`](crate::Device::Metal). The network is our own fp32 code per backbone
//! ([`modernbert`], [`deberta`]) and per head ([`heads`]), built from the model's `arch` layer;
//! every op runs on the runner's one MLX thread ([`ollaya_mlx::Worker`]).
//!
//! Design and parity evidence: `docs/decisions/0001-mlx-engine.md`.

pub mod arch;
mod deberta;
mod heads;
mod modernbert;
mod qwen3;
mod qwen35;
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
pub const LAYOUTS: &[&str] = &[
    "laya-markers-v1",
    "nli-pairs-v1",
    "gliclass-uni-v1",
    "qwen3guard-gen-v1",
    "decider-slots-v1",
    "decision-endpoint-v1",
    "kev-pointer-v1",
];

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
    backbone: Encoder,
    head: HeadModel,
}

enum Encoder {
    ModernBert(modernbert::ModernBert),
    DebertaV2(deberta::DebertaV2),
}

impl Encoder {
    fn embed(&self, ids: &Array) -> Result<Array, Error> {
        match self {
            Encoder::ModernBert(m) => m.embed(ids),
            Encoder::DebertaV2(m) => m.embed(ids),
        }
    }

    fn forward(&self, x: &Array, attention: &Array) -> Result<Array, Error> {
        match self {
            Encoder::ModernBert(m) => m.forward(x, attention),
            Encoder::DebertaV2(m) => m.forward(x, attention),
        }
    }

    /// How many rows one pass takes: DeBERTa's scores and biases are [rows, heads, seq, seq].
    fn rows_per_pass(&self, seq: usize) -> usize {
        match self {
            Encoder::ModernBert(_) => usize::MAX,
            Encoder::DebertaV2(m) => m.rows_per_pass(seq),
        }
    }
}

enum HeadModel {
    Laya(heads::Laya),
    SequenceClassification(heads::SequenceClassification),
    ContextPooler(heads::ContextPooler),
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
        let (paths, dir) = (files.weights.clone(), arch_dir(arch_path));
        let metallib = metallib().map_err(Error::Model)?;
        let worker = Worker::start("ollaya-mlx", &metallib, move || {
            Model::load(&arch, &paths, &dir).map_err(|e| ollaya_mlx::Error(e.to_string()))
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
                m.forward_chunked(rows, seq, ids, attention, markers, qtype)
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

/// The directory of a development model: the weights files sit next to `arch.json` under their
/// upstream names.
fn arch_dir(arch_path: &Path) -> PathBuf {
    arch_path.parent().unwrap_or(Path::new(".")).to_path_buf()
}

impl Model {
    fn load(arch: &Arch, paths: &[PathBuf], dir: &Path) -> Result<Model, Error> {
        let t = Tensors::open(paths, dir, &arch.weights)?;
        let backbone = match &arch.backbone {
            Backbone::Modernbert(c) => Encoder::ModernBert(modernbert::ModernBert::load(c, &t)?),
            Backbone::DebertaV2(c) => Encoder::DebertaV2(deberta::DebertaV2::load(c, &t)?),
            Backbone::Qwen3(_) | Backbone::Qwen35(_) => {
                return Err(Error::Model(
                    "a decoder backbone needs a decoder head".into(),
                ));
            }
        };
        let d = arch.backbone.hidden_size();
        let head = match &arch.head {
            Head::Laya(h) => HeadModel::Laya(heads::Laya::load(h, d, &t)?),
            Head::SequenceClassification(h) => {
                // ModernBERT's head shares the encoder's norm settings.
                let Backbone::Modernbert(c) = &arch.backbone else {
                    return Err(Error::Model(
                        "the sequence-classification head needs a ModernBERT backbone".into(),
                    ));
                };
                HeadModel::SequenceClassification(heads::SequenceClassification::load(
                    h,
                    d,
                    c.norm_bias,
                    c.norm_eps,
                    &t,
                )?)
            }
            Head::ContextPooler(h) => {
                HeadModel::ContextPooler(heads::ContextPooler::load(h, d, &t)?)
            }
            Head::OptionMarker(h) => HeadModel::OptionMarker(heads::OptionMarker::load(h, d, &t)?),
            Head::GliclassUni(h) => HeadModel::Gliclass(heads::Gliclass::load(h, d, &t)?),
            Head::TiedRows | Head::Endpoint(_) | Head::Pointer(_) => {
                return Err(Error::Model(
                    "a decoder head needs a decoder backbone".into(),
                ));
            }
        };
        Ok(Model { backbone, head })
    }

    /// [`Self::forward`] in passes of as many rows as the backbone takes at once, the outputs
    /// joined in row order.
    fn forward_chunked(
        &self,
        rows: usize,
        seq: usize,
        ids: Vec<i32>,
        attention: Vec<bool>,
        markers: Option<(usize, Vec<i32>, Vec<bool>)>,
        qtype: Option<Vec<i32>>,
    ) -> Result<(Output, Option<Output>), Error> {
        let per = self.backbone.rows_per_pass(seq).min(rows.max(1));
        if per >= rows {
            return self.forward(rows, seq, ids, attention, markers, qtype);
        }
        let mut main: Option<Output> = None;
        let mut act: Option<Output> = None;
        let mut start = 0;
        while start < rows {
            let end = (start + per).min(rows);
            let part = |v: &[i32], width: usize| v[start * width..end * width].to_vec();
            let (m, a) = self.forward(
                end - start,
                seq,
                part(&ids, seq),
                attention[start * seq..end * seq].to_vec(),
                markers
                    .as_ref()
                    .map(|(k, pos, mask)| (*k, part(pos, *k), mask[start * k..end * k].to_vec())),
                qtype.as_ref().map(|q| q[start..end].to_vec()),
            )?;
            let join = |into: &mut Option<Output>, (width, data): Output| match into {
                Some((_, v)) => v.extend(data),
                None => *into = Some((width, data)),
            };
            join(&mut main, m);
            if let Some(a) = a {
                join(&mut act, a);
            }
            start = end;
        }
        Ok((main.expect("at least one pass"), act))
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
            HeadModel::ContextPooler(head) => Ok((read(&head.forward(&h)?)?, None)),
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

// --- decoders --------------------------------------------------------------------------------------

/// A causal decoder and its head, on the MLX thread.
struct DecoderModel {
    backbone: DecoderBackbone,
    head: DecoderHead,
}

enum DecoderBackbone {
    Qwen3(qwen3::Qwen3),
    Qwen35(qwen35::Qwen35),
}

enum DecoderHead {
    /// The tied LM head at one position per row: no weights of its own.
    TiedRows,
    Endpoint(heads::Endpoint),
    Pointer(heads::Pointer),
}

impl DecoderBackbone {
    /// The token embedding, the tied LM head.
    fn embed(&self) -> &Array {
        match self {
            DecoderBackbone::Qwen3(m) => &m.embed,
            DecoderBackbone::Qwen35(m) => &m.embed,
        }
    }

    /// Rows are padded to a multiple of this many tokens.
    fn seq_multiple(&self) -> i32 {
        match self {
            DecoderBackbone::Qwen3(_) => 1,
            DecoderBackbone::Qwen35(_) => qwen35::CHUNK,
        }
    }

    fn forward(&self, ids: &Array) -> Result<Array, Error> {
        match self {
            DecoderBackbone::Qwen3(m) => m.forward(ids),
            DecoderBackbone::Qwen35(m) => m.forward(ids),
        }
    }
}

/// `ids` [rows, seq] right-padded, padded further to `multiple` with the last column's id
/// (causal: never read); returns the padded ids and the padded length.
fn pad_rows(ids: Vec<i32>, rows: usize, seq: usize, multiple: usize) -> (Vec<i32>, usize) {
    let padded = seq.div_ceil(multiple) * multiple;
    if padded == seq {
        return (ids, seq);
    }
    let mut out = Vec::with_capacity(rows * padded);
    for r in 0..rows {
        let row = &ids[r * seq..(r + 1) * seq];
        out.extend_from_slice(row);
        out.extend(std::iter::repeat_n(row[seq - 1], padded - seq));
    }
    (out, padded)
}

/// A decoder network (qwen3guard) on the MLX thread: rows of token ids in, the tied LM head at one
/// position per row out.
pub struct MlxDecoder {
    worker: Worker<DecoderModel>,
}

impl MlxDecoder {
    pub fn load(files: &ModelFiles, expect: HeadKind) -> Result<MlxDecoder, Error> {
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
        let (paths, dir) = (files.weights.clone(), arch_dir(arch_path));
        let metallib = metallib().map_err(Error::Model)?;
        let worker = Worker::start("ollaya-mlx", &metallib, move || {
            let t = Tensors::open(&paths, &dir, &arch.weights)
                .map_err(|e| ollaya_mlx::Error(e.to_string()))?;
            let t = match &arch.lora {
                Some(lora) => t.with_lora(arch.backbone.prefix(), lora),
                None => t,
            };
            let load = || -> Result<DecoderModel, Error> {
                let backbone = match &arch.backbone {
                    Backbone::Qwen3(c) => DecoderBackbone::Qwen3(qwen3::Qwen3::load(c, &t)?),
                    Backbone::Qwen35(c) => DecoderBackbone::Qwen35(qwen35::Qwen35::load(c, &t)?),
                    _ => {
                        return Err(Error::Model(
                            "a decoder head needs a decoder backbone".into(),
                        ));
                    }
                };
                let d = arch.backbone.hidden_size();
                let head = match &arch.head {
                    Head::TiedRows => DecoderHead::TiedRows,
                    Head::Endpoint(h) => DecoderHead::Endpoint(heads::Endpoint::load(h, d, &t)?),
                    Head::Pointer(h) => DecoderHead::Pointer(heads::Pointer::load(h, d, &t)?),
                    other => {
                        return Err(Error::Model(format!(
                            "head {} is an encoder's",
                            other.name()
                        )));
                    }
                };
                Ok(DecoderModel { backbone, head })
            };
            load().map_err(|e| ollaya_mlx::Error(e.to_string()))
        })
        .map_err(mlx)?;
        Ok(MlxDecoder { worker })
    }

    pub fn info(&self) -> &MetalInfo {
        self.worker.info()
    }

    /// The tied LM head at `pos[r]` of each row of `ids` [rows, seq] (right-padded), restricted to
    /// `tokens`: [rows, tokens.len()]. Rows are padded further to the backbone's multiple.
    pub fn tied_rows(
        &self,
        ids: &Array2<i64>,
        pos: &[i64],
        tokens: &[u32],
    ) -> Result<Array2<f32>, Error> {
        let (rows, seq) = ids.dim();
        let ids: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        let pos: Vec<i32> = pos.iter().map(|&v| v as i32).collect();
        let tokens: Vec<i32> = tokens.iter().map(|&v| v as i32).collect();
        let width = tokens.len();
        let data = self
            .worker
            .run(move |m| {
                let r = || -> Result<Vec<f32>, Error> {
                    if !matches!(m.head, DecoderHead::TiedRows) {
                        return Err(Error::Model("this model's head is not tied-rows".into()));
                    }
                    let (ids, padded) =
                        pad_rows(ids, rows, seq, m.backbone.seq_multiple() as usize);
                    let ids =
                        Array::from_slice(&ids, &[rows as i32, padded as i32]).map_err(mlx)?;
                    let pos = Array::from_slice(&pos, &[rows as i32]).map_err(mlx)?;
                    let tokens = Array::from_slice(&tokens, &[width as i32]).map_err(mlx)?;
                    let h = m.backbone.forward(&ids)?;
                    qwen3::tied_rows(&h, m.backbone.embed(), &pos, &tokens)?
                        .to_f32_vec()
                        .map_err(mlx)
                };
                r().map_err(|e| ollaya_mlx::Error(e.to_string()))
            })
            .map_err(mlx)?;
        Array2::from_shape_vec((rows, width), data).map_err(|e| Error::Model(e.to_string()))
    }

    /// Decision 1.0's candidate head or kev's pointer head: the logits [rows, k] of each row's
    /// options (`cand_pos` [rows, k], padded with 0) against its query or decide token (`query_pos`
    /// [rows]).
    pub fn endpoint(
        &self,
        ids: &Array2<i64>,
        query_pos: &[i64],
        cand_pos: &Array2<i64>,
    ) -> Result<Array2<f32>, Error> {
        let (rows, seq) = ids.dim();
        let k = cand_pos.ncols();
        let ids: Vec<i32> = ids.iter().map(|&v| v as i32).collect();
        let query_pos: Vec<i32> = query_pos.iter().map(|&v| v as i32).collect();
        let cand_pos: Vec<i32> = cand_pos.iter().map(|&v| v as i32).collect();
        let data = self
            .worker
            .run(move |m| {
                let r = || -> Result<Vec<f32>, Error> {
                    let (ids, padded) =
                        pad_rows(ids, rows, seq, m.backbone.seq_multiple() as usize);
                    let ids =
                        Array::from_slice(&ids, &[rows as i32, padded as i32]).map_err(mlx)?;
                    let q = Array::from_slice(&query_pos, &[rows as i32]).map_err(mlx)?;
                    let c = Array::from_slice(&cand_pos, &[rows as i32, k as i32]).map_err(mlx)?;
                    let h = m.backbone.forward(&ids)?;
                    match &m.head {
                        DecoderHead::Endpoint(head) => head.forward(&h, &q, &c)?,
                        DecoderHead::Pointer(head) => head.forward(&h, &q, &c)?,
                        DecoderHead::TiedRows => {
                            return Err(Error::Model("this model's head scores no options".into()));
                        }
                    }
                    .to_f32_vec()
                    .map_err(mlx)
                };
                r().map_err(|e| ollaya_mlx::Error(e.to_string()))
            })
            .map_err(mlx)?;
        Array2::from_shape_vec((rows, k), data).map_err(|e| Error::Model(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::usable;

    #[test]
    fn only_models_with_an_arch_layer_and_a_known_layout_try_mlx() {
        let arch = Some(Path::new("arch.json"));
        assert!(usable(arch, "clm-v1").unwrap_err().contains("layout"));
        assert!(
            usable(None, "laya-markers-v1")
                .unwrap_err()
                .contains("arch layer")
        );
        // A development build finds the Metal library it built.
        assert!(usable(arch, "laya-markers-v1").unwrap().is_file());
    }
}
