//! The network behind an encoder engine: an ONNX Runtime session, or (with the `mlx` feature) an
//! MLX network on the Metal GPU. Both take the same padded batch and give the same outputs, so a
//! family engine keeps one tokenizer, layout and readout for every device.

use std::borrow::Cow;
use std::sync::Mutex;

use ndarray::{Array2, Ix2};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::Error;
use crate::onnx::{Device, ModelFiles};

/// One padded batch, in the tensor names every encoder graph uses.
pub struct Batch {
    /// `input_ids` [rows, seq], padded with the pad id.
    pub input_ids: Array2<i64>,
    /// `attention_mask` [rows, seq]: 1 for real tokens.
    pub attention: Array2<i64>,
    /// `marker_pos` and `marker_mask` [rows, markers] (contract M and Laya).
    pub markers: Option<(Array2<i64>, Array2<bool>)>,
    /// `qtype` [rows] (contract M and Laya).
    pub qtype: Option<Vec<i64>>,
}

/// The head an MLX network must have for a layout (ONNX graphs carry their own).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Head {
    Laya,
    SequenceClassification,
    OptionMarker,
    Gliclass,
}

impl Head {
    /// The arch layer's name for it.
    pub fn name(self) -> &'static str {
        match self {
            Head::Laya => "laya",
            Head::SequenceClassification => "sequence-classification",
            Head::OptionMarker => "option-marker",
            Head::Gliclass => "gliclass-uni",
        }
    }
}

/// The network for `device`: MLX on [`Device::Metal`], an ONNX Runtime session otherwise.
pub fn load(
    files: &ModelFiles,
    device: Device,
    intra_threads: Option<usize>,
    head: Head,
) -> Result<Net, Error> {
    match device {
        Device::Metal => metal(files, head),
        _ => Ok(Net::Ort(Mutex::new(crate::onnx::session(
            &files.graph,
            device,
            intra_threads,
        )?))),
    }
}

#[cfg(feature = "mlx")]
fn metal(files: &ModelFiles, head: Head) -> Result<Net, Error> {
    Ok(Net::Mlx(crate::mlx::MlxNet::load(files, head)?))
}

#[cfg(not(feature = "mlx"))]
fn metal(_files: &ModelFiles, _head: Head) -> Result<Net, Error> {
    Err(Error::Model(
        "this build of ollaya has no MLX support".into(),
    ))
}

pub enum Net {
    Ort(Mutex<Session>),
    #[cfg(feature = "mlx")]
    Mlx(crate::mlx::MlxNet),
}

impl Net {
    /// The ONNX Runtime session, if this network is one (for graph contract checks).
    pub fn session(&self) -> Option<&Mutex<Session>> {
        match self {
            Net::Ort(s) => Some(s),
            #[cfg(feature = "mlx")]
            Net::Mlx(_) => None,
        }
    }

    /// Run `batch` and return the named outputs, each [rows, width].
    pub fn run(&self, batch: Batch, outputs: &[&str]) -> Result<Vec<Array2<f32>>, Error> {
        match self {
            Net::Ort(session) => {
                let rows = batch.input_ids.nrows();
                let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = ort::inputs![
                    "input_ids" => Tensor::from_array(batch.input_ids)?,
                    "attention_mask" => Tensor::from_array(batch.attention)?,
                ];
                if let Some((pos, mask)) = batch.markers {
                    inputs.push(("marker_pos".into(), Tensor::from_array(pos)?.into()));
                    inputs.push(("marker_mask".into(), Tensor::from_array(mask)?.into()));
                }
                if let Some(qtype) = batch.qtype {
                    inputs.push(("qtype".into(), Tensor::from_array(([rows], qtype))?.into()));
                }
                let mut session = session.lock().expect("session mutex poisoned");
                let out = session.run(inputs)?;
                outputs
                    .iter()
                    .map(|name| {
                        let value = out.get(name).ok_or_else(|| {
                            Error::Model(format!("the graph has no output {name}"))
                        })?;
                        Ok(value
                            .try_extract_array::<f32>()?
                            .into_dimensionality::<Ix2>()
                            .map_err(|e| Error::Model(format!("{name}: {e}")))?
                            .to_owned())
                    })
                    .collect()
            }
            #[cfg(feature = "mlx")]
            Net::Mlx(net) => net.run(&batch, outputs),
        }
    }
}
