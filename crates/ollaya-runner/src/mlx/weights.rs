//! Tensors read straight from the author's weights files, upcast to float32 on the MLX thread.
//!
//! * safetensors: the index is the file's own JSON header.
//! * PyTorch zip (`torch.save`): every storage is an uncompressed zip member, so a tensor's bytes
//!   sit at a fixed offset. The arch layer lists those offsets (derived at build time); the pickle
//!   is never read here.
//!
//! A model's tensors may be spread over several files (a backbone and a head file); one index
//! covers them all, each name pointing into the file that holds it.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt as _;
use std::path::{Path, PathBuf};

use ollaya_mlx::{Array, Dtype};

use super::arch::{Lora, Weights, WeightsFormat};
use crate::Error;

#[derive(Debug, Clone)]
struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    offset: u64,
    len: u64,
    /// Which of `files` holds it.
    file: usize,
}

pub struct Tensors {
    files: Vec<(File, PathBuf)>,
    index: HashMap<String, Entry>,
    /// A LoRA adapter to merge into the backbone's weights: the backbone's name prefix, the
    /// adapter's, and the scaling.
    lora: Option<(String, String, f32)>,
}

fn dtype(tag: &str) -> Result<Dtype, Error> {
    match tag {
        "F32" => Ok(Dtype::Float32),
        "F16" => Ok(Dtype::Float16),
        "BF16" => Ok(Dtype::BFloat16),
        other => Err(Error::Model(format!("unsupported tensor dtype {other}"))),
    }
}

fn entry(dt: Dtype, shape: Vec<usize>, offset: u64) -> Entry {
    let len = shape.iter().product::<usize>() as u64 * dt.size() as u64;
    Entry {
        dtype: dt,
        shape,
        offset,
        len,
        file: 0,
    }
}

impl Tensors {
    /// Open the weights files the arch layer lists, `paths` in the same order (or, when `paths` is
    /// empty, the files by their upstream names in `dir`).
    pub fn open(paths: &[PathBuf], dir: &Path, weights: &Weights) -> Result<Tensors, Error> {
        let listed = weights.files();
        if !paths.is_empty() && paths.len() != listed.len() {
            return Err(Error::Model(format!(
                "the arch layer lists {} weights files, {} were given",
                listed.len(),
                paths.len()
            )));
        }
        let mut files = Vec::with_capacity(listed.len());
        let mut index: HashMap<String, Entry> = HashMap::new();
        for (i, (format, name, tensors)) in listed.into_iter().enumerate() {
            let path = paths.get(i).cloned().unwrap_or_else(|| dir.join(name));
            let err = |e: &dyn std::fmt::Display| Error::Model(format!("{}: {e}", path.display()));
            let file = File::open(&path).map_err(|e| err(&e))?;
            let size = file.metadata().map_err(|e| err(&e))?.len();
            let part = match format {
                WeightsFormat::Safetensors => safetensors_index(&file).map_err(|e| err(&e))?,
                WeightsFormat::TorchZip => tensors
                    .iter()
                    .map(|(name, t)| {
                        Ok((
                            name.clone(),
                            entry(dtype(&t.dtype)?, t.shape.clone(), t.offset),
                        ))
                    })
                    .collect::<Result<_, Error>>()?,
            };
            if let Some((name, _)) = part.iter().find(|(_, e)| e.offset + e.len > size) {
                return Err(err(&format!("tensor {name} lies past the end of the file")));
            }
            for (name, mut e) in part {
                e.file = i;
                index.entry(name).or_insert(e);
            }
            files.push((file, path));
        }
        Ok(Tensors {
            files,
            index,
            lora: None,
        })
    }

    /// Merge `lora` into every backbone weight (`<backbone_prefix>…weight`) that the adapter holds
    /// `lora_A` and `lora_B` for, as [`Self::get`] reads them.
    pub fn with_lora(mut self, backbone_prefix: &str, lora: &Lora) -> Tensors {
        self.lora = Some((backbone_prefix.into(), lora.prefix.clone(), lora.scaling));
        self
    }

    /// The adapter's `(A, B)` for a backbone weight, when it has one.
    fn adapter(&self, name: &str) -> Option<(String, String)> {
        let (backbone, prefix, _) = self.lora.as_ref()?;
        let rel = name
            .strip_prefix(backbone.as_str())?
            .strip_suffix(".weight")?;
        let (a, b) = (
            format!("{prefix}{rel}.lora_A.weight"),
            format!("{prefix}{rel}.lora_B.weight"),
        );
        (self.index.contains_key(&a) && self.index.contains_key(&b)).then_some((a, b))
    }

    /// `name` as float32, checked against `shape`; a LoRA-adapted weight comes merged.
    pub fn get(&self, name: &str, shape: &[usize]) -> Result<Array, Error> {
        let w = self.raw(name, shape)?;
        let Some((a, b)) = self.adapter(name) else {
            return Ok(w);
        };
        let scaling = self.lora.as_ref().map(|l| l.2).unwrap_or(1.0);
        let r = self
            .shape(&a)
            .map(|s| s[0])
            .ok_or_else(|| Error::Model(format!("no tensor {a}")))?;
        let (a, b) = (self.raw(&a, &[r, shape[1]])?, self.raw(&b, &[shape[0], r])?);
        let merged = b
            .matmul(&a)
            .and_then(|d| d.multiply(&Array::scalar(scaling)))
            .and_then(|d| w.add(&d))
            .map_err(mlx)?;
        merged.eval().map_err(mlx)?;
        Ok(merged)
    }

    /// `name` as float32, checked against `shape`, as stored.
    fn raw(&self, name: &str, shape: &[usize]) -> Result<Array, Error> {
        let e = self.index.get(name).ok_or_else(|| {
            Error::Model(format!("{}: no tensor {name}", self.files[0].1.display()))
        })?;
        let (file, path) = &self.files[e.file];
        if e.shape != shape {
            return Err(Error::Model(format!(
                "{}: tensor {name} has shape {:?}, expected {shape:?}",
                path.display(),
                e.shape
            )));
        }
        let mut bytes = vec![0u8; e.len as usize];
        file.read_exact_at(&mut bytes, e.offset)
            .map_err(|err| Error::Model(format!("{}: {name}: {err}", path.display())))?;
        let dims: Vec<i32> = e.shape.iter().map(|&d| d as i32).collect();
        let a = Array::from_bytes(&bytes, &dims, e.dtype).map_err(mlx)?;
        let a = if e.dtype == Dtype::Float32 {
            a
        } else {
            a.astype(Dtype::Float32).map_err(mlx)?
        };
        // Materialise now, so the file bytes and any half-precision copy are freed per tensor.
        a.eval().map_err(mlx)?;
        Ok(a)
    }

    /// The shape of `name`, if present.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.index.get(name).map(|e| e.shape.as_slice())
    }
}

pub fn mlx(e: ollaya_mlx::Error) -> Error {
    Error::Model(e.to_string())
}

/// The header of a safetensors file: `u64` length, then JSON `{name: {dtype, shape,
/// data_offsets: [begin, end]}}` with offsets relative to the end of the header.
fn safetensors_index(file: &File) -> Result<HashMap<String, Entry>, String> {
    let mut len = [0u8; 8];
    file.read_exact_at(&mut len, 0).map_err(|e| e.to_string())?;
    let n = u64::from_le_bytes(len);
    if n > 100 << 20 {
        return Err(format!("safetensors header of {n} bytes"));
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact_at(&mut header, 8)
        .map_err(|e| e.to_string())?;
    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|e| e.to_string())?;
    let base = 8 + n;
    let mut index = HashMap::new();
    for (name, v) in header {
        if name == "__metadata__" {
            continue;
        }
        let tag = v["dtype"].as_str().ok_or("tensor without dtype")?;
        let shape: Vec<usize> =
            serde_json::from_value(v["shape"].clone()).map_err(|e| e.to_string())?;
        let offsets: [u64; 2] =
            serde_json::from_value(v["data_offsets"].clone()).map_err(|e| e.to_string())?;
        // Tensors of dtypes the models never read (I64 buffers and the like) are left out.
        let Ok(dt) = dtype(tag) else { continue };
        let e = entry(dt, shape, base + offsets[0]);
        if offsets[1] - offsets[0] != e.len {
            return Err(format!(
                "tensor {name}: {} bytes for its shape",
                offsets[1] - offsets[0]
            ));
        }
        index.insert(name, e);
    }
    Ok(index)
}
