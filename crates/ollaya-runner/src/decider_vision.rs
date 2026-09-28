//! ONNX Runtime engine for Mapika's decider-2b-vision (layout `decider-vision-v1`).
//!
//! Text rows are `decider-slots-v1` rows narrowed to ten letters (v5's prompt): one row per
//! question, the shared context, then the question, ending at the `(` slot. Upstream decodes each
//! row's ids and tokenizes the text again, with the image placeholder in front, so a row is
//!
//! ```text
//! [vision_start] + [image] * n + [vision_end] + encode(decode(ids))      with an image
//! encode(decode(ids))                                                   without
//! ```
//!
//! Two graphs, both reading the author's `model.safetensors`:
//! * `vision.onnx`: `patches` [N, 1536], `pos_idx` / `pos_w` [N, 4], `rot_ids` [N, 2] ->
//!   `image_embeds` [N / 4, 2048] (see [`crate::vision`] for the inputs);
//! * `model.onnx`: `input_ids` [R, T] (T a multiple of `chunk`), `position_ids` [3, R, T]
//!   (interleaved M-RoPE: text positions count up on all three axes; an image token at merged
//!   grid cell (i, j) of an image starting at p is (p, p + i, p + j), and the text after the image
//!   continues from p + max(rows, columns)), `image_embeds` [M, 2048] placed at the flat indices
//!   `image_pos` [M] (r * T + t), `slot_pos` [R] -> `letter_logits` [R, 10].
//!
//! Without an image the decoder still takes one image slot: a zero embedding at a padding
//! position after every row's tokens, which no slot can attend to (the model is causal).

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, Ix2};
use ollaya_decision::decider::{Context, DeciderLayout, Scoring};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::{WeightsInMemory, configure};
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::vision::{self, ImageConfig, Patches, Rgb};
use crate::{Error, Output, QuestionOutput};

/// Rows per decoder run: the export's row axis is 1..=1024.
const MAX_ROWS: usize = 1024;
/// Padded tokens per decoder run, as for the text decider.
const TOKEN_BUDGET: usize = 8192;
/// Patches per image: 4,096 (1,024 visual tokens), about one megapixel after the resize. The vision
/// tower attends over all of an image's patches at once and every row carries the image, so memory
/// grows fast with the image. Measured on an RTX 4090 with five questions: 8.3 GB for a 256x240
/// image, 10.6 GB at 0.8 MP, 13.7 GB at 1.3 MP, 19.3 GB at 2 MP.
pub const MAX_PATCHES: usize = 4096;
const HIDDEN: usize = 2048;
const VISION_INPUTS: [&str; 4] = ["patches", "pos_idx", "pos_w", "rot_ids"];
const VISION_OUTPUT: &str = "image_embeds";
const INPUTS: [&str; 5] = [
    "input_ids",
    "position_ids",
    "image_embeds",
    "image_pos",
    "slot_pos",
];
const OUTPUT: &str = "letter_logits";

#[derive(Debug, Clone, Deserialize)]
struct VisionTokens {
    image: u32,
    vision_start: u32,
    vision_end: u32,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    /// Rows are padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    chunk: usize,
    tokens: VisionTokens,
    image: ImageConfig,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    decider: DeciderLayout,
}

/// The rows of one request and the image they share.
#[derive(Debug, Clone)]
pub struct VisionEncoding {
    pub questions: Vec<Scoring>,
    pub context: Context,
    /// Each row's (t, h, w) positions, in row order.
    pub positions: Vec<[Vec<i64>; 3]>,
    pub image: Option<EncodedImage>,
}

/// An image after preprocessing.
#[derive(Debug, Clone)]
pub struct EncodedImage {
    pub resized: Rgb,
    pub patches: Patches,
    /// Visual tokens each row holds (after the 2x2 merge).
    pub tokens: usize,
}

pub struct VisionDeciderModel {
    vision: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    chunk: usize,
    tokens: VisionTokens,
    pub image: ImageConfig,
    pub layout: DeciderLayout,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

impl Tokenizer {
    /// `encode(decode(ids))`, as upstream's `tok.decode` then the processor's tokenizer call.
    fn retokenize(&self, ids: &[u32]) -> Result<Vec<u32>, Error> {
        let text = self
            .0
            .decode(ids, false)
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))?;
        Ok(self.encode(&text)?)
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

fn check_io(session: &Session, inputs: &[&str], output: &str, what: &str) -> Result<(), Error> {
    let got: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
    if got.len() != inputs.len()
        || !inputs.iter().all(|n| got.contains(n))
        || !session.outputs().iter().any(|o| o.name() == output)
    {
        return Err(Error::Model(format!(
            "{what} graph inputs {got:?} do not match the decider-vision contract ({inputs:?} -> {output:?})"
        )));
    }
    Ok(())
}

impl Engine for VisionDeciderModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        VisionDeciderModel::run(self, state, questions, None)
    }

    fn reads_images(&self) -> bool {
        true
    }

    fn run_images(
        &self,
        state: &Value,
        questions: &Value,
        images: &[Vec<u8>],
    ) -> Result<Output, Error> {
        let questions = ollaya_decision::parse_questions(questions)?;
        let image = match images {
            [] => None,
            [one] => Some(vision::decode(one)?),
            _ => {
                return Err(Error::Image(vision::ImageError::Count(images.len())));
            }
        };
        VisionDeciderModel::run(self, state, &questions, image.as_ref())
    }
}

impl VisionDeciderModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != "decider-vision-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/decider-vision-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.decider.validate().map_err(|e| bad(e.to_string()))?;
        if config.chunk == 0 {
            return Err(bad("chunk must be positive".into()));
        }
        let im = &config.image;
        if im.patch_size == 0 || im.merge_size == 0 || im.temporal_patch_size == 0 {
            return Err(bad(
                "image: patch, merge and temporal sizes must be positive".into(),
            ));
        }
        let vision_graph = files
            .vision
            .as_ref()
            .ok_or_else(|| Error::Model("a decider-vision model needs its vision graph".into()))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;

        let weights = config.weights_in_memory;
        let decoder = session_for(
            &files.graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(configure(b, device)?),
        )?;
        check_io(&decoder, &INPUTS, OUTPUT, "decoder")?;
        let vision = session_for(
            vision_graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(configure(b, device)?),
        )?;
        check_io(&vision, &VISION_INPUTS, VISION_OUTPUT, "vision")?;

        Ok(VisionDeciderModel {
            vision: Mutex::new(vision),
            decoder: Mutex::new(decoder),
            tokenizer: Tokenizer(tokenizer),
            chunk: config.chunk,
            tokens: config.tokens,
            image: config.image,
            layout: config.decider,
            calibration,
            device,
        })
    }

    /// Resize and cut an image into the vision graph's inputs.
    pub fn prepare_image(&self, image: &Rgb) -> Result<EncodedImage, Error> {
        let (resized, patches) = vision::preprocess(image, &self.image)?;
        if patches.count() > MAX_PATCHES {
            return Err(Error::Image(vision::ImageError::TooLarge {
                width: image.width,
                height: image.height,
                resized: (resized.width, resized.height),
                patches: patches.count(),
                max: MAX_PATCHES,
            }));
        }
        let tokens = patches.tokens(self.image.merge_size);
        Ok(EncodedImage {
            resized,
            patches,
            tokens,
        })
    }

    /// Every question's row, after the shared context and the image.
    pub fn encode(
        &self,
        state: &Value,
        questions: &Questions,
        image: Option<&Rgb>,
    ) -> Result<VisionEncoding, Error> {
        let image = image.map(|i| self.prepare_image(i)).transpose()?;
        let context = self.layout.encode_context(&self.tokenizer, state)?;
        let mut scorings = Vec::with_capacity(questions.len());
        let mut positions = Vec::new();
        for (qid, q) in questions.iter() {
            let mut s = self.layout.encode(&self.tokenizer, &context.ids, qid, q)?;
            for row in &mut s.rows {
                let text = self.tokenizer.retokenize(row)?;
                let (ids, pos) = self.with_image(&text, image.as_ref());
                *row = ids;
                positions.push(pos);
            }
            scorings.push(s);
        }
        Ok(VisionEncoding {
            questions: scorings,
            context,
            positions,
            image,
        })
    }

    /// A row's ids with the image in front, and its (t, h, w) positions.
    fn with_image(&self, text: &[u32], image: Option<&EncodedImage>) -> (Vec<u32>, [Vec<i64>; 3]) {
        let Some(img) = image else {
            let p: Vec<i64> = (0..text.len() as i64).collect();
            return (text.to_vec(), [p.clone(), p.clone(), p]);
        };
        let m = self.image.merge_size;
        let (gh, gw) = (img.patches.grid_h / m, img.patches.grid_w / m);
        let len = text.len() + img.tokens + 2;
        let mut ids = Vec::with_capacity(len);
        ids.push(self.tokens.vision_start);
        ids.extend(std::iter::repeat_n(self.tokens.image, img.tokens));
        ids.push(self.tokens.vision_end);
        ids.extend_from_slice(text);
        let mut pos: [Vec<i64>; 3] = std::array::from_fn(|_| Vec::with_capacity(len));
        for p in &mut pos {
            p.push(0);
        }
        let start = 1i64;
        for i in 0..gh as i64 {
            for j in 0..gw as i64 {
                pos[0].push(start);
                pos[1].push(start + i);
                pos[2].push(start + j);
            }
        }
        let next = start + gh.max(gw) as i64;
        for k in 0..text.len() as i64 + 1 {
            for p in &mut pos {
                p.push(next + k);
            }
        }
        (ids, pos)
    }

    /// Answer every question, about the image when there is one.
    pub fn run(
        &self,
        state: &Value,
        questions: &Questions,
        image: Option<&Rgb>,
    ) -> Result<Output, Error> {
        let encoding = self.encode(state, questions, image)?;
        let logits = self.letter_logits(&encoding)?;
        Ok(self.output(&encoding, &logits))
    }

    /// The vision tower's output for an image: [tokens, 2048], row-major.
    pub fn image_embeds(&self, image: &EncodedImage) -> Result<Vec<f32>, Error> {
        let p = &image.patches;
        let n = p.count();
        let patches = Array2::from_shape_vec((n, p.dim), p.values.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let pos_idx = Array2::from_shape_vec((n, 4), p.pos_idx.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let pos_w = Array2::from_shape_vec((n, 4), p.pos_w.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let rot_ids = Array2::from_shape_vec((n, 2), p.rot_ids.clone())
            .map_err(|e| Error::Model(e.to_string()))?;
        let mut session = self.vision.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "patches" => ort::value::Tensor::from_array(patches)?,
            "pos_idx" => ort::value::Tensor::from_array(pos_idx)?,
            "pos_w" => ort::value::Tensor::from_array(pos_w)?,
            "rot_ids" => ort::value::Tensor::from_array(rot_ids)?,
        ])?;
        let out = outputs[VISION_OUTPUT]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("{VISION_OUTPUT}: {e}")))?;
        if out.shape() != [image.tokens, HIDDEN] {
            return Err(Error::Model(format!(
                "{VISION_OUTPUT} has shape {:?} for {} visual tokens",
                out.shape(),
                image.tokens
            )));
        }
        Ok(out.iter().copied().collect())
    }

    /// Letter logits of every row, in row order. Rows run shortest first, in batches padded to a
    /// multiple of `chunk` with at least one padding position (the image slot of a text-only
    /// batch).
    pub fn letter_logits(&self, encoding: &VisionEncoding) -> Result<Vec<Vec<f32>>, Error> {
        let rows: Vec<&[u32]> = encoding
            .questions
            .iter()
            .flat_map(|s| s.rows.iter().map(Vec::as_slice))
            .collect();
        let width = encoding
            .questions
            .iter()
            .map(|s| s.options)
            .max()
            .unwrap_or(0)
            .max(2);
        let embeds = encoding
            .image
            .as_ref()
            .map(|i| self.image_embeds(i))
            .transpose()?;
        let image_tokens = encoding.image.as_ref().map_or(0, |i| i.tokens);
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| (rows[i].len() + 1).div_ceil(self.chunk) * self.chunk)
            .collect();
        let pad = i64::from(self.layout.special_tokens.pad);

        let mut logits = vec![Vec::new(); rows.len()];
        let mut session = self.decoder.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let n = batch.len();
            let mut input_ids = Array2::<i64>::from_elem((n, seq), pad);
            let mut position_ids = Array3::<i64>::zeros((3, n, seq));
            let mut slot_pos = Array1::<i64>::zeros(n);
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in rows[i].iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                for (a, axis) in encoding.positions[i].iter().enumerate() {
                    for (c, &p) in axis.iter().enumerate() {
                        position_ids[[a, r, c]] = p;
                    }
                }
                slot_pos[r] = rows[i].len() as i64 - 1;
            }
            let (image_embeds, image_pos) = match &embeds {
                Some(e) => {
                    let mut all = Vec::with_capacity(n * e.len());
                    let mut pos = Vec::with_capacity(n * image_tokens);
                    for r in 0..n {
                        all.extend_from_slice(e);
                        pos.extend((0..image_tokens).map(|j| (r * seq + 1 + j) as i64));
                    }
                    (
                        Array2::from_shape_vec((n * image_tokens, HIDDEN), all)
                            .map_err(|e| Error::Model(e.to_string()))?,
                        Array1::from_vec(pos),
                    )
                }
                // Every row is shorter than `seq`, so row 0's last position is padding.
                None => (
                    Array2::<f32>::zeros((1, HIDDEN)),
                    Array1::from_vec(vec![seq as i64 - 1]),
                ),
            };
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "position_ids" => ort::value::Tensor::from_array(position_ids)?,
                "image_embeds" => ort::value::Tensor::from_array(image_embeds)?,
                "image_pos" => ort::value::Tensor::from_array(image_pos)?,
                "slot_pos" => ort::value::Tensor::from_array(slot_pos)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != n || out.ncols() < width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {n} rows of up to {width} options",
                    out.shape(),
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.to_vec();
            }
        }
        Ok(logits)
    }

    /// Option logits per question from the letter logits of every row.
    pub fn output(&self, encoding: &VisionEncoding, logits: &[Vec<f32>]) -> Output {
        let mut next = 0;
        let questions = encoding
            .questions
            .iter()
            .map(|s| {
                let rows = &logits[next..next + s.rows.len()];
                next += s.rows.len();
                QuestionOutput {
                    logits: self.layout.option_logits(
                        s.readout,
                        s.options,
                        rows.iter().map(Vec::as_slice),
                    ),
                    act_logits: None,
                }
            })
            .collect();
        Output {
            questions,
            input_tokens: encoding
                .questions
                .iter()
                .flat_map(|s| &s.rows)
                .map(Vec::len)
                .sum(),
            state_tokens: encoding.context.tokens,
            state_truncated: encoding.context.truncated,
        }
    }
}
