//! ONNX Runtime engine for Valen-Team's Valen-Decision-1.5 (layout `valen-decision-1.5-v1`).
//!
//! A request is one sequence holding the state, its image and every question
//! (`ollaya_decision::valen`). Two graphs, both reading the author's safetensors:
//! * `vision.onnx`: `patches` [N, 1536], `pos_idx` / `pos_w` [N, 4], `rot_ids` [N, 2] ->
//!   `image_embeds` [N / 4, hidden] (see [`crate::vision`] for the inputs);
//! * `model.onnx`: `input_ids` [1, T] (T a multiple of `chunk` with at least one padding
//!   position), `position_ids` [3, 1, T] (interleaved M-RoPE), `image_embeds` [M, hidden] placed
//!   at `image_pos` [M], `blocks` [1, T] (each token's attention block: the full-attention layers
//!   use upstream's `bbox_blocks` mask, the Gated DeltaNet layers stay causal),
//!   `instruction_spans` [Q, 2], `decision_pos` [Q], `candidate_spans` [O, 2] and
//!   `candidate_question` [O] -> the Mixer head's raw `logits` [O], every question's candidates
//!   in turn.
//!
//! Without an image the decoder still takes one image slot: a zero embedding at the last padding
//! position, which no token attends to (padding is outside every block, and the DeltaNet layers
//! are causal).

use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, Ix1, Ix2};
use ollaya_decision::valen::{OffsetEncoder, Offsets, PADDING, Valen15Layout, Valen15Row};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::{WeightsInMemory, configure};
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::vision::{self, ImageConfig, Patches, Rgb};
use crate::{Error, Output, QuestionOutput};

const VISION_INPUTS: [&str; 4] = ["patches", "pos_idx", "pos_w", "rot_ids"];
const VISION_OUTPUT: &str = "image_embeds";
const INPUTS: [&str; 9] = [
    "input_ids",
    "position_ids",
    "image_embeds",
    "image_pos",
    "blocks",
    "instruction_spans",
    "decision_pos",
    "candidate_spans",
    "candidate_question",
];
const OUTPUT: &str = "logits";
/// A request longer than this gives the CUDA arena back after its run (`engine::run_options`).
const TOKEN_BUDGET: usize = 8192;

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    /// The sequence is padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    chunk: usize,
    image: ImageConfig,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    valen: Valen15Layout,
}

/// An image after preprocessing.
#[derive(Debug, Clone)]
pub struct Valen15Image {
    pub resized: Rgb,
    pub patches: Patches,
    /// The merged (rows, columns) grid of visual tokens.
    pub grid: (usize, usize),
}

/// A request's sequence and the image it holds.
#[derive(Debug, Clone)]
pub struct Valen15Encoding {
    pub row: Valen15Row,
    pub image: Option<Valen15Image>,
}

pub struct Valen15Model {
    vision: Mutex<Session>,
    decoder: Mutex<Session>,
    tokenizer: Tokenizer,
    chunk: usize,
    pub image: ImageConfig,
    pub layout: Valen15Layout,
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

impl OffsetEncoder for Tokenizer {
    fn encode_offsets(&self, text: &str) -> Result<Offsets, ollaya_decision::Error> {
        self.0
            .encode_char_offsets(text, false)
            .map(|e| (e.get_ids().to_vec(), e.get_offsets().to_vec()))
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
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
            "{what} graph inputs {got:?} do not match the valen contract ({inputs:?} -> {output:?})"
        )));
    }
    Ok(())
}

fn array2<T>(rows: usize, cols: usize, values: Vec<T>) -> Result<Array2<T>, Error> {
    Array2::from_shape_vec((rows, cols), values).map_err(|e| Error::Model(e.to_string()))
}

impl Engine for Valen15Model {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        Valen15Model::run(self, state, questions, None)
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
        Valen15Model::run(self, state, &questions, image.as_ref())
    }
}

impl Valen15Model {
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
        if config.engine != "onnx" || config.layout != ollaya_decision::valen::LAYOUT {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/{}",
                config.engine,
                config.layout,
                ollaya_decision::valen::LAYOUT
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        config.valen.validate().map_err(|e| bad(e.to_string()))?;
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
            .ok_or_else(|| Error::Model("a valen model needs its vision graph".into()))?;
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

        Ok(Valen15Model {
            vision: Mutex::new(vision),
            decoder: Mutex::new(decoder),
            tokenizer: Tokenizer(tokenizer),
            chunk: config.chunk,
            image: config.image,
            layout: config.valen,
            calibration,
            device,
        })
    }

    /// Resize and cut an image into the vision graph's inputs. The size limits are the model's
    /// own (`image.min_pixels`, `image.max_pixels`): a larger image is resized down, as upstream.
    pub fn prepare_image(&self, image: &Rgb) -> Result<Valen15Image, Error> {
        let (resized, patches) = vision::preprocess(image, &self.image)?;
        let m = self.image.merge_size;
        let grid = (patches.grid_h / m, patches.grid_w / m);
        Ok(Valen15Image {
            resized,
            patches,
            grid,
        })
    }

    /// The request's sequence, with the image's visual tokens in the shared prefix.
    pub fn encode(
        &self,
        state: &Value,
        defs: &[(&str, &Value)],
        image: Option<&Rgb>,
    ) -> Result<Valen15Encoding, Error> {
        let image = image.map(|i| self.prepare_image(i)).transpose()?;
        let row =
            self.layout
                .encode(&self.tokenizer, state, defs, image.as_ref().map(|i| i.grid))?;
        Ok(Valen15Encoding { row, image })
    }

    /// Answer every question, about the image when there is one.
    pub fn run(
        &self,
        state: &Value,
        questions: &Questions,
        image: Option<&Rgb>,
    ) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        let encoding = self.encode(state, &defs, image)?;
        let logits = self.logits(&encoding)?;
        self.output(&encoding, &logits)
    }

    /// The vision tower's output for an image: [tokens, hidden].
    pub fn image_embeds(&self, image: &Valen15Image) -> Result<Array2<f32>, Error> {
        let p = &image.patches;
        let n = p.count();
        let tokens = image.grid.0 * image.grid.1;
        let mut session = self.vision.lock().expect("session mutex poisoned");
        let outputs = session.run(ort::inputs![
            "patches" => ort::value::Tensor::from_array(array2(n, p.dim, p.values.clone())?)?,
            "pos_idx" => ort::value::Tensor::from_array(array2(n, 4, p.pos_idx.clone())?)?,
            "pos_w" => ort::value::Tensor::from_array(array2(n, 4, p.pos_w.clone())?)?,
            "rot_ids" => ort::value::Tensor::from_array(array2(n, 2, p.rot_ids.clone())?)?,
        ])?;
        let out = outputs[VISION_OUTPUT]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("{VISION_OUTPUT}: {e}")))?;
        if out.shape() != [tokens, self.layout.hidden_size] {
            return Err(Error::Model(format!(
                "{VISION_OUTPUT} has shape {:?} for {tokens} visual tokens",
                out.shape(),
            )));
        }
        Ok(out.to_owned())
    }

    /// Each question's raw logits, in the head's candidate order (noul `[true, false]`).
    pub fn logits(&self, encoding: &Valen15Encoding) -> Result<Vec<Vec<f32>>, Error> {
        let row = &encoding.row;
        let n = row.ids.len();
        let seq = (n + 1).div_ceil(self.chunk) * self.chunk;
        let mut input_ids = Array2::<i64>::from_elem((1, seq), i64::from(self.layout.tokens.pad));
        let mut blocks = Array2::<i64>::from_elem((1, seq), PADDING);
        let mut position_ids = Array3::<i64>::zeros((3, 1, seq));
        for c in 0..n {
            input_ids[[0, c]] = i64::from(row.ids[c]);
            blocks[[0, c]] = row.blocks[c];
            for (a, axis) in row.positions.iter().enumerate() {
                position_ids[[a, 0, c]] = axis[c];
            }
        }
        let (image_embeds, image_pos) = match &encoding.image {
            Some(image) => {
                let embeds = self.image_embeds(image)?;
                let pos: Vec<i64> = (0..n)
                    .filter(|&c| row.ids[c] == self.layout.tokens.image)
                    .map(|c| c as i64)
                    .collect();
                if pos.len() != embeds.nrows() {
                    return Err(Error::Model(format!(
                        "{} image positions for {} visual tokens",
                        pos.len(),
                        embeds.nrows()
                    )));
                }
                (embeds, Array1::from_vec(pos))
            }
            // The sequence is shorter than `seq`, so its last position is padding.
            None => (
                Array2::<f32>::zeros((1, self.layout.hidden_size)),
                Array1::from_vec(vec![seq as i64 - 1]),
            ),
        };
        let qs = &row.questions;
        let count: usize = qs.iter().map(|q| q.candidates.len()).sum();
        let mut instruction_spans = Array2::<i64>::zeros((qs.len(), 2));
        let mut decision_pos = Array1::<i64>::zeros(qs.len());
        let mut candidate_spans = Array2::<i64>::zeros((count, 2));
        let mut candidate_question = Array1::<i64>::zeros(count);
        let mut o = 0;
        for (i, q) in qs.iter().enumerate() {
            instruction_spans[[i, 0]] = q.instruction.0 as i64;
            instruction_spans[[i, 1]] = q.instruction.1 as i64;
            decision_pos[i] = q.decision as i64;
            for &(a, b) in &q.candidates {
                candidate_spans[[o, 0]] = a as i64;
                candidate_spans[[o, 1]] = b as i64;
                candidate_question[o] = i as i64;
                o += 1;
            }
        }
        let mut session = self.decoder.lock().expect("session mutex poisoned");
        let options = crate::engine::run_options(self.device, seq, TOKEN_BUDGET)?;
        let outputs = session.run_with_options(
            ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "position_ids" => ort::value::Tensor::from_array(position_ids)?,
                "image_embeds" => ort::value::Tensor::from_array(image_embeds)?,
                "image_pos" => ort::value::Tensor::from_array(image_pos)?,
                "blocks" => ort::value::Tensor::from_array(blocks)?,
                "instruction_spans" => ort::value::Tensor::from_array(instruction_spans)?,
                "decision_pos" => ort::value::Tensor::from_array(decision_pos)?,
                "candidate_spans" => ort::value::Tensor::from_array(candidate_spans)?,
                "candidate_question" => ort::value::Tensor::from_array(candidate_question)?,
            ],
            &options,
        )?;
        let out = outputs[OUTPUT]
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix1>()
            .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
        if out.len() != count {
            return Err(Error::Model(format!(
                "{OUTPUT} has {} values for {count} candidates",
                out.len()
            )));
        }
        let mut it = out.iter().copied();
        Ok(qs
            .iter()
            .map(|q| it.by_ref().take(q.candidates.len()).collect())
            .collect())
    }

    /// Option logits per question, in the order answers use.
    pub fn output(&self, encoding: &Valen15Encoding, logits: &[Vec<f32>]) -> Result<Output, Error> {
        let row = &encoding.row;
        let questions = row
            .questions
            .iter()
            .zip(logits)
            .map(|(q, z)| {
                Ok(QuestionOutput {
                    logits: q.in_question_order(z)?,
                    act_logits: None,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Output {
            questions,
            input_tokens: row.ids.len(),
            state_tokens: row.prefix,
            state_truncated: false,
        })
    }
}
