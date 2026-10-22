# Our own decision model, in the browser and ternary

Status: trained and checked, not published (2026-10-23). The name is not chosen yet; this document says "the model".

## Goal

A decision model published under Ollaya's own name that:

- runs entirely in a web page (no server, no install) and in Ollaya, with the same answers in both;
- is accurate enough to stand behind in public: measured on both held-out suites Ollaya already uses
  (typed-decisions test, Bespoke Labs' public benchmark), never trained on their test data (the in-domain
  variant trains on the benchmark datasets' train splits and says so next to every public score);
- comes in a ternary version (weights in {-g, 0, +g}) small enough to load on a phone.

## Why this design

Ollaya's measurements (`results/runs/`) show what each kind of model costs and gives:

| Kind | Example | Public benchmark (macro) | typed-decisions | Size |
|---|---|---|---|---|
| Distilled 4B decoder | `pplx-decider:flash` | 0.824 | 0.726 | 8 GB (fp16) |
| 0.8B decoders | `decider:0.8b`, `kev:0.8b` | 0.61–0.67 | 0.46–0.51 | 1.5 GB |
| Laya-network encoders | `laya:multilingual`, `laya:en` | 0.53–0.58 | 0.35–0.36 | 0.6–0.8 GB (fp16) |
| Laya network, small | `largha:150m` | 0.40 | (trained on the split) | 0.3 GB (fp16) |

A decoder good enough to be worth showing is too large for a web page; the encoders are small but
were trained on narrow data. The plan takes the encoder's shape and the decoder's knowledge:

- **Network: Laya's** (`laya` 0.3.7, Apache-2.0): the question, a `[MASK]` before every option and the
  state in one sequence; a two-layer head scores each option at its `[MASK]`. One forward pass per
  question, no generation. Ollaya already runs it (`laya-markers-v1`, parity on CPU, CUDA and Metal), so a
  checkpoint in Laya's format needs no new runtime code, only a manifest.
- **Teacher: `pplx-decider:flash`** (Apache-2.0, the most accurate and best calibrated model Ollaya runs),
  through Ollaya, on the training states. The student learns its full probability distributions, mixed
  with the gold label where the data has one.
- **Backbones** (all ModernBERT-architecture, which Laya and Ollaya's export already handle):

  | Backbone | License | Params (with head) | Ternary + int8 embeddings |
  |---|---|---|---|
  | `jhu-clsp/ettin-encoder-68m` | MIT | ~74M | ~38 MB |
  | `jhu-clsp/ettin-encoder-150m` | MIT | ~163M | ~70 MB |
  | `jhu-clsp/ettin-encoder-400m` | MIT | ~420M | ~145 MB |
  | `jhu-clsp/ettin-encoder-1b` | MIT | ~1.05B | ~330 MB |

  English only: accuracy comes first. The 150m trains first (about half an hour per 400k questions on an RTX
  4090), then the 400m; the 1b only if the 400m gains enough over the 150m to suggest more. The web page can
  offer the most accurate one that still loads in reasonable time, with the smaller one as a fallback.

## Data

Built by `train/data/` (see its README). Every source is permissively licensed and listed per source:

- `tasksource/tasksource-jev-typed-decisions` (only its sources with a permissive license),
  `tasksource/synthetic-typed-decisions`, `tasksource/procedural-typed-decisions` (Apache-2.0),
  `n4ze3m/typed-decisions-synth` (MIT), `fastino/fast-decisions` (Apache-2.0).
- **Held out:** every row derived from the 11 datasets of the public benchmark, and
  `LocalLLaMA/typed-decisions` (its train split is kept apart, for an optional fine-tuned variant whose
  typed-decisions score would then be marked as trained on the split).
- Rows must fit the 512-token sequence. The teacher labels them through Ollaya on an RTX 5090.
- **In-domain variant** (`train/data/indomain.py`): the *train* splits of ten of the public benchmark's
  datasets (MNLI, BoolQ, PAWS, VitaminC, civil_comments, Aegis 2.0, HelpSteer2, PubMedQA artificial,
  SQuAD 2.0, MASSIVE en-US), up to 20k rows each, asked exactly as the benchmark asks them (nimble's own
  source modules build the records). Any row sharing a text of 30+ characters with a benchmark record is
  dropped (195 rows, mostly BoolQ passages that recur across its splits). SummEval has no train split and
  stays out. A model trained with these rows reports its public-benchmark score as in-domain, next to the
  held-out model's zero-shot score, never in its place.

## Training

`train/train.py` (Python, build-time only, as `convert/` is):

1. **fp:** soft cross-entropy against the mixed target, plus the ranked probability score on score
   questions; a choice's options are permuted half the time so position never decides; the act head
   learns whether the top answer is right. AdamW, bf16, cosine schedule.
2. **Temperatures:** one per question type, fitted on a validation split, written into
   `rl_agent_config.json` as Laya expects.
3. **Ternary:** continue from the fp model with every weight matrix of the encoder layers and the head
   trained through absmean ternarization per output row (BitNet b1.58, straight-through estimator), the
   token embeddings at 4 bits per row, distilled from the fp model (logits, attention relations, every
   layer's hidden states) with a warm-up that blends fp into quantized weights. The saved weights are
   exactly ternary, so Ollaya runs them as ordinary fp32 weights and gets the same answers, and
   `train/webgraph.py` keeps them at 2 bits in the browser's graph.

`train/serve.py` serves a checkpoint over `/v1/systemone` so `bench_typed` and `bench_public` score it during
development. Released numbers come from Ollaya, as for every other model.

## Results

All rows are one epoch over the training set; `val` is accuracy on a held-out slice of the training data, the
other two are Ollaya's held-out suites (typed-decisions test, 400 rows; the public benchmark, 3,880 records,
macro accuracy). **Zero-shot** models never saw any of the public benchmark's datasets. **In-domain** models
also trained on the train splits of ten of its eleven datasets (see Data), so their public score is in-domain and
is never quoted without that word. Scores are from `train/eval.sh` (the PyTorch reference served by `serve.py`);
Ollaya's runtime gives the same answers (parity below).

| Run | Backbone | Weights | Data | val | typed | public |
|---|---|---|---|---:|---:|---:|
| fp150-gold | 150m | fp | gold only | 0.749 | 0.527 | 0.564 zero-shot |
| fp150-T | 150m | fp | gold + teacher | 0.765 | 0.541 | 0.594 zero-shot |
| fp150-id-T | 150m | fp | + in-domain | 0.765 | 0.544 | 0.705 in-domain |
| fp68-id-T | 68m | fp | + in-domain | 0.731 | 0.476 | 0.697 in-domain |
| fp400-id-T | 400m | fp | + in-domain | **0.811** | **0.573** | **0.749** in-domain |
| t150-T3w | 150m | ternary | gold + teacher | 0.718 | 0.502 | 0.478 zero-shot |
| t150-id-T3w | 150m | ternary | + in-domain | 0.736 | 0.531 | 0.688 in-domain |
| t68-id-T3w | 68m | ternary | + in-domain | 0.705 | 0.448 | 0.664 in-domain |

For reference (Ollaya bench): `pplx-decider:flash` (the teacher, 4B) 0.726 typed and 0.824 public;
`decider:0.8b` 0.46–0.51 typed and 0.669 public; `laya:typed-decisions` 0.594 public; `laya:en` 0.532 public.

- The teacher's soft labels help (+0.016 val, +0.03 zero-shot public over gold only); the in-domain rows lift the
  public score by 0.11 and leave typed-decisions where it was.
- 150m to 400m gains 0.046 val, 0.029 typed and 0.044 public.
- Ternary (recipe T3w: QAT from the fp model, logit + relation + every-layer hidden-state distillation from it,
  quantization warm-up over 30% of the steps, 4-bit embeddings) keeps 96% of the fp model's val and 98% of its
  typed and in-domain public scores at 150m.

## Browser runtime

`crates/ollaya-web` compiles `ollaya-decision` (question rules, the Laya layout, calibration, answers) and the
`tokenizers` crate to WebAssembly; `web/` runs the graph with onnxruntime-web (WASM, WebGPU when available)
and exposes `decide({state, questions})`, the same request and answer as `/api/decide`.

A ternary model gets its own graph for the browser (`train/webgraph.py`): every ternary MatMul becomes
`MatMulNBits` with 2-bit codes and the row's scale, the int4 token embeddings `GatherBlockQuantized`. Both
dequantize to exactly the float32 weights. Unpacking the weights to float32 instead cost 2.5 GB of resident
memory for the 150m (onnxruntime-web holds several copies while it loads), which no phone has to spare.

| 150m ternary, Node 26, 4 WASM threads | float32 graph + weights | 2-bit graph |
|---|---:|---:|
| Download | 57 MB (packed) | 63 MB, one file, + 3.6 MB tokenizer |
| Resident memory after load | 2.5 GB | 0.5 GB |
| Load | 1.2 s | 0.3 s |
| Five requests | 1.6 s | 1.9 s |

Parity, against the native runtime on the CPU (`web/parity.mjs`, the same 391 questions as the native goldens):
decisions 100% the same, calibrated probabilities within 2e-6 (fp150-id-T, t150-id-T3w float32 and 2-bit).

The model loads straight from its Hugging Face repository: `load(fromHub('<org>/<repo>'))`.

## Release

- Weights in a Hugging Face repository under Ollaya's organization, Apache-2.0, with the base encoder's MIT
  notice and the data sources' attributions (CC-BY sources are credited).
- Ollaya: a registry tag (fp32 and fp16 graphs, as for `largha`), a library page, parity and benchmark rows in
  `results/`.
- Hugging Face repository (`train/hub.py` stages it): the Laya checkpoint (model.safetensors, rl_agent_config.json,
  encoder/, tokenizer/) that `laya` and Ollaya's registry read; for a ternary model also onnx/model.onnx (the 2-bit
  graph), tokenizer.json, decision.json and calibration.json, which the browser reads; README (model card),
  LICENSE, NOTICE (Ettin's MIT notice) and DATA_SOURCES.tsv (every source and its license).
- Web: an npm package and a demo page on ollaya.dev (`/try`); install guide: `ollaya pull <name>`, or
  `npm install` plus three lines of JavaScript, or one `<script type="module">` tag.

## Milestones

1. Browser runtime with `largha:150m`, parity in Node and Chrome.
2. Data and teacher labels.
3. fp students (ettin-150m, then 400m, maybe 1b): benchmark, pick one.
4. Ternary student: accuracy loss against the fp one, packed size.
5. Name, Hugging Face repository, registry, demo page, install guide; release with the next batched version.

## Risks

- Score questions are the encoders' weak spot (Laya models score 0.09–0.35 on the public benchmark's score
  subsets). The teacher's distributions and the RPS term target this; if it stays weak, the page says so.
- 512 tokens is short for long agent traces; the model is for messages, tickets and small JSON states.
- Ternary training may cost accuracy at 68M–150M parameters; the fp model ships either way.
