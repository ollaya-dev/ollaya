# Our own decision model, in the browser and ternary

Status: in progress (started 2026-10-22). The name is not chosen yet; this document says "the model".

## Goal

A decision model published under Ollaya's own name that:

- runs entirely in a web page (no server, no install) and in Ollaya, with the same answers in both;
- is accurate enough to stand behind in public: measured on both held-out suites Ollaya already uses
  (typed-decisions test, Bespoke Labs' public benchmark), never trained on either;
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

  | Backbone | License | Params (with head) | Languages | Ternary + int8 embeddings |
  |---|---|---|---|---|
  | `jhu-clsp/ettin-encoder-68m` | MIT | ~74M | English | ~38 MB |
  | `jhu-clsp/ettin-encoder-150m` | MIT | ~163M | English | ~70 MB |
  | `jhu-clsp/mmBERT-small` | MIT | ~144M (98M of it embeddings) | 100+ | ~110 MB, less with a pruned vocabulary |

  The first runs train ettin-150m and mmBERT-small and keep the better trade-off; Turkish matters for the
  launch, which favours mmBERT if it is close.

## Data

Built by `train/data/` (see its README). Every source is permissively licensed and listed per source:

- `tasksource/tasksource-jev-typed-decisions` (only its sources with a permissive license),
  `tasksource/synthetic-typed-decisions`, `tasksource/procedural-typed-decisions` (Apache-2.0),
  `n4ze3m/typed-decisions-synth` (MIT), `fastino/fast-decisions` (Apache-2.0).
- **Held out:** every row derived from the 11 datasets of the public benchmark, and
  `LocalLLaMA/typed-decisions` (its train split is kept apart, for an optional fine-tuned variant whose
  typed-decisions score would then be marked as trained on the split).
- Rows must fit the 512-token sequence. The teacher labels them through Ollaya on an RTX 5090.

## Training

`train/train.py` (Python, build-time only, as `convert/` is):

1. **fp:** soft cross-entropy against the mixed target, plus the ranked probability score on score
   questions; a choice's options are permuted half the time so position never decides; the act head
   learns whether the top answer is right. AdamW, bf16, cosine schedule.
2. **Temperatures:** one per question type, fitted on a validation split, written into
   `rl_agent_config.json` as Laya expects.
3. **Ternary:** continue from the fp model with every weight matrix of the encoder layers and the head
   trained through absmean ternarization per output row (BitNet b1.58, straight-through estimator);
   optionally the token embeddings rounded to int8 per row. The saved weights are exactly ternary, so
   Ollaya runs them as ordinary fp32 weights and gets the same answers, and `train/ternary.py` packs them
   into 2 bits per weight for the browser and unpacks them bit for bit.

`train/serve.py` serves a checkpoint over `/v1/systemone` so `bench_typed` and `bench_public` score it during
development. Released numbers come from Ollaya, as for every other model.

## Browser runtime

`crates/ollaya-web` compiles `ollaya-decision` (question rules, the Laya layout, calibration, answers) and the
`tokenizers` crate to WebAssembly; `web/` runs the graph with onnxruntime-web (WASM, WebGPU when available)
and exposes `decide({state, questions})`, the same request and answer as `/api/decide`. Parity against the
native runtime is checked like every other engine, first on `largha:150m`, then on the model. Weights are
fetched from the model's Hugging Face repository, never from ollaya.dev.

## Release

- Weights in a Hugging Face repository under Ollaya's organization, Apache-2.0, with the base encoder's MIT
  notice and the data sources' attributions (CC-BY sources are credited).
- Ollaya: a registry tag (fp32 and fp16 graphs, as for `largha`), a library page, parity and benchmark rows in
  `results/`.
- Web: an npm package and a demo page on ollaya.dev; install guide: `ollaya pull <name>`, or
  `npm install` plus three lines of JavaScript, or one `<script type="module">` tag.

## Milestones

1. Browser runtime with `largha:150m`, parity in Node and Chrome.
2. Data and teacher labels.
3. fp students (ettin-150m, mmBERT-small): benchmark, pick one.
4. Ternary student: accuracy loss against the fp one, packed size.
5. Name, Hugging Face repository, registry, demo page, install guide; release with the next batched version.

## Risks

- Score questions are the encoders' weak spot (Laya models score 0.09–0.35 on the public benchmark's score
  subsets). The teacher's distributions and the RPS term target this; if it stays weak, the page says so.
- 512 tokens is short for long agent traces; the model is for messages, tickets and small JSON states.
- Ternary training may cost accuracy at 68M–150M parameters; the fp model ships either way.
