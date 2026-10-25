# decision2 (`decision2-endpoint-v1`)

**Decision 2.0** by the vLLM Semantic Router contributors (Hugging Face org
[vllm-sr](https://huggingface.co/vllm-sr), Apache-2.0) is the second generation of
[Decision 1.0](decision.md): the same two parts and the same graph, retrained.

- a **fully fine-tuned text backbone**: Qwen3 for Kai (a dense causal decoder; Decision 1.0's Kai
  was an encoder), Qwen3.5 (Gated-DeltaNet + gated attention, without the vision tower) for the
  others. Linear weights are BF16; embeddings, norms, the DeltaNet's `A_log`, `dt_bias` and conv
  filters are F32;
- the 10-tensor **endpoint head** of Decision 1.0 in F32 (`decision_head.safetensors`), reading the
  last token of every option segment and of the row.

Each repo ships its runtime (`decision2/`, with its training code vendored under
`decision2/_vendor/dev2model`). This document is the spec the port implements; it runs on
Decision 1.0's ONNX graph and Rust engine, with its own layout.

| Tag | Repo @ commit | Base | Params | Context | Checkpoint |
|---|---|---|---|---|---|
| `decision2:kai` | `vllm-sr/Decision-2.0-Kai-0.6B` @ `cc7f30cd` | Qwen/Qwen3-0.6B-Base @ `da87bfb6` | 0.60B | 8,192 | 1.5 GB, 1 file |
| `decision2:eos` | `vllm-sr/Decision-2.0-Eos-0.8B` @ `52cad37e` | Decision 1.0 Eos (Qwen3.5-0.8B @ `2fc06364`) | 0.75B | 16,384 | 2.0 GB, 1 file |
| `decision2:sol` | `vllm-sr/Decision-2.0-Sol-2B` @ `df30237f` | Decision 1.0 Sol (Qwen3.5-2B @ `15852e8c`) | 1.88B | 16,384 | 4.8 GB, 2 shards |
| `decision2:nox` | `vllm-sr/Decision-2.0-Nox-4B` @ `ae269ecf` | Qwen/Qwen3.5-4B-Base @ `1001bb4d` (merged LoRA) | 4.21B | 16,384 | 9.7 GB, 5 shards |
| `decision2:lux` | `vllm-sr/Decision-2.0-Lux-9B` @ `1dcd1e7f` | Decision 1.0 Lux (Qwen3.5-9B @ `c2022362`) | 7.94B | 16,384 | 17.9 GB, 9 shards |

Every package is MODEL_MANIFEST profile `qwen-full` with `calibration: null`: probabilities are the
plain softmax of the option logits (temperature 1). The 27B Vega, the `share_context` mode and the
GGUF repos (which need the author's llama.cpp fork) are out of scope.

## Pinned sources

Full commits: Kai `cc7f30cda9e7c7b28c5f71ef33f01e09385f1c4b`, Eos
`52cad37ef21f598302fe964cbfa164739cb965d5`, Sol `df30237f9c1ced5765fdbb499dd1f2c7dca478ad`, Nox
`ae269ecf34a0b3a89026bc1afd9c995279212fad`, Lux `1dcd1e7f240eab37ac09a6d0b0f77654b59411f9`.

| Model | Weights (bytes, sha256 = HF LFS oid) | Head | Tokenizer |
|---|---|---|---|
| Kai | `backbone/model.safetensors` 1,503,429,208 `43369a2b…3b69` | 4,213,584 `508dfe38…409d` | own, `be756060…5506` |
| Eos | `backbone/model.safetensors` 2,014,377,424 `049651c3…5e5f` | 4,213,584 `1cfc3523…b394` | `06b95093…e523` |
| Sol | `backbone/model-0000{1,2}-of-00002.safetensors` (`2213a655…`, `ff7a564a…`) | 8,424,272 `0999f3c4…4e7a` | `06b95093…e523` |
| Nox | `backbone/model-0000{1..5}-of-00005.safetensors` (`43d4d5f9…`, `244d869a…`, `8add5d59…`, `e542793a…`, `f86e1c89…`) | 10,529,624 `5ebd2939…e590` | `06b95093…e523` |
| Lux | `backbone/model-0000{1..9}-of-00009.safetensors` (`03cd850d…`, `cad4b1e5…`, `6ac88bf9…`, `0e2f45de…`, `214acdbd…`, `05b290c5…`, `5e934511…`, `17476a39…`, `30952702…`) | 16,845,656 `47f7b4d7…d102` | `06b95093…e523` |

`06b95093…e523` is the Qwen3.5 file Decision 1.0, `decider` and `kev` use. Kai's is the Qwen3 one.

## Request → rows

Source: the author's `decision2/qwen.py` `QwenDecision.system_one` (with `share_context` off), then
`infer.question_to_row`, `decision_model.encode` and `collate` (prompt version
`decision2-segmented-options-global-query-v1`). Port: `families/decision2/layout.py` (Python) and
`ollaya_decision::decision` with `prompt_version` 2 (Rust).

The **text is Decision 1.0's** (prefix, option segments, suffix, `payload`, `canonical`; see
[decision.md](decision.md#text-and-tokens)). What changes is which requests are valid and which
options a question has.

### Validation

Upstream answers a failing question with an error and the others normally; Ollaya rejects the whole
request, as for every family (the goldens check both sides on each question alone too).

- `questions` is a non-empty object with non-empty ids.
- `state` is text, an object or an array (`null`, a number or a boolean is rejected; `""` is fine).
- `instructions` is non-empty text, an object or an array.
- **choice.** `criteria` is an object of 2..255 entries with non-empty keys; each description is
  text, an object, an array or `null`.
- **noul.** `criteria` is absent, `null` or an object whose keys are exactly `"true"` and/or
  `"false"`. Each given description is text, an object or an array (not `null`, not a number).
- **score.** `criteria` is a list of 2..10 levels, each text, an object or an array.
- A row longer than the model's limit (Kai 8,192, the others 16,384 tokens) rejects the request.
  Nothing is truncated.

### Options per type

| Type | Options `{"key", "description"}`, in order |
|---|---|
| choice | the caller's order |
| noul | both sides given: the caller's order. Otherwise `false` then `true`, a missing side described as `"No"` / `"Yes"` |
| score | `(str(i), level_i)` |

The graph reads the options in that order. The engine returns noul logits as `[false, true]`
whatever the order (`DecisionRow::true_first`), as every Ollaya answer expects.

### Score offsets (Kai)

Kai's `score_bias.json`, bound to the weights' sha256 in MODEL_MANIFEST, adds fixed per-level
offsets to the logits of five-level score questions before the softmax:
`[0.039188, 0.203049, 0.079362, -0.15162, -0.169979]`. `decision.json` carries them as
`score_bias: {"5": [...]}`, and the engine adds them in f64 (`DecisionRow::offsets`). No other
model has offsets.

### Answers

`product_answer` is TypeSafe's shape at temperature 1: `noul` = P(true), choice = argmax key, score
= the expected level with the legend. Its confidence is the normalized entropy `1 − H(p)/log K`;
Ollaya's `/v1/systemone` reports TypeSafe's `(K·max p − 1)/(K − 1)` for every model, so the parity
check compares the entropy confidence for this family.

## ONNX contract

Decision 1.0's ([decision.md](decision.md#onnx-contract)): `input_ids` [rows, seq], `query_pos`
[rows], `cand_pos` [rows, k] → `logits` [rows, k]. Rows are padded to `contract.seq_multiple`: 64
(one DeltaNet scan chunk) for the Qwen3.5 models, 8 for Kai, whose layers are all causal attention.

- **Export.** `families/decision2/export.py` builds `decision.export.DecisionGraph` with
  `llm_common/qwen35.py` or, for Kai, `llm_common/qwen3.py`, and checks the eager graph against the
  author's forward before exporting. Lux in fp32 does not fit the export job's memory, so it is
  exported with `--empty-weights` (initializers mapped by name and shape, as for Decision 1.0's Lux).
- **Weights.** Every parameter references the author's files by byte offset: BF16 tensors through
  one `Cast` to float32, F32 tensors directly. `decision.json` sets `weights_in_memory` from the
  BF16 bytes (`onnx_export.weights_in_memory`): above 6 GiB the casts are not folded at load, so the
  weights stay in BF16 in memory and widen one layer at a time, as for decider.

## Measured parity

**Goldens.** `families/decision2/goldens.py` runs the author's code in fp32 (TF32 off, SDPA on its
exact math kernel, each request's questions in one `collate` batch as `system_one` does) on 107
records: the shared edge and typed-decisions cases (20 rows) and 13 cases of its own (noul order and
defaults, Kai's five-level offsets, object and list levels and descriptions, every rejection rule, a
question over every limit, a 5,000-token state). For every record the Python port must reproduce the
author's token rows, positions and option order before the record is written. Lux in fp32 (32 GB)
does not fit the 32 GB GPU with its attention transients, so its reference runs the first 16 decoder
layers on the GPU and the rest on the CPU, still in fp32 (`DECISION_REF_GPU_LAYERS`).

**Runtime.** `crates/ollaya-runner/examples/parity_decision.rs` against the goldens, the same gate
as Decision 1.0 (rejections, rows and positions identical, option logits within 1e-3, the same
decision, the wire answers equal the author's up to Ollaya's 4-decimal rounding). Two of the
author's answer conventions differ from Ollaya's `/v1/systemone` by design, and the check reads them
in the author's form: the entropy confidence, and a non-text score level written into the legend as
canonical JSON text (Ollaya returns it verbatim).

hope (RTX 5090, CUDA 13; Threadripper 3960X), ONNX Runtime 1.28.2 from the GPU pack (`cuda-dynamic`):
132 cases (25 rejected by the author, each question on its own too), 463 questions.

| Tag | rejections, rows, positions | decisions agree | max \|Δ option logit\| CUDA / CPU | max \|Δ probability\| CUDA / CPU |
|---|---|---|---|---|
| `decision2:kai` | 0 mismatches | **100 %** | 3.0e-5 / 2.7e-5 | 1.5e-6 / 2.1e-6 |
| `decision2:eos` | 0 mismatches | **100 %** | 1.3e-5 / 2.0e-5 | 2.5e-6 / 4.8e-6 |
| `decision2:sol` | 0 mismatches | **100 %** | 1.0e-5 / 8.4e-6 | 2.4e-6 / 2.1e-6 |
| `decision2:nox` | 0 mismatches | **100 %** | 1.6e-5 / 1.1e-5 | 4.6e-6 / 2.6e-6 |
| `decision2:lux` | 0 mismatches | **100 %** | 4.2e-5 / 5.0e-5 (smoke) | 5.2e-6 / 2.9e-6 (smoke) |

On the CPU, Lux runs the `--smoke` cases (26 cases, 45 questions), as K2 asks of models of 9B and over.
The wire answers equal the author's on every question (max difference 5.1e-5, the rounding).
The records are in `results/runs/2026-10-25-parity-rtx5090-{cuda,cpu}-cob561.json`.

## Limits

- **Options.** Choice 2..255, score 2..10 levels, noul.
- **Context.** Kai 8,192 tokens per question, the others 16,384, the state included.
- **Cost.** Every question re-reads the whole state; `share_context` (one pass for all questions)
  is not ported.
- **Runtime qualification.** Upstream runs the backbone under BF16 autocast on a GPU; Ollaya's
  graph is checked against the author's code in fp32.

## Attribution

Decision 2.0 is by the vLLM Semantic Router contributors (https://huggingface.co/vllm-sr), released
under Apache-2.0 and built on Qwen3 and Qwen3.5 by the Qwen team (Apache-2.0). The reference loads
the runtime shipped in each model repo at the pinned commit.
