# decision (`decision-endpoint-v1`)

**Decision 1.0** by the vLLM Semantic Router contributors (Hugging Face org
[llm-semantic-router](https://huggingface.co/llm-semantic-router), Apache-2.0) is a family of decision
models built from two parts:

- a **fully fine-tuned Qwen3.5 text backbone** (the hybrid Gated-DeltaNet + gated-attention model,
  without its vision tower and LM head), stored in BF16;
- a 10-tensor **endpoint head** in FP32 (`decision_head.safetensors`). It reads the hidden state at
  the last token of every option segment (the candidate endpoints) and at the last token of the
  row (the global query), and scores each candidate against the query.

They answer TypeSafe-style `choice`, `noul` and `score` questions with one forward pass per question
and never generate text. Each model repo ships its own inference code (`decision_model.py`, identical
in every repo, plus a typed request adapter); this document is the spec the Rust port implements.

| Model | Base (license) | Params | Checkpoint | Temperature | Status in Ollaya |
|---|---|---|---|---|---|
| `Decision-1.0-Eos-0.8B` | Qwen/Qwen3.5-0.8B @ `2fc06364` (Apache-2.0) | 753M (752.4M body + 1.05M head) | 1.5 GB BF16, 1 file | 1.0389 | **converted, ONNX** (`decision:eos`) |
| `Decision-1.0-Nox-4B` | Qwen/Qwen3.5-4B @ `851bf6e8` | 4.21B (+ 2.6M head) | 8.4 GB BF16, 3 shards | 1.3231 | see [Larger models](#larger-models) |
| `Decision-1.0-Lux-9B` | Qwen/Qwen3.5-9B @ `c2022362` | 7.94B (+ 4.2M head) | 15.9 GB BF16, 4 shards | 2.0054 | see [Larger models](#larger-models) |

The same org also publishes `Decision-1.0-Sol-2B` (same code, T 1.3004) and two 0.6B encoders
(Kai, Lex) that use a different architecture; they are out of scope here.

**Why these.** On Decision Index 0.2 (balanced skill, Jev 51.67) Eos scores 17.49 (ECE 0.083), Nox
31.06 (0.140) and Lux 38.98 (0.076). Lux is the strongest open model at 9B or below on that index.

## Pinned sources

The repos were 3 days old when this was written and change often (Lux's weights were replaced
three times on 2026-09-22), so every file is pinned to a commit.

| Model | Repo @ commit | Weights (sha256 = HF LFS oid) |
|---|---|---|
| Eos | `llm-semantic-router/Decision-1.0-Eos-0.8B` @ `3c2d632609ceb66f3a13bbc5f77f3ab8cdeebcdd` | `backbone/model.safetensors` 1,504,820,888 B `613f491d…8abd`; `decision_head.safetensors` 4,213,584 B `75f4ee8b…72f6` |
| Nox | `llm-semantic-router/Decision-1.0-Nox-4B` @ `0bb833504965c0eabdb9630b7bbd385cb2fe5cd4` | `backbone/model-0000{1,2,3}-of-00003.safetensors` (`5ebccc39…`, `990e2e79…`, `c47859a3…`); head `9cb6f639…89de` |
| Lux | `llm-semantic-router/Decision-1.0-Lux-9B` @ `bd45a30aee8c84032791c245c70f86dee5389cc8` | `backbone/model-0000{1,2,3,4}-of-00004.safetensors` (`38b8c6b3…`, `2b568637…`, `ff152c4b…`, `99d7d3bf…`); head `f810788b…137f` |

All three share `tokenizer.json` sha256 `06b95093…e523`, the same file as `decider` and `kev`.

## Recommended engine: ONNX (single forward)

- **Why ONNX.** The head is not the LM head (the checkpoint has none), so llama.cpp could not produce
  these logits without custom code. A decision is one causal row per question plus a small head,
  which is exactly what the decider/kev ONNX path already runs.
- **Export.** `convert/ollaya_convert/families/decision/export.py` reuses `llm_common/qwen35.py` (the
  scan-based Gated-DeltaNet export, see [decider.md](decider.md)) for the backbone and recomputes the
  head from the author's own `CandidateHead` modules. The eager graph is checked against the author's
  forward before export.
- **Weights.** Every initializer references the author's files by byte offset: the backbone
  safetensors (BF16, one `Cast` to float32 each, folded by ONNX Runtime at load) and
  `decision_head.safetensors` (F32). Nothing is re-hosted.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph only) | derived, hosted by Ollaya |
| backbone weights, BF16 | the model repo's `backbone/*.safetensors` at the pinned commit |
| head weights, F32 | the model repo's `decision_head.safetensors` |
| `tokenizer.json` | the model repo's, used as-is |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya (temperature copied from `runtime.json` for Eos, `temperature.json` for Nox and Lux) |
| license | Apache-2.0 (the repo's `LICENSE`); the Qwen3.5 base is Apache-2.0 (`QWEN-LICENSE`) |

**Tokenizer.** The author loads the tokenizer with transformers 5.17 `AutoTokenizer`. Its
`tokenizer_config.json` carries a `pretokenize_regex` with `[\p{L}\p{M}]+`, but transformers ignores
it: the loaded pre-tokenizer is the `tokenizer.json` one (`\p{L}+`), verified token for token on
Hindi, Thai, emoji, control-token and prompt strings. So the plain `tokenizers` crate reproduces
upstream exactly.

## Request → rows

Source: the author's `question_row` (Eos `decision/types.py`; Nox and Lux `code/decision_api.py`) and
`decision_model.py` `segments` + `encode` (prompt version
`structured-segmented-candidate-endpoints-global-query-v2`). Port: `families/decision/layout.py`
(Python) and `ollaya_decision::decision` (Rust).

### Validation

A failing question rejects the whole request, as upstream does (it encodes every row before any
forward pass).

- `questions` is a non-empty object; each question is an object with a `type` and an
  `instructions` key (any JSON value, including `null`).
- **choice.** `criteria` is an **object** of **2..255** entries. A list is rejected (upstream
  requires a mapping), and so is a single option.
- **noul.** `criteria` is `null`/absent or an object whose keys are only `"true"` and `"false"`,
  spelled exactly. Any other key (`"True"`, `"maybe"`) is rejected.
- **score.** `criteria` is a **list** of **2..10** levels. A legend object is rejected.
- A row longer than **16,384** tokens rejects the request. Nothing is ever truncated: not the state,
  not the options.

### Candidates per type

| Type | Candidates `{"key", "description"}`, in order |
|---|---|
| choice | `(key, value)` in key order. A `null` value stays `null` (Eos, Lux) or becomes the key (Nox 1.3.2, `choice_null_description: "key"`). |
| noul | `("false", criteria.get("false", "The answer to the question is no."))`, `("true", criteria.get("true", "The answer to the question is yes."))`. A key given as `null` keeps `null`; only a missing key takes the default. |
| score | `(str(i), level_i)` for each level; the level can be any JSON value. |

### Text and tokens

```
payload(v)   = v if v is a string else canonical(v)
canonical(v) = json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

prefix = "Context:\n" + payload(state) + "\n\nTask type: " + type + "\nQuestion:\n"
         + payload(instructions) + "\nOptions:"
option = "\n<option>\n" + canonical({"key": key, "description": description}) + "\n</option>"
suffix = "\n\nSelect the single option best supported by the context and instructions.\nDecision:"

row         = enc(prefix) + enc(option_1) + ... + enc(option_k) + enc(suffix)
cand_pos[j] = index of the last token of enc(option_j)
query_pos   = len(row) - 1
```

- `enc` is `tokenizer.json` with `add_special_tokens=False`. Each segment is tokenized on its own, so
  no BPE merge crosses a boundary. The state is part of the prefix, so it is tokenized once per
  question (upstream does the same; there is no shared-state cache).
- Special-token text in user input (`<|im_start|>`) becomes the special token, as upstream does. It
  is not escaped.
- `canonical` sorts object keys in code-point order and writes floats as Python's `repr` (`1e-05`,
  `1.0`). The Rust port is `pyjson::dumps_canonical`.

### Option logits and calibration

- **All types.** The option logits are `logits[row, :k]`: choice in key order, noul
  `[false, true]`, score in level order.
- **Calibration.** `softmax(logits / T)` with the released temperature, the same for every type:
  Eos 1.0389139156246665 (`runtime.json`), Nox 1.3231350559653137 and Lux 2.0054410339959294
  (`temperature.json`, per-type entries that are all equal). `calibration.json` stores
  `[T_choice, T_score, T_noul]`.
- **Answers.** The author's `typed_answer` is TypeSafe's shape: `noul` = P(true); choice = argmax
  key; score = the expected level index `Σ i·p_i` with the legend; confidence
  `(K·max p − 1)/(K − 1)`. Ollaya's `/v1/systemone` renders the same fields, rounded to 4 decimals
  (the author does not round).

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question; **`seq` a multiple of 64**; right-pad with any id (`pad` = 248044) |
| `query_pos` | int64 | `[rows]` | index of the row's last token |
| `cand_pos` | int64 | `[rows, k]` | index of each option segment's last token; pad short rows with 0 |
| `logits` | float32 | `[rows, k]` | raw candidate logits (temperature not applied); use the first `k_row` |

The head, from the author's `CandidateHead` (all fp32):

```
c = LayerNorm_cand(h[cand_pos]);  q = LayerNorm_query(h[query_pos])
logits = (W_key c · W_query q) / sqrt(256) + w_scalar · gelu(W_cmlp c + b_cmlp + W_qmlp q)
```

- **Positions and masking.** Positions are implicit (`0..seq-1`). There is no mask input: every
  layer is causal, so right padding never reaches a read position. Upstream pads to a multiple of 32
  with an attention mask; the outputs at real positions are the same.
- **Precision.** Opset 20; fp32 compute, BF16 backbone weights cast at load. Upstream serves the
  backbone under BF16 autocast on AMD gfx942; the reference here is the same code in fp32.

## Measured parity

`families/decision/parity.py`: ONNX Runtime CPU on the weightless graph against the author's code in
fp32 (`DecisionModel.from_checkpoint(dtype=float32)`, no autocast, CUDA with TF32 off and SDPA on its
exact math kernel, physical batches of 8 as upstream).

**Layout port.** The Python port reproduces the author's token rows, candidate and query positions
on **all 481 requests** (77 Laya edge cases, 4 decoder cases, 400 typed-decisions rows; 2,292 rows)
for all three models, and rejects the same 17 requests.

**Eos** (`convert/out/decision-eos-0.8b/parity.json`): ONNX Runtime 1.30 CPU, 131 requests (77 Laya
edge + 4 decoder edge + 50 typed-decisions rows).

| | value |
|---|---|
| questions / rows | 616 / 616; 17 requests rejected by both sides (their accepted questions scored one by one) |
| token rows, query and candidate positions identical | 616 / 616 |
| argmax agreement, ONNX vs fp32 / vs the author's answers | **100 % / 100 %** |
| max \|Δ logit\| | 3.8e-5 choice, 1.3e-5 noul, 1.3e-5 score (p99 1.9e-5) |
| max \|Δ probability\| (T = 1.0389) | 3.0e-6 choice, 1.8e-6 noul, 2.5e-6 score |
| contract (logits / T → softmax) vs the author's `typed_answer` | ≤ 8.4e-8 |

- **Graph.** 7.7 MB. 330 initializers reference the checkpoint (320 BF16 backbone tensors through a
  `Cast`, 10 F32 head tensors); 3 small masks (100 KB) stay inline; no checkpoint tensor is unused.
  The eager export module matches the author's forward to 4.8e-6.
- **Goldens.** `convert/out/goldens-decision-eos-0.8b.jsonl`, written by `families/decision/goldens.py`
  (the 101 edge and typed-decisions requests of the other families, 20 typed-decisions rows): per
  request the exact rows, query and candidate positions, fp32 option logits, probabilities and the
  author's answers.

## Rust runtime parity (measured)

`crates/ollaya-runner/examples/parity_decision.rs` against the goldens. It checks the rejections
(each question of a rejected request on its own too), then the token rows, query and candidate
positions, then the option logits (tolerance 1e-3, as for the other decoder families), the decisions
and the TypeSafe answers against the author's `typed_answer` (choice, noul, expected score,
confidence, legend and probabilities; Ollaya rounds to 4 decimals and the author does not, so the
answers may differ by half a step).

```sh
cargo run --release -p ollaya-runner --example parity_decision -- convert/out/decision-eos-0.8b convert/out/goldens-decision-eos-0.8b.jsonl cpu
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_decision -- convert/out/decision-eos-0.8b convert/out/goldens-decision-eos-0.8b.jsonl cuda
```

**Eos** on choso-wsl (24 cores; RTX 4090, CUDA 13), `ort` 2.0.0-rc.13 with ONNX Runtime 1.28: 117
records (17 requests the author rejects, their 16 `#valid` subsets), 466 questions, 100,474 tokens.

| | CPU | CUDA |
|---|---|---|
| rejections as the author (each question on its own) | 0 mismatches | 0 mismatches |
| token rows, query and candidate positions identical | 466 / 466 | 466 / 466 |
| max \|Δ option logit\| | 3.6e-5 | 1.6e-5 |
| decisions agree | **100 %** | **100 %** |
| max \|Δ probability\| (T = 1.0389) | 2.4e-6 (p99 2.1e-6) | 2.9e-6 (p99 2.1e-6) |
| TypeSafe answers vs the author's `typed_answer` | 0 differ, max 5.1e-5 (rounding) | 0 differ, max 5.1e-5 |

The graph has two `Gemm`s (the head's query-side projections `query` and `query_mlp`), both fed by
the query `LayerNormalization`, not by a `Transpose`, so the ORT 1.28 `GemmTransposeFusion` bug that kev works
around ([kev.md](kev.md)) cannot apply; the engine uses the decider session options unchanged.

## Limits

- **Options.** Choice 2..255 (as an object), score 2..10 levels, noul. TypeSafe also caps score at
  10.
- **Context.** Up to 16,384 tokens per question row, the state included. The author trained on
  inputs of up to 9,129 tokens and does not claim long-context quality.
- **Cost.** Every question re-reads the whole state; there is no shared-state cache.
- **Candidate order.** Endpoints are causal, so a candidate can see the ones before it; the author
  notes that candidate-order invariance is not guaranteed.
- **Calibration.** Probabilities can be overconfident (the author says so for Eos).
- **Runtime qualification.** Upstream qualified BF16 inference on AMD gfx942 only; its CPU path is
  "unsupported". Ollaya's fp32 graph is checked against the author's code in fp32, not against the
  BF16 GPU outputs.

## Larger models

Nox and Lux use the same code, prompt, tokenizer and layout as Eos; only the dimensions, the
temperature and (Nox) the null choice description differ. Their graphs, `decision.json` and
`calibration.json` come from the same exporter. What stops them today is memory: ONNX Runtime folds
every BF16 → fp32 `Cast` at load, so the weights sit in memory in fp32, next to the BF16 originals.

| | Eos | Nox | Lux |
|---|---|---|---|
| backbone + head parameters | 0.75B | 4.21B | 7.94B |
| BF16 checkpoint | 1.5 GB | 8.4 GB | 15.9 GB |
| fp32 weights after the casts fold | 3.0 GB | 16.8 GB | 31.8 GB |
| graph | 7.7 MB | 10.6 MB | 10.6 MB |

**Nox (measured, choso-wsl).** Exported, goldens built (`convert/out/goldens-decision-nox-4b.jsonl`,
same 117 records), and parity passes everywhere it runs; it is left out of the library for memory:

| | ONNX Runtime 1.30 Python, CPU | Rust, CPU | Rust, CUDA (RTX 4090) |
|---|---|---|---|
| rejections, token rows and positions | (from the goldens) | 0 mismatches, 466 / 466 | 0 mismatches, 466 / 466 |
| max \|Δ option logit\| | 1.7e-4 | 1.7e-4 | 9.1e-5 |
| decisions agree | 100 % | **100 %** | **100 %** |
| max \|Δ probability\| (T = 1.3231) | 6.8e-6 | 5.5e-6 (p99 3.5e-6) | 9.4e-6 (p99 4.5e-6) |
| TypeSafe answers vs the author | | 0 differ, max 5.7e-5 | 0 differ, max 5.7e-5 |
| peak memory | 28 GB RSS | 32 GB RSS | 24.0 of 24.5 GB GPU, 29 GB RSS |
| 5-question request, ~1,050 tokens | | p50 10.0 s | p50 0.56 s, p95 1.8 s, max 93 s |

On the 4090 the fp32 weights and the activations fill the GPU, and WSL pages GPU memory to the host
rather than failing, hence the 93 s outliers. The 24 GB Mac mini cannot load it at all.

**Lux (not verified).** Lux's goldens did not finish: with 16 GPU layers the split reference overflowed
the 24 GB GPU and WSL paged it, and a rerun with 10 GPU layers was still queued. Lux has no
manifest and no library entry.

**Exports.** Nox is exported like Eos (the fp32 model in memory, every initializer compared with the
checkpoint by value). Lux in fp32 does not fit the export job's memory, so `--empty-weights` builds the
same modules with uninitialized storage and maps the initializers by name and shape
(`weightless_sharded`, `verify=False`): every parameter must map, every checkpoint tensor must be
used, and only buffers (the Scan masks and rotary frequencies) stay inline. On Eos and Nox this path
gives the value-checked graph exactly, apart from the exporter's debug stack traces.

**Goldens.** Built like Eos's, from the author's code in fp32. Nox and Lux do not fit the 24 GB GPU
in fp32 together with the attention transients, so the reference runs the first decoder layers on the
GPU and the rest on the CPU, still in fp32 (`ref.split`, `DECISION_REF_GPU_LAYERS`; on Eos it
reproduces the plain reference's logits to 8.6e-6). Under WSL a CUDA run that overflows the GPU does
not fail: the driver pages GPU memory to the host and the run crawls, which is why the split is sized
with room to spare.

## Attribution

Decision 1.0 is by the vLLM Semantic Router contributors (https://huggingface.co/llm-semantic-router),
released under Apache-2.0 and built on Qwen3.5 by the Qwen team (Apache-2.0). The reference loads the
inference code shipped in each model repo at the pinned commit.
