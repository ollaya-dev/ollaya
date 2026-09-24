# decider (`decider-slots-v1`)

Mapika's **decider** models are Qwen3.5 base models fine-tuned to answer typed questions from one
forward pass. The readout is **the model's own LM head restricted to option-label tokens**, read at an
`Answer: (` slot. There is no custom scalar head: the "head" is the tied embedding rows of the 255 label
tokens. Upstream serves them on `POST /v1/systemone` in TypeSafe's wire format.

| Model | Base | Params | Checkpoint | Upstream temperature | Status in Ollaya |
|---|---|---|---|---|---|
| `Mapika/decider-0.8b` (v1) | Qwen3.5-0.8B-Base | 0.75B | 1.5 GB BF16, 1 file | 1.03 | **converted, ONNX** |
| `Mapika/decider-2b` (v10) | Qwen3.5-2B-Base | 1.9B | 3.6 GB BF16, 1 file | 1.30 | **converted, ONNX** |
| `Mapika/decider-4b` (v1) | Qwen3.5-4B-Base | 4B | 8.4 GB BF16 | – | same exporter; not converted |
| `Mapika/decider-35b-a3b` (v1) | Qwen3.5-35B-A3B-Base (MoE) | 35B / 3B active | 65 GB BF16, 15 shards | 1.08 | not converted (see "Larger models") |

## Recommended engine: ONNX (single forward)

A decision is one prefill plus a 255-row matmul, with no generation, so option (a) fits exactly.

- **Export.** The Qwen3.5 backbone is a hybrid: 18 Gated-DeltaNet (linear-attention) layers and 6 gated
  full-attention layers. `transformers` cannot export it with a dynamic sequence axis, because its
  chunked delta rule loops over chunks in Python.
  `convert/ollaya_convert/families/llm_common/qwen35.py` recomputes the same function from the same HF
  submodules:
  - The inter-chunk recurrence runs as `torch._higher_order_ops.scan`, which becomes an ONNX `Scan` over
    64-token chunks.
  - The UT-transform triangular solve uses recursive block doubling: 6 masked 64x64 matmul levels,
    exact in real arithmetic.

  Against HF eager in fp32, the hidden states differ by at most 5e-5 (scale 38).
- **Why not llama.cpp.** The same readout is plain next-token logits, so a GGUF of these weights would
  also work with the `llm-logits` bias trick fed with this layout's token ids. But no GGUF is published,
  and a converted GGUF would be derived weights that Ollaya would have to host. ONNX references the
  author's safetensors byte for byte.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph, 8.0 MB; weights are external references) | derived, hosted by Ollaya |
| `model.safetensors` (BF16) | `Mapika/decider-0.8b@a0a01d6f8135298f400a8c856b355793012ae971/model.safetensors`, sha256 `6926f82e…563d` (matches HF LFS oid) |
| `tokenizer.json` | same repo and commit, `tokenizer.json`, used as-is (sha256 `06b95093…e523`) |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |
| license | Apache-2.0, stated in the model card. The repo has no LICENSE file; the Qwen3.5 base is Apache-2.0. |

For `decider-2b` the source is `Mapika/decider-2b@9839cc9d908be16c5988c0d041034b5fdf82c7a2` with the same file
names. Each export writes `files.json` with sizes and sha256 values.

- **Weights.** All 320 checkpoint tensors map to graph initializers (`weightless_sharded.py`: 320
  external, 320 `Cast` BF16→F32). Only masks and index constants stay inline, about 100 KB.
- **Load behavior.** ONNX Runtime folds the casts at session creation. On CPU the fp32 graph for the
  0.8B model loads in about 7 s and holds about 6.5 GB of RSS.
- **Tokenizer caveat.** The repo's `tokenizer.json` was re-saved by transformers 5.x. Its pre-tokenizer
  regex is `\p{L}+`, while the original Qwen3.5 file uses `[\p{L}\p{M}]+`. transformers 5.17's
  `Qwen2Tokenizer` uses `\p{L}+` whichever file it loads, and that is how decider was trained and is
  served. **Use the decider repo's file, not the base model's.** With the base file, Hindi, Thai and
  other combining-mark scripts tokenize differently (measured: 22 vs 36 tokens on the Hindi test state).

## Request → rows

These rules are upstream `decider/systemone.py` + `decider/prompt.py` (layout `state_first`,
`independent=True`, the default). Port: `families/decider/layout.py`.

### State text

- **String state.** Used verbatim.
- **Any other JSON.** `json.dumps(annotate_indices(state), ensure_ascii=False)`, with Python's default
  separators `", "` and `": "` (Python float repr, as in `pyjson.rs`).
- **`annotate_indices`.** Recursive: an array with at least 8 elements has each element rewritten:
  - a dict element `e` becomes `{"_index": i, **e}`: `_index` goes first; if `e` already has `_index`,
    `e`'s value wins but the key stays first;
  - any other element becomes `{"_index": i, "value": e}`.

  Shorter arrays and dicts are only recursed into.

### Question text and options

**Instructions.** A string is used verbatim; anything else is `json.dumps(v, ensure_ascii=False)`. This
differs from Laya, which renders structured instructions with ensure_ascii on. A missing or empty value
gets HTTP 422. A JSON `null` renders as the text `null`, as upstream does.

**Rendering per type** (`_txt(x)` is a string verbatim, otherwise `json.dumps(x, ensure_ascii=False)`):

| Type | Criteria rules | Option text |
|---|---|---|
| choice | dict of 2..255 entries; a list of strings becomes `{s: null}`, and duplicates collapse | `name` if the description is `null` or `""`, else `name: _txt(description)` |
| noul | object with keys exactly `"true"` / `"false"`, case-sensitive; other keys are ignored | `no` / `no: _txt(false)`, then `yes` / `yes: _txt(true)`. Wire order is `[false, true]`. |
| score | list of 2..10 levels, or a legend object `{"0": …}` sorted by `float(key)` | the level text `_txt(c)` |

Two upstream-compatible differences:

- **Non-string list items in choice criteria.** Upstream `str()`-renders them; Ollaya rejects them with 400.
- **Non-object noul criteria.** Upstream crashes with a 500; Ollaya returns 400.

**Isolated score levels.** The config sets `isolated_levels=true`, so each level becomes its own yes/no row:

```
question = "{instructions}\nProposed answer: {strip(level)}\nDoes the proposed answer fit?"
options  = ["no", "yes"]
strip    = re.sub(r"^\s*-?\d+\s*:\s*", "", level, count=1)    # Unicode \s and \d, no MULTILINE
```

### Row tokens

Tokenization uses `tokenizer.json` with `add_special_tokens=False`. There is no BOS, and added tokens in
the text are matched, as in upstream.

```
ctx      = enc("Context:\n" + state_text)[:32768]                  # shared by every row
<= 10 options (narrow):
  piece  = enc("\n\nQuestion: " + q + "\nOptions:" + "".join(f"\n({L}) {opt}") + "\nAnswer: (")   # one string
> 10 options (wide):
  piece  = enc("\n\nQuestion: " + q + "\nOptions:")
         + concat_j( enc("\n(") + [label_id[j]] + enc(") " + opt_j) )
         + enc("\nAnswer: (")
row      = ctx + piece;  slot = len(row) - 1
```

**Labels.** `L` for narrow rows is A..J. The wide table has 255 labels: A..Z, then the first 229
two-letter uppercase strings that are single tokens (AA, AB, …, JT). The table is stored in
`decision.json` (`labels.strings` / `labels.ids`), and the first 10 wide labels are A..J.

**Budgets.**
- The state is capped at 32,768 tokens, counted including `Context:\n`. Questions and options are never
  truncated.
- The model supports 262,144 positions. Ollaya should cap total row length by memory rather than
  truncate differently from upstream.

### Option logits (runner → server)

The graph returns `label_logits[row, 0:255]`.

| Type | Option logits returned by the runner | Calibration temperature |
|---|---|---|
| choice | `label_logits[row, :k]` | `t_choice` = upstream T |
| noul | `label_logits[row, :2]`, which is `[no, yes]` = `[false, true]` | `t_noul` = upstream T |
| score | `z_j = log_sigmoid((l_yes_j − l_no_j) / T)` from level row `j` | `t_score` = **1.0** |

- **Why this works for score.** The server computes `softmax(z)` = `p_yes_j / Σ p_yes`, which is
  upstream's normalised per-level fit exactly.
- **Where T lives.** `decision.json.isolated_row_temperature` holds T, so the score temperature is
  already inside `z`.

`calibration.json` stores `[T, 1.0, T]`: `[1.03, 1.0, 1.03]` for 0.8b and `[1.3, 1.0, 1.3]` for 2b.
Upstream fitted these on in-task data; decider ships one temperature per model.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per scoring row; **`seq` a multiple of 64**; right-pad with any id (`pad` = 248044) |
| `slot_pos` | int64 | `[rows]` | index of the row's `(` slot token |
| `label_logits` | float32 | `[rows, 255]` | `h[slot] · E[label_ids]ᵀ` (E = tied embedding) |

Additional properties:

- Positions are implicit (`0..seq-1`) and there is no attention-mask input: every layer is causal, so
  right padding never reaches the slot.
- The graph uses opset 20. Weights are BF16 in the checkpoint and computed in fp32.
- **Padding to 64.** Each 64-token chunk is one `Scan` step. Padding in-graph would need `T % 64` on a
  dynamic axis, which torch.export cannot keep symbolic.

## Measured parity

`families/decider/parity.py`: ONNX Runtime CPU on the weightless graph vs upstream `decider` in fp32
(CUDA, TF32 off).

- **Token rows.** The port reproduces upstream token ids and slots on **all 481 requests**: 77 Laya edge
  cases, 4 decoder edge cases and all 400 typed-decisions rows. It also rejects the same requests.
- **Numbers.** Reported per model under "Parity results" below.

## Limits

- **Options.** Choice 2..255 (1 is rejected); score 2..10 levels; noul.
- **Context.** State ≤ 32,768 tokens. A single row can therefore reach tens of thousands of tokens, and
  CPU cost grows linearly with it.
- **Language.** English only, per the model card. Other scripts tokenize but are out of distribution.
- **Special tokens in the state.** Text like `<|im_start|>` becomes the special token, as upstream does.
  It is not escaped.
- **Response extras.** Upstream adds `certainty`, `level_fit` and `fit_mass`; they are not part of the
  TypeSafe wire and Ollaya does not emit them.
- **Unsupported upstream options.**
  - The schema-first cached layout (`temperature_schema_first`), which upstream turns on explicitly.
  - Packed rows (`independent=false`).
  - Listwise score (`"isolated": false`).
  - `neutralize_none`, which is false in both configs.

## Larger models

- **decider-4b.** Exports with the same code. Its fp32 graph is about 16 GB at runtime, so it needs a
  fp16/bf16 compute variant first (not validated here).
- **decider-35b-a3b.** 15 BF16 shards and a MoE backbone. An fp32 ONNX graph is impractical. The best
  path is llama.cpp with a `qwen35moe` GGUF, built from the upstream shards and fed this layout's token
  ids with the label-logit read. That GGUF is derived (conversion rewrites the tensors), so unless Mapika
  publishes one it conflicts with "never re-host weights". The NVFP4 repo is Blackwell-only (vLLM or
  TensorRT-LLM).

## Upstream benchmark numbers (model cards)

| | decider-0.8b | decider-2b v10 |
|---|---|---|
| 69 in-task tasks, acc / ECE | 0.776 / 0.032 | 0.805 / 0.037 (v8 recipe: 0.809 / 0.030) |
| 24–28 held-out tasks, acc / ECE | 0.707 / 0.096 | 0.755 / 0.084 (rebuilt set) |
| JevBench public items, easy / standard / hard | – | 1.000 / 0.889 / 0.459 |
| Bespoke public suite, macro | – | 0.704 |
| OpenJev 5,252 rows, acc | – | 63.3 % |

## Attribution

decider is by Mapika (https://github.com/Mapika/decider), released under Apache-2.0 and built on
Qwen3.5-Base by the Qwen team (Apache-2.0). The reference imports the `decider/` package shipped in each
model repo at the pinned commit.

## Parity results

decider-0.8b: weightless graph on ONNX Runtime 1.30 CPU, against upstream fp32 on CUDA with TF32 off.
Report in `convert/out/decider-0.8b/parity.json`.

| | value |
|---|---|
| requests | 131 (77 Laya edge + 4 decoder edge + 50 typed-decisions, spread over all workflows) |
| questions / scoring rows | 632 / 1,248. Isolated score levels are one row each. 14 requests were rejected by both sides (a 1-option choice); their valid questions were scored individually. |
| token rows and slots identical to upstream | 1,248 / 1,248 |
| argmax agreement, ONNX vs fp32 | **100 %** |
| argmax agreement, ONNX vs upstream `system_one` | **100 %** |
| max \|Δ logit\|, all 255 labels | 3.2e-5 (p99 2.3e-5) |
| max \|Δ probability\| | 3.5e-6 (choice), 7.4e-6 (noul), 2.9e-6 (score) |
| contract (option logits + calibration) vs upstream answers | ≤ 5.0e-5, which is upstream's 4-decimal rounding |

Goldens: `convert/out/goldens-decider-0.8b.jsonl`, written by `families/decider/goldens.py`. Per request
it holds the exact rows and slots, label logits, option logits, probabilities and upstream answers.

decider-2b: same harness, same 131 requests (the full edge set plus 50 typed-decisions rows). ONNX
Runtime CPU, 12 threads; reference fp32 on CUDA. Report in `convert/out/decider-2b/parity.json`.

| | value |
|---|---|
| questions / scoring rows | 632 / 1,248; token rows identical 1,248 / 1,248 |
| argmax agreement, ONNX vs fp32 / vs upstream `system_one` | **100 % / 100 %** |
| max \|Δ logit\| | 3.4e-5 (p99 2.5e-5) |
| max \|Δ probability\| | 5.5e-6 (choice), 4.0e-6 (noul), 5.9e-6 (score) |
| contract vs upstream answers | ≤ 5.0e-5 (rounding) |

- **Files.** The graph is 8.0 MB. It references `Mapika/decider-2b@9839cc9d…/model.safetensors`
  (3.76 GB, sha256 `1bf79b6a…`, matches the HF LFS oid): 320 / 320 tensors, with no unused tensors.
- **Goldens.** `convert/out/goldens-decider-2b.jsonl`.

## ONNX Runtime CUDA EP note (measured)

- **The bug.** ORT 1.30 CUDA EP runs these raw dynamo exports only with
  `SessionOptions.enable_mem_reuse = false`. With memory reuse on, `Reshape` reads shape values from
  reused buffers and fails, for example with "requested shape {3,3,-1,128}". The CPU EP is unaffected;
  every parity number above is from the CPU EP.
- **Verified workaround.** `enable_mem_reuse=false` matches CPU to 1e-5, checked on the Qwen3Guard graph.
- **Alternative.** Re-export with `OLLAYA_FOLD_LIMIT=64`, which runs the onnxscript optimizer with
  folding capped at 64 elements. The CUDA EP then works with default options (checked on Qwen3Guard:
  CUDA vs CPU 1.3e-5). But it folds a few tiny weight-derived tensors into the graph (`-exp(A_log)`,
  16 floats per DeltaNet layer), and those graphs have not been through full parity.
- **Rust runner.** It uses ORT 1.28, which was not tested. It should check this with a goldens run on CUDA.

## Typed-decisions quality (measured, for the catalog comparison)

Test split: 400 rows, 2,000 questions, teacher gold. Scored with the fp32 upstream reference, which the
ONNX graphs match to about 1e-5. Metrics are those of [llm-logits.md](llm-logits.md).

| | acc (as shipped) | ECE (as shipped) | cross-fitted acc / ECE |
|---|---|---|---|
| decider-2b | 0.591 | 0.089 | 0.591 / 0.081 |
| decider-0.8b | 0.506 | 0.172 | 0.506 / 0.048 |

For reference: Laya typed-decisions scores 0.768 (it trains on this dataset), Jev 0.738, Winnow-12B
0.700 (Winnow's report), and generic Gemma 4 12B-it through `llm-logits-v1` 0.717.

decider is trained on public benchmark mixtures, not this dataset's workflow schemas. Its model-card
numbers above are its intended evaluation.
