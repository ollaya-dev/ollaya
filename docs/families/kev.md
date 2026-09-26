# kev (`kev-pointer-v1`)

jaredpalmer's **Kev** models are decision models built from three parts:
- a LoRA adapter (r=16, α=32) on a Qwen base;
- a **pointer head**: two `Linear(hidden→256)` projections, where the option score is
  `k(h_opt)·q(h_decide)/16`;
- a fitted temperature stored in `head.pt`.

They serve TypeSafe's `POST /v1/systemone`. Inference code lives in https://github.com/jaredpalmer/kev,
pinned here at `234e5a7498f82f253de34e67b9fa99aefb5f20f5`. The model repos hold only the adapter,
`head.pt` and a tokenizer.

**The pin still describes what upstream serves** (checked at kev HEAD `3d9973b`, 2026-09-25): `kev/api.py` is
byte-identical (sha256 `7bffacfb…`), and in `kev/model.py` the delimiter tokens, `user()`, `encode`, `rows_of`
and `PointerHead` are unchanged; later commits add serving paths (a prefix cache, CUDA graphs, fused kernels,
batching) that upstream describes as equal up to fp32 rounding. The checkpoints' own `provenance.json` agree:
kev-0.8b r15 and kev-4b r10 were trained with this `api.py` and a `model.py` whose encoding and head code are
identical to the pin's; kev-9b was trained with an earlier revision (`406f464`) whose token rows and head math
are the same.

| Model | Checkpoint | Base (license) | Temperature | Status in Ollaya |
|---|---|---|---|---|
| `jaredpalmer/kev-0.8b` | round 15, `9a45d25e` (2026-09-24) | Qwen/Qwen3.5-0.8B-Base @ `dc7cdfe2` (Apache-2.0), 1 shard | 2.351 | **converted, ONNX** |
| `jaredpalmer/kev-4b` | round 10, `139fdd94` (2026-09-24) | Qwen/Qwen3.5-4B-Base @ `1001bb4d` (Apache-2.0), 2 shards | 2.406 | **converted, ONNX** (weights stay BF16 in memory) |
| `jaredpalmer/kev-9b` | `2629c06a` (2026-09-21) | Qwen/Qwen3.5-9B-Base @ `68c46c4b` (Apache-2.0), 4 shards | 2.297 | **converted, ONNX** (weights stay BF16 in memory) |
| `jaredpalmer/kev-0.5b` / `kev-0.6b` / `kev-8b` | | Qwen2.5-0.5B / Qwen3-0.6B-Base / Qwen3-8B-Base | | attention-only bases: upstream uses a packed block-causal form; the row form here is equivalent (upstream tests `test_rows_match_packed`) but they need a plain-attention trunk; not converted |

kev-0.8b was first converted at `54f4f877` (round 7 plus the dates delta, T 2.406); Ollaya now pins round 15.

## Recommended engine: ONNX (single forward)

- **Why ONNX.** A decision is one causal row per question plus the pointer head. The head is not the
  LM head, so llama.cpp could not produce these scores without custom code; ONNX runs the head directly.
- **Export.** It reuses `llm_common/qwen35.py` (the scan-based Gated-DeltaNet export, see
  [decider.md](decider.md)).
- **LoRA.** It is **not merged**. Each adapted Linear runs as `x·Wᵀ + 2.0·(x·Aᵀ)·Bᵀ`, so base and adapter
  weights both stay byte-referenced.
- **Numerics.** Merged vs unmerged differ by at most 3e-6 on the scores.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph, 10 to 11 MB) | derived, hosted by Ollaya |
| base weights, BF16 | the base repo's `model.safetensors-0000i-of-0000n.safetensors` shards at the pinned revision: 0.8B 1 shard (1.7 GB), 4B 2 shards (9.3 GB), 9B 4 shards (19.3 GB). The vision tower and MTP tensors (and the 9B's untied `lm_head`) are unused. |
| LoRA adapter, F32 | the kev repo's `adapter_model.safetensors`: 43 MB (0.8b), 130 MB (4b), 173 MB (9b); every tensor used |
| pointer head, F32 | same repo, `head.pt` (2.1 to 8.4 MB). A `torch.save` zip whose tensors are stored uncompressed, so the graph references them **by byte offset inside the zip**. Nothing is re-packed. |
| `tokenizer.json` | same repo, used as-is (the same file in all three repos) |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya (temperature copied from `head.pt`) |
| license | Apache-2.0 per the model card (adapter + head). The base has its own LICENSE (Apache-2.0). |

- **Mapping.** Every base tensor of the language model, every adapter tensor and the four head tensors map to
  graph initializers; the base's are a `Cast` from BF16, the rest F32. About 100 KB of masks stay inline. The
  embedding lookup reads rows before widening them (`Cast(Gather(E, ids))`, see
  [decider.md](decider.md)).
- **Weights in memory.** `decision.json` `weights_in_memory` is `fp32` for 0.8b (widened once at load) and `bf16`
  for 4b and 9b (kept as stored, widened per forward pass): see
  [ADR-0001](../decisions/0002-decoder-weights-in-memory.md).
- **Hashes.** sha256 values are in each export's `files.json` (`convert/out/kev-0.8b-r15`, `kev-4b`, `kev-9b`).
- **Tokenizer caveat (important).** Use the kev repo's `tokenizer.json`, not the base's. Upstream loads
  the base tokenizer through transformers 5.17. Its `Qwen2Tokenizer` replaces the base file's
  pre-tokenizer `[\p{L}\p{M}]+` with `\p{L}+`, and the kev repo file is that re-saved tokenizer. With the
  base file, combining-mark scripts such as Hindi or Thai tokenize differently.

## Request → rows

Source: `kev.api.to_record` + `kev.model.encode` + `rows_of`. Port: `families/kev/layout.py`.

### Validation

These rules follow pydantic `SystemOneRequest`; failures are HTTP 422.
- `questions` needs at least one entry.
- `type` must be one of `noul`, `choice`, `score`.
- **noul.** `criteria` is an object or null.
- **choice.** `criteria` is an object of **1..255** entries. A list is rejected, unlike Laya.
- **score.** `criteria` is a list of **1..255** entries.
- **instructions.** Optional; any JSON.

### Text rendering (`render`)

- `None` renders as `""`.
- A str, int, float or bool renders as Python `str(v)`: `True`, `1.0`, `1e-05`, big ints as-is.
  JSON `1` and `1.0` differ (`1` vs `1.0`), so the Rust port must keep float-vs-int.
- A list renders as `"\n".join(f"{pad}- {render(x, indent+1).lstrip()}")`.
- A dict renders as `"\n".join(f"{pad}{k}:\n{render(x, indent+1)}" if x is a dict or list else f"{pad}{k}: {render(x)}")`.
- `pad = "  " * indent`.

### Options per type

`option_text(name, d)` is `name` if `d` is null or `""`, otherwise `name + ": " + render(d)`.

| Type | Options, in order |
|---|---|
| noul | `option_text("no", criteria.false)`, `option_text("yes", criteria.true)`. Wire order `[false, true]`; keys are case-sensitive. |
| choice | `option_text(key, value)` in key order |
| score | `render(level)` per level. There is no `i:` prefix, and a null level renders as an empty option. |

### Row tokens

Tokenization uses `tokenizer.json` with `add_special_tokens=False`. Before tokenizing, `user(text)`
rewrites every `<|name|>` (`<\|([A-Za-z0-9_]+)\|>`) to `<¦name¦>`, so user text cannot produce the
delimiters.

```
S    = [<|fim_prefix|>=248060] + user(render(state))[:8191]            # state, shared by every row
row  = S + [<|fim_middle|>=248061] + user(render(instructions))
         + Σ_j ([<|box_start|>=248049] + user(option_j) + [<|box_end|>=248050])
         + [<|fim_suffix|>=248062]                                     # decide token
decide_pos = len(row) - 1;  opt_pos[j] = index of option j's <|box_end|>
```

The special-token ids come from `decision.json.special_tokens` (`state`, `question`, `option_open`,
`option_close`, `decide`).

**Budgets.**
- The state is cut to 8,191 tokens plus the `<state>` token.
- A row longer than 8,192 tokens is rejected with 422 (upstream `ContextOverflow`, `SERVE_MAX_BRANCH`).
- Options are never truncated.

### Option logits

- **All types.** Option logits are `scores[row, :k]`.
- **Calibration.** `calibration.json` holds the checkpoint's temperature for every type: 2.351 (0.8b r15), 2.406
  (4b r10), 2.297 (9b). Upstream fitted each on in-distribution dev rows (`head.pt["temperature_fit"]`) and applies
  it for every type.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question; **`seq` a multiple of 64**; right-pad with any id (`pad` = 248044) |
| `decide_pos` | int64 | `[rows]` | index of the decide token |
| `opt_pos` | int64 | `[rows, k]` | indices of the option-closing tokens; pad short rows with 0 |
| `scores` | float32 | `[rows, k]` | raw pointer scores (temperature not applied); use the first `k_row` |

- **Positions and masking.** Positions are implicit (`0..seq-1`), matching upstream's row form. There is
  no mask input: the model is causal and right padding is inert.
- **Precision.** Opset 20; fp32 compute on the BF16 base weights, widened at load (0.8b) or per forward pass
  (4b, 9b; `weights_in_memory`).

## Measured parity

`families/kev/parity.py`: ONNX Runtime CPU with the LoRA unmerged, against upstream `kev` in fp32
(`Checkpoint.load(dtype=fp32, merge=True)`, row form, no prefix cache). Results are under "Parity
results" below.

The port reproduces upstream token rows, decide positions and option positions on **all 481
requests**: 77 Laya edge cases, 4 decoder cases and 400 typed-decisions rows. It also rejects the same
requests; for example, list-valued choice criteria get a pydantic 422 in both.

## Limits

- **Options.** Choice 1..255; score 1..255 levels. TypeSafe caps score at 10; kev does not.
- **Context.** Rows up to 8,192 tokens, beyond the 384-token training state length (upstream flags this
  as "untested there").
- **Accuracy.** Out-of-domain accuracy is modest at 0.8B: 0.648 on transfer-v4 development, where Kev-4B r10
  gets 0.817 and Kev-9B 0.822.
- **Memory.** The weights take 8.5 GB (4b) and 16 GB (9b), kept BF16, plus activations that grow with the
  row length. On an RTX 4090, kev-9b's parity run peaked at 22.0 GiB of the card's 24.0 with rows of up to 2,033
  tokens, so rows much longer than that do not fit a 24 GB GPU.
- **Tokenizer.** The Qwen3.5 tokenizer caveat above applies.
- **Not ported.** The opt-in `KEV_DATE_FACTS` preprocessing, and the demo endpoints `/permute` and
  `/separate`.

## Upstream benchmark numbers (model cards)

| | Kev-0.8B r15 | Kev-4B r10 | Kev-9B | Jev |
|---|---|---|---|---|
| decision-v7 dev, in-distribution, accuracy | 0.827 (1,264 q.) | 0.873 (1,264 q.) | 0.872 (1,204 records) | 0.845 |
| transfer-v4 dev, out-of-domain, accuracy / Brier | 0.648 / 0.430 (656 q.) | 0.817 / 0.243 (656 q.) | 0.822 / 0.286 (764 records) | 0.857 / 0.211 |
| transfer-v4 locked test, out-of-domain accuracy / Brier | 0.697 / 0.397 | 0.838 / 0.224 | 0.852 / 0.237 | – |
| real documents (documents-v1 locked test), accuracy | 0.851 | – | – | – |
| skill records (hard-v1 locked test), accuracy | 0.665 | 0.803 | – | – |
| developer tooling (devtools-v1 locked test), accuracy | 0.637 | 0.756 | – | – |

The Decision Index 0.2 (2026-09-25) scores Kev-9B at 35.41 balanced skill, Kev-4B at 31.31 (an earlier round than
r10) and Kev-0.8B at 13.26 (Jev 51.67).

## Attribution

Kev is by Jared Palmer (https://github.com/jaredpalmer/kev), released under Apache-2.0 (adapter and
head). The bases, Qwen3.5-0.8B-Base, Qwen3.5-4B-Base and Qwen3.5-9B-Base, are by the Qwen team
(Apache-2.0). Training data sources carry their own licenses; see the model cards.

## Parity results

Goldens come from upstream `kev` at the pinned commit in fp32 (`Checkpoint.load(dtype=fp32, merge=True)`, row form,
TF32 off), written by `families/kev/goldens.py`: 117 records per checkpoint (77 Laya edge cases, 4 decoder cases and
20 typed-decisions rows, plus the `#valid` subsets of the 16 requests upstream rejects), 480 questions. Per request they
hold the exact rows, decide and option positions, fp32 option logits, probabilities at the checkpoint's temperature
and upstream answers.

| checkpoint | goldens | reference ran on |
|---|---|---|
| kev-0.8b round 15 | `convert/out/goldens-kev-0.8b-r15.jsonl` | CUDA |
| kev-4b round 10 | `convert/out/goldens-kev-4b.jsonl` | CUDA |
| kev-9b | `convert/out/goldens-kev-9b.jsonl` | CPU (its 32 GB of fp32 weights do not fit the GPU) |

The token rows of all three are identical to each other and to the round-7 kev-0.8b goldens
(`goldens-kev-0.8b.jsonl`), record for record: the encoding did not change between checkpoints or code revisions.
The round-7 conversion also had an export-side check (ONNX Runtime 1.30 CPU vs upstream fp32 on 131 requests, max
|Δ probability| 1.9e-6; `convert/out/kev-0.8b/parity.json`).

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
- **Rust runner.** ORT 1.28 through the C API, which has no `enable_mem_reuse`. The runner uses the
  decider's CUDA session options instead (parallel execution mode, which reuses no buffers of the main
  graph); see `crates/ollaya-runner/src/decider.rs`. Measured below: CUDA matches the goldens.

## Rust runtime parity (measured)

`crates/ollaya-runner/examples/parity_kev.rs` against each checkpoint's goldens: 117 records, of which 16 are
requests upstream rejects (list-valued choice criteria) and 16 their `#valid` subsets, 480 questions. It checks the
rejections, then the token rows, decide and option positions, then the option logits (tolerance 1e-3), the decisions
and the TypeSafe answers against upstream `to_answers`.

```sh
cargo run --release -p ollaya-runner --example parity_kev -- convert/out/kev-4b convert/out/goldens-kev-4b.jsonl cpu
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_kev -- convert/out/kev-4b convert/out/goldens-kev-4b.jsonl cuda
```

Results on choso-wsl (i9-13900K; RTX 4090, CUDA 13), `ort` 2.0.0-rc.13 with ONNX Runtime 1.28, each checkpoint in its
shipped configuration (0.8b folds its weights to fp32 at load; 4b and 9b keep them BF16). Every run: 0 rejection
mismatches, token rows and positions identical 480 / 480, decisions agree **100 %**, and 0 TypeSafe answers differ
from upstream (max 1.0e-4, its 4-decimal rounding).

| | max \|Δ option logit\| | max \|Δ probability\| (p99) |
|---|---|---|
| kev-0.8b r15, CPU | 4.6e-5 | 2.8e-6 (1.3e-6) |
| kev-0.8b r15, CUDA | 3.4e-5 | 1.9e-6 (1.5e-6) |
| kev-4b r10, CPU | 2.3e-4 | 3.1e-5 (2.5e-6) |
| kev-4b r10, CUDA | 2.2e-4 | 3.0e-5 (2.8e-6) |
| kev-9b, CPU | 3.6e-5 | 3.8e-6 (2.0e-6) |
| kev-9b, CUDA | 7.5e-5 | 4.2e-6 (3.3e-6) |

### ORT 1.28 `GemmTransposeFusion` (disabled for kev)

- **What the graph has.** Its only `Gemm` is the pointer head's query projection (`head.q`,
  1,024 → 256, node `node_linear_558`). Its input, the decide-token hidden states `[rows, 1024]`,
  comes from the export's row gather (`GatherND`) through an identity `Transpose` (`node_index`,
  perm `[0, 1]`). Neither the Qwen3Guard nor the decider graphs have a `Gemm`.
- **The bug.** ORT 1.28's `GemmTransposeFusion` folds a `Transpose` that feeds only `Gemm`s into
  them by flipping `transA`, without checking `perm`. On this graph the fused `Gemm` multiplies the
  transposed input. With the pass left on, `parity_kev` fails on its first batch: "Non-zero status
  code returned while running Gemm node. Name:'node_linear_558/GemmTransposeFusion/' Status Message:
  GEMM: Dimension mismatch, W: {256,1024} K: 5 N:256" (K is the batch's 5 rows, read as the inner
  dimension). A batch of exactly 1,024 rows (the hidden size) would pass the shape check and return
  wrong scores; the runner's batches never get there (at most 8,192 / 64 = 128 rows). ORT 1.30
  checks that the `Transpose` is a real matrix transpose (perm `[1, 0]`) first
  (microsoft/onnxruntime#32435), so the Python parity with ORT 1.30 never met it.
- **The fix.** kev sessions disable the pass (`with_disabled_optimizers("GemmTransposeFusion")`), so
  ORT runs the graph as exported. It is a workaround for a wrong rewrite, not a precision trade: with
  it, the Rust runtime matches the goldens on CPU and CUDA (above). The head is two small matmuls, so
  the fusion is worth nothing here. The setting can go once `ort` links ORT 1.30 or newer.
- **Larger checkpoints.** The 4b and 9b graphs have the same head, with 2,560 and 4,096 inputs; the
  workaround applies to every kev session.

## Typed-decisions quality (measured, for the catalog comparison)

The typed-decisions test split (400 rows, 2,000 questions, teacher gold), at each checkpoint's shipped temperature.
0.8b and 4b are scored with the fp32 upstream reference (`llm_common/eval_refs.py`); 9b's fp32 reference does not fit
the GPU, so it is scored through Ollaya's runtime on CUDA, which matches the reference to the numbers above (the same
path gives kev-4b 0.669 / 0.095 against its reference's 0.669).

| | accuracy | ECE | choice / score / noul | cross-fitted accuracy / ECE |
|---|---|---|---|---|
| kev-0.8b r15 (T 2.351) | 0.460 | 0.055 | 0.505 / 0.376 / 0.525 | 0.460 / 0.049 |
| kev-4b r10 (T 2.406) | 0.669 | 0.095 | 0.645 / 0.615 / 0.765 | 0.669 / 0.123 |
| kev-9b (T 2.297) | 0.722 | 0.095 | 0.693 / 0.675 / 0.812 | – |

For reference: round-7 kev-0.8b scored 0.447 / 0.055, decider-2b 0.591 and decider-4b 0.680; Winnow's report has
Kev-4B 0.658 and Kev-9B 0.716 (it does not say which revisions it scored).
