# kev (`kev-pointer-v1`)

jaredpalmer's **Kev** models are decision models built from three parts:
- a LoRA adapter (r=16, α=32) on a Qwen base;
- a **pointer head**: two `Linear(hidden→256)` projections, where the option score is
  `k(h_opt)·q(h_decide)/16`;
- a fitted temperature stored in `head.pt`.

They serve TypeSafe's `POST /v1/systemone`. Inference code lives in https://github.com/jaredpalmer/kev,
pinned here at `234e5a7498f82f253de34e67b9fa99aefb5f20f5`. The model repos hold only the adapter,
`head.pt` and a tokenizer.

| Model | Base (license) | Status in Ollaya |
|---|---|---|
| `jaredpalmer/kev-0.8b` | Qwen/Qwen3.5-0.8B-Base @ `dc7cdfe2` (Apache-2.0) | **converted, ONNX** |
| `jaredpalmer/kev-4b` | Qwen/Qwen3.5-4B-Base | same exporter (hybrid Qwen3.5); not converted, fp32 is ~16 GB |
| `jaredpalmer/kev-9b` | Qwen/Qwen3.5-9B-Base | not converted |
| `jaredpalmer/kev-0.5b` / `kev-0.6b` / `kev-8b` | Qwen2.5-0.5B / Qwen3-0.6B-Base / Qwen3-8B-Base | attention-only bases: upstream uses a packed block-causal form; the row form here is equivalent (upstream tests `test_rows_match_packed`) but they need a plain-attention trunk; not converted |

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
| `model.onnx` (graph, 9.9 MB) | derived, hosted by Ollaya |
| base weights, BF16 + F32 | `Qwen/Qwen3.5-0.8B-Base@dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68/model.safetensors-00001-of-00001.safetensors` (1.7 GB; its vision and MTP tensors are unused, 168 of 488) |
| LoRA adapter, F32 | `jaredpalmer/kev-0.8b@54f4f8777356cd5bbbb6c6919c657f26e6f2f6d8/adapter_model.safetensors` (43 MB; all 372 used) |
| pointer head, F32 | same repo, `head.pt` (2.1 MB). A `torch.save` zip whose tensors are stored uncompressed, so the graph references them **by byte offset inside the zip**. Nothing is re-packed. |
| `tokenizer.json` | same repo, used as-is |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya (temperature copied from `head.pt`) |
| license | Apache-2.0 per the model card (adapter + head). The base has its own LICENSE (Apache-2.0). |

- **Mapping.** 696 checkpoint tensors map to graph initializers: 284 casts from BF16, the rest F32.
  About 100 KB of masks stay inline.
- **Hashes.** sha256 values are in `convert/out/kev-0.8b/files.json`.
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
- **Calibration.** `calibration.json` = `{"temperature": [2.406, 2.406, 2.406]}`. Upstream fitted this on
  in-distribution dev rows and applies it for every type.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question; **`seq` a multiple of 64**; right-pad with any id (`pad` = 248044) |
| `decide_pos` | int64 | `[rows]` | index of the decide token |
| `opt_pos` | int64 | `[rows, k]` | indices of the option-closing tokens; pad short rows with 0 |
| `scores` | float32 | `[rows, k]` | raw pointer scores (temperature not applied); use the first `k_row` |

- **Positions and masking.** Positions are implicit (`0..seq-1`), matching upstream's row form. There is
  no mask input: the model is causal and right padding is inert.
- **Precision.** Opset 20; fp32 compute, BF16 base weights cast at load.

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
- **Accuracy.** Out-of-domain accuracy is modest at 0.8B: 0.652 on transfer-v4, where Kev-4B gets 0.797.
- **Tokenizer.** The Qwen3.5 tokenizer caveat above applies.
- **Not ported.** The opt-in `KEV_DATE_FACTS` preprocessing, and the demo endpoints `/permute` and
  `/separate`.

## Upstream benchmark numbers (model card, Kev-0.8B)

| | value |
|---|---|
| decision-v7 dev, in-distribution (1,204 records), accuracy | 0.825 |
| transfer-v4 dev, out-of-domain (764 records), accuracy / Brier | 0.652 / 0.499 |
| locked test, in-distribution / out-of-domain accuracy | 0.834 / 0.684 |
| as served (T = 2.41), Brier / ECE / confident errors | 0.430 / 0.054 / 0.3 % |
| Jev on the same items, in-distribution / out-of-domain | 0.845 / 0.857 |

## Attribution

Kev is by Jared Palmer (https://github.com/jaredpalmer/kev), released under Apache-2.0 (adapter and
head). The base, Qwen3.5-0.8B-Base, is by the Qwen team (Apache-2.0). Training data sources carry their
own licenses; see the model card.

## Parity results

kev-0.8b: weightless graph on ONNX Runtime 1.30 CPU, against upstream fp32 on CUDA with TF32 off.
Report in `convert/out/kev-0.8b/parity.json`.

| | value |
|---|---|
| requests | 131 (77 Laya edge + 4 decoder edge + 50 typed-decisions, spread over all workflows) |
| questions / rows | 630 / 630. Some requests were rejected: 16 by both sides (list-valued choice criteria → pydantic 422), with their valid questions scored individually. |
| token rows, decide and option positions identical | 630 / 630 |
| argmax agreement, ONNX vs fp32 | **100 %** |
| argmax agreement, ONNX vs upstream `to_answers` | **100 %** |
| max \|Δ score\| (raw pointer logits) | 3.0e-5 (choice), 1.9e-5 (noul), 2.0e-5 (score) |
| max \|Δ probability\| (T = 2.406) | 1.9e-6 |
| contract (scores / T → softmax) vs upstream answers | ≤ 5.0e-5, which is upstream's 4-decimal rounding |

Goldens: `convert/out/goldens-kev-0.8b.jsonl`, written by `families/kev/goldens.py`. Per request it holds
the exact rows, decide and option positions, fp32 option logits, probabilities and upstream answers.

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

kev-0.8b on the typed-decisions test split (400 rows, 2,000 questions, teacher gold), scored with the
fp32 upstream reference:

- **As shipped (T = 2.41).** Accuracy 0.447, ECE 0.055.
- **Per type.** Choice 0.480, score 0.361, noul 0.528.
- **Cross-fitted per-type temperatures.** 0.447 / 0.053.
- **Reference points on the same decisions.** Kev-4B 0.658 and Kev-9B 0.716 (Winnow's report),
  decider-0.8b 0.506.
