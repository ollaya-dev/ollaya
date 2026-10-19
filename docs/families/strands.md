# strands (`strands-pointer-v1`)

Strands Agents' **Strands Decider** is a decision model built from three parts:
- a LoRA adapter (r=48, α=96) on `Qwen/Qwen3.5-2B-Base`;
- a **pointer head**: a LayerNorm and two `Linear(2048→768)` projections, where option j's score is
  `k(h_opt_j)·q(h_decide)/√768`;
- fitted temperatures per question type in `strands_decider_config.json`.

It serves TypeSafe's `POST /v1/systemone`. The code is https://github.com/strands-labs/strands-decider, pinned
here at `930e6d5de16309be32579f5631a7369a3322140c`; the model repo holds the adapter (`lora/`), `head.safetensors`,
the config and a tokenizer. Ollama 0.40 lists the model as `strands-decider`, but only for its MLX engine
(macOS); on Linux and Windows its pull fails.

| Model | Checkpoint | Base (license) | Temperatures (choice / score / noul) | Status in Ollaya |
|---|---|---|---|---|
| `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | `216d1ccb` (2026-10) | Qwen/Qwen3.5-2B-Base @ `b1485b2f` (Apache-2.0), 1 shard | 0.863 / 1.834 / 1.070 | **converted, ONNX** (`strands-decider:2b`) |
| `strands-decider-2B-hobson-v19`, `-v21` | | same base | | earlier recipe, same layout per upstream's docs; not converted |
| Gemma 4 variants | | Gemma 4 | | different trunk; not converted |

## Engine: ONNX (single forward per question)

- **Export.** `families/strands/export.py` reuses `llm_common/qwen35.py` (the scan-based Gated-DeltaNet trunk,
  see [decider.md](decider.md)) and adds the head. The graph has kev's contract, so the runner shares
  `kev::pointer_scores`.
- **LoRA.** Not merged: each adapted Linear runs as `x·Wᵀ + 2.0·(x·Aᵀ)·Bᵀ`, so base and adapter weights stay
  byte-referenced. The eager graph matches upstream's fp32 model to 1.8e-6 on the scores.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph, 10.1 MB) | derived, hosted by Ollaya |
| base weights, BF16 | `Qwen/Qwen3.5-2B-Base`'s `model.safetensors-00001-of-00001.safetensors` (4.5 GB) at the pinned revision; the vision tower (312 tensors) and MTP tensors are unused |
| LoRA adapter, F32 | the model repo's `lora/adapter_model.safetensors` (202 MB), every tensor used |
| pointer head, F32 | same repo, `head.safetensors` (12.6 MB) |
| `tokenizer.json` | same repo, as-is (byte-identical to what transformers 5.17 builds for the base) |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya (temperatures copied from `strands_decider_config.json`) |
| license | `LICENSE.md` in the model repo: Apache-2.0 (adapter, head and code); the base is Apache-2.0 |

`weights_in_memory` is `fp32` (widened once at load, about 8 GB).

## Request → rows

Source: `strands_decider.prompting` (`render_state`, `render_question`), `SystemOneEngine._chunks` / `_option_idx`
and the pydantic request models in `schema.py`. Port: `families/strands/layout.py`, mirrored by
`crates/ollaya-decision/src/strands.rs`.

### Validation (HTTP 422 upstream)

- `state` is required; it and `instructions` must be a string, an object or an array (a bare number is rejected).
- **noul.** `criteria` is null or an object with only `true` / `false` keys.
- **choice.** `criteria` is an object of **2..255** entries. A list of labels is rejected (write
  `{"label": null}`).
- **score.** `criteria` is a list of **2..10** strings.

### Text

`render` is `""` for null, `str.strip()` for a string, else `json.dumps(indent=2, ensure_ascii=False)`;
descriptions are also collapsed to single spaces.

```
state    = "<state>\n" render(state) "\n</state>\n"
question = "<question type=\"{type}\">\n" header "\n" render(instructions) "\n<options>\n"
           line_1 "\n" … line_k "\n</options>\n</question>\n<answer>"
line_j   = "{j}. {name}"  or  "{j}. {name} — {collapse(description)}"
```

| Type | Header | Options, in order (wire order) |
|---|---|---|
| noul | "Decide whether the statement is true of the state." | `false`, `true`; a missing side reads "the statement does not hold / holds for this state" |
| choice | "Select exactly one option." | each label with its description, in key order |
| score | "Rate the state against the ordered levels below (lowest first)." | `0`, `1`, … with the level as description |

### Fitting the window

Tokenizer: `tokenizer.json` with no special tokens. Questions are fitted in runs of `max_batch` = 32. With `L` the
run's longest question, every question of the run keeps its **last** `min(L, int(4096 × 0.75))` tokens, and the
state its **first** `4096 − that` tokens. A row is `state_ids + question_ids`; `decide_pos` is its last token
(`<answer>`), and option j's position is the last token of line j that lies wholly inside the kept text. If a
question's cut leaves an option with no token, the request is rejected, as upstream does.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question; `seq` a multiple of 64; right-pad with any id (`pad` = 248044) |
| `decide_pos` | int64 | `[rows]` | the row's last token |
| `opt_pos` | int64 | `[rows, k]` | each option line's last token; pad short rows with 0 |
| `scores` | float32 | `[rows, k]` | raw pointer scores (temperature not applied) |

Probabilities are `softmax(scores / T[type])`. TypeSafe answers follow upstream `infer._to_answer`: noul is P(true);
choice is the argmax with confidence `(N·p_max − 1)/(N − 1)`; score is the expected level with upstream's ordinal
confidence. Answers are rounded to 4 decimals.

## Parity (measured)

Goldens: `families/strands/goldens.py`, upstream `strands_decider` in fp32 on CUDA (TF32 off, no prefix cache):
134 records from 152 requests (the shared Laya edge cases, the decoder cases, 50 typed-decisions rows and five
Strands cases: a long question, a long state, an option cut away, 40 questions in two runs, whitespace), 18 of them
rejected upstream. The layout port reproduces every row before a record is written.

`crates/ollaya-runner/examples/parity_strands.rs` on choso-wsl (i9-13900K, RTX 4090, ONNX Runtime 1.28):

| | rows | rejections | max \|Δ option logit\| | max \|Δ probability\| (p99) | answers vs upstream |
|---|---|---|---|---|---|
| CUDA | 650 / 650 identical | 0 mismatches | 1.1e-5 | 2.8e-6 (1.6e-6) | 0 differ (max 1.0e-4, the rounding) |
| CPU | 650 / 650 identical | 0 mismatches | 8.2e-6 | 2.2e-6 (1.5e-6) | 0 differ (max 1.0e-4, the rounding) |

Latency on the RTX 4090, five-question requests (1,383 tokens each): p50 344 ms, p95 614 ms.

```sh
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_strands -- \
    convert/out/strands-decider-2b convert/out/goldens-strands-decider-2b.jsonl cuda --latency
cargo run --release -p ollaya-runner --example parity_strands -- \
    convert/out/strands-decider-2b convert/out/goldens-strands-decider-2b.jsonl cpu
```

The CPU run takes 11.4 s per request on the i9-13900K (the goldens include 4,096-token rows); a CPU is not a
practical target for this model.

## fp16 tag

`strands-decider:2b-fp16` runs the same weights with the trunk in fp16 on NVIDIA GPUs ([ADR 0008](../decisions/0008-fp16-decoder-tags.md)).
The RMSNorms, the gated delta rule, the attention softmax, the rotary frequencies, the masks and the heads stay fp32,
as the authors' code computes them. The gate (`parity_strands --fp16`, run by `parity_run`) keeps the encodings exact and
requires at least 99.5 % of the decisions to match the fp32 reference. On the RTX 4090
(`results/runs/2026-10-19-parity-rtx4090-cuda.json`):

| Tag | Decisions the same | Max option logit diff | Max probability diff (p99) |
|---|---|---:|---:|
| `strands-decider:2b-fp16` | 649 / 650 (99.85 %) | 2.3e-02 | 5.6e-03 (4.4e-03) |

The fp32 tag is unchanged and still matches the authors' code. Speed and typed-decisions accuracy of
the fp16 tag are on the model page and on bench.ollaya.dev. On a CPU an fp16 tag runs, slowly; use the fp32 tag there.

## Limits

- **Not ported.** Images (`serve --vision`), the prefix cache (an exact-result speedup upstream), and the
  `strict_window` option (off by default upstream).
- **Context.** 4,096 tokens per row; a long state is cut from the end, a long question from the front.

## Upstream numbers (model card)

JevBench public at window 4096: 180 / 231 tasks, Brier 0.280, ECE 0.072. Internal held-out short tasks 0.653
(n = 6,000), ContractNLI 0.878, HotpotQA (held out) 0.832, MuSiQue 0.882.

## Attribution

Strands Decider is by Strands Agents (also published as `amazon/strands-decider-2B-qwen3.5-v1-2610`), Apache-2.0.
The base, Qwen3.5-2B-Base, is by the Qwen team (Apache-2.0). Training data sources carry their own licenses; see
the model card.
