# pplx (`pplx-joint-v1`)

Perplexity's **pplx-decider** models are post-trained Qwen3.5 backbones with a decision head:
- the backbone is a `Qwen3_5Model` whose **full-attention layers read the whole sequence** (upstream
  `attention_mode: noncausal_full_attention`); the Gated-DeltaNet layers stay causal;
- the head (`readout.safetensors`) is one `Linear(hidden → 255, bias=False)`, read at a `<|fim_suffix|>` marker
  placed after each question; its outputs are the logits of the answer codes `A`, `B`, …;
- every question of a request shares **one sequence** (protocol `joint-question-inline-readout-v1`, version 3);
- one fitted temperature (`decision_config.json`).

The inference code ships in the model repository (`source/src/pplx_decider`); the reference imports it from the
pinned snapshot. Ollama 0.40's `pplx-decider` is the 27B v1.1 (`pplx-decider-v1.1-27b`, by its temperature 1.0087; Q4_K_M, 17 GB), served by llama.cpp's
decision server behind Ollama's request limits (1 to 64 questions, 2 to 10 score levels).

| Model | Checkpoint | Base | Temperature | Status in Ollaya |
|---|---|---|---|---|
| `perplexity-ai/pplx-decider-flash-v1.2` | `6d771ad7` (2026-10) | Qwen3.5-4B, fully post-trained (distilled from the 27B models, souped) | 1.554 | **converted, ONNX** (`pplx-decider:flash`) |
| `perplexity-ai/pplx-decider-v1.2`, `-v1.1-27b` | | Qwen3.8-27B (11 weight shards and the readout) | | same layout; not converted (about 54 GB of BF16 weights, beyond a 24 GB GPU) |

## Engine: ONNX (one forward per request)

- **Export.** `families/pplx/export.py` reuses `llm_common/qwen35.py` (the scan-based Gated-DeltaNet trunk, see
  [decider.md](decider.md)). The trunk takes an optional additive mask for its full-attention layers; pplx passes
  a key-padding mask (`lengths`), the other families keep the causal default. The readout gathers each marker's
  hidden state.
- **Padding.** Upstream left-pads; the graph right-pads to a multiple of 64. The linear-attention layers are causal,
  so right padding cannot reach real tokens, and the full-attention layers mask padded keys.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph) | derived, hosted by Ollaya |
| weights, BF16 | the model repo's `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`; the vision tower is unused |
| readout, BF16 | same repo, `readout.safetensors` |
| `tokenizer.json` | same repo, as-is |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya (temperature copied from `decision_config.json`) |
| license | `LICENSE` in the model repo: Apache-2.0 (see its `NOTICE`); the base is Apache-2.0 |

## Request → sequences

Source: `pplx_decider.protocol` (`options`, `Encoder.messages`, `Encoder.plan`). Port: `families/pplx/layout.py`,
mirrored by `crates/ollaya-decision/src/pplx.rs`.

```
text = pre "State:\n" esc(describe(state))
       for question i: "\n\nQuestion {i}:\n" esc(describe(instructions or default)) "\nOptions:\n"
                       "A: " esc(option_1) "\n" … "{code_k}: " esc(option_k) "\n<|fim_suffix|>"
       post
```

- `pre` / `post` are the chat template around the user text: the system prompt ("Classify the shared state for
  each numbered question. …"), the user turn and the generation prompt with thinking off.
- `describe` is the string itself, else `json.dumps(ensure_ascii=False)`. `esc` writes every special token (and the
  marker) with full-width brackets, so user text cannot produce one.
- **Options.** choice: `label`, or `label: describe(description)`, in the given order (1 to 255 options). score: the
  levels (2 to 255). noul: `false`, `true`, described by the criteria or "No / false" / "Yes / true" when a side is
  missing or empty; keys are case-sensitive, as upstream.
- **Instructions.** Optional upstream (default "Choose the best matching option."). Ollaya's parser requires the
  `instructions` field, as for every model; an empty or null value takes the default.
- **Window.** Ollaya runs at most 8,192 tokens per sequence (upstream's default is the backbone's 262,144: the graph
  materializes full attention). A longer request is split as upstream splits at its own limit: the questions are
  halved recursively, every part keeping the whole state; a question that does not fit alone is rejected. Requests
  under 8,192 tokens (nearly all) run exactly as upstream.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one sequence per row; `seq` a multiple of 64; right-pad with any id |
| `lengths` | int64 | `[rows]` | each row's real tokens |
| `markers` | int64 | `[rows, q]` | each question's marker position; pad short rows with 0 |
| `logits` | float32 | `[rows, q, 255]` | raw readout logits (temperature not applied); a question reads its first k |

Probabilities are `softmax(logits[:k] / T)` over each question's options. TypeSafe answers: noul is P(true), choice
the argmax, score the expected level. Upstream's own `confidence` formulas are not ported; Ollaya reports
TypeSafe's for every model.

## Parity (measured)

Goldens: `families/pplx/goldens.py`, upstream `pplx_decider` (its reference path, `Decider(fast=False)`) in fp32 on
CUDA with TF32 off, each sequence run alone: 138 requests (the shared Laya edge cases, the decoder cases, 50
typed-decisions rows and seven pplx cases: a request split in two, a question over the window, a long state, special
tokens in user text, falsy and non-string values, a one-level score, 40 questions), 5 of them rejected. The layout
port reproduces every sequence before a record is written. The export checks the graph on two right-padded rows
against upstream (5.7e-6).

`crates/ollaya-runner/examples/parity_pplx.rs` on choso-wsl (i9-13900K, RTX 4090, ONNX Runtime 1.28):

| | sequences | rejections | max \|Δ option logit\| | max \|Δ probability\| (p99) | answers vs upstream |
|---|---|---|---|---|---|
| CUDA | 138 / 138 identical (696 questions) | 0 mismatches | 8.2e-5 | 6.8e-6 (3.6e-6) | 0 differ (max 5.1e-5, the rounding) |
| CPU | 138 / 138 identical (696 questions) | 0 mismatches | 9.5e-5 | 9.5e-6 (2.7e-6) | 0 differ (max 5.1e-5, the rounding) |

Latency on the RTX 4090, five-question requests (474 tokens each, one sequence): p50 288 ms, p95 446 ms.

```sh
uv run python -m ollaya_convert.families.pplx.export pplx-decider-flash --out out/pplx-decider-flash
uv run python -m ollaya_convert.families.pplx.goldens pplx-decider-flash out/pplx-decider-flash --device cuda
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_pplx -- \
    convert/out/pplx-decider-flash convert/out/goldens-pplx-decider-flash.jsonl cuda --latency
cargo run --release -p ollaya-runner --example parity_pplx -- \
    convert/out/pplx-decider-flash convert/out/goldens-pplx-decider-flash.jsonl cpu
```

The CPU run takes 8 s per request on the i9-13900K (the goldens include sequences of up to 6,609 tokens). The reference needs
`torchvision` and `pillow` (upstream's processor loads its image side).

## Limits

- **Not ported.** Images (upstream supports them in `images`), and upstream's fast path (packed sequences, CUDA
  graphs), which matches the reference up to BF16 noise.
- **Window.** 8,192 tokens per sequence (see above).
- **Grouping.** The model card finds the flash model more accurate with one question per request on most
  multi-question tasks; Ollaya runs a request as sent.

## Upstream numbers (model card)

Decision Index 0.3 (public), overall: 58.88 with joint requests, 60.05 with one question per request (the 27B
pplx-decider-v1.2: 68.09). Vision Decision Index 0.3: 72.53. Development set (1,497 questions, joint, at the fitted
temperature): accuracy 88.8%, ECE 0.021.

## Attribution

pplx-decider-flash-v1.2 is by Perplexity AI, Apache-2.0. The base, Qwen3.5-4B, is by the Qwen team (Apache-2.0).
