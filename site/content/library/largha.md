Largha by [Phocinae](https://huggingface.co/Phocinae) is a small decision model: Laya's decision network on mmBERT-small (144M parameters), fine-tuned on the typed-decisions workflows. It reads a *state* and typed questions and returns calibrated probabilities for every question in one forward pass. It never generates text.

> Needs Ollaya 0.15.0 or newer.

## Models

| Tag | Backbone | Params | Context | Languages |
|---|---|---|---|---|
| `largha:latest`, `largha:150m` | mmBERT-small | 144M | 512 | 100+ (the author evaluated English and Chinese) |

`largha:150m` carries an fp16 and an fp32 graph over the author's own weights file. It loads fp16 on a CUDA GPU and fp32 on the CPU; `largha:150m-fp16` and `largha:150m-fp32` pin one. On Apple silicon it runs on the GPU through MLX.

## Usage

```shell
ollaya run largha --preset triage "I was charged twice for my subscription this month. Please refund the second charge."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `largha`.

## Accuracy and speed

Measured by Ollaya on an RTX 5090: {{result largha:150m public.macro_accuracy 3}} macro accuracy on Bespoke Labs' public benchmark (3,880 human-labeled questions, pooled ECE {{result largha:150m public.pooled_ece 3}}), and {{result largha:150m gpu.p50_ms 0}} ms for five questions through the HTTP API ({{result largha:150m cpu.p50_ms 0}} ms on the CPU of the same machine).

Largha has no typed-decisions score here. It was fine-tuned on that dataset's train split, and by its card its training also saw 500 of the 2,000 test decisions, so its typed-decisions score is not a held-out measurement. The author reports 0.5455 (126 of 231) on JevBench's public items.

## How it works

- **Network.** The same network as `laya`: the question, a `[MASK]` before every option, and the state go into one sequence, and a two-layer head scores each option at its `[MASK]`. The author's `rl_agent_config.json` sets the budget (512 tokens, 192 for the question and its options) and the three temperatures Ollaya applies.
- **Parity.** Ollaya's runtime matches laya's own loader on the author's v1.1 checkpoint in fp32 on {{result largha:150m parity.cuda.questions 0}} questions, on the CPU, on CUDA and on the Apple GPU: identical token ids and markers, the same decision on every question, probabilities within 3.5e-6. The fp16 graph agrees on {{result largha:150m-fp16 parity.cuda.decisions_same 0}} of {{result largha:150m-fp16 parity.cuda.questions 0}} decisions; the others are near-ties.

## Limits

- **State.** 512 tokens including the question. `/v1/systemone` answers a longer state with `STATE_TRUNCATED`; `/api/decide` answers from its first tokens.
- **Options.** Every option's `[MASK]` has to fit in the 512-token sequence; long options are shortened to share the 192-token question budget, and options that still don't fit return `TOO_MANY_OPTIONS`.
- **Calibration.** Ollaya applies the author's three temperatures. The repository's `calib/` power transform is not applied.

## Weights and license

Apache-2.0. Largha is by Phocinae; its base encoder, [jhu-clsp/mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small), is MIT. Ollaya downloads the weights from [Phocinae/Phocinae-Largha-150M-v1](https://huggingface.co/Phocinae/Phocinae-Largha-150M-v1), pinned to the `v1.1` commit and checked against sha256; it never re-hosts them.
