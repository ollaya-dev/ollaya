pplx-decider is Perplexity's family of decision models, released under Apache-2.0 with their code in each model repository. Ollaya runs the flash model: Qwen3.5-4B post-trained by Perplexity, distilled from its 27B decision models, with a readout over 255 answer codes. A request is one sequence: the state, then every question with its options, each followed by a marker where the readout picks the answer. It never generates text.

## Models

| Tag | Base | Checkpoint | Params | Decision Index 0.3 (upstream) |
|---|---|---|---|---|
| `pplx-decider:latest`, `pplx-decider:flash` | Qwen3.5-4B | flash v1.2, 2026-10 (`pplx-decider-flash-v1.2`) | 4B | 58.9 joint, 60.1 one question per request |

The Decision Index numbers are from the model card. The 27B models (`pplx-decider-v1.2`, `-v1.1-27b`) use the same layout but need about 54 GB of memory in Ollaya's fp32 runtime, so they are not in the library.

## Usage

```shell
ollaya run pplx-decider --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `pplx-decider`.

## Speed

- **RTX 5090, over HTTP (Ollaya Bench):** {{result pplx-decider:flash gpu.p50_ms 0}} ms at the median for five triage questions (one forward pass), {{result pplx-decider:flash public.median_ms 0}} ms per question on Bespoke Labs' public benchmark and {{result pplx-decider:flash typed.median_ms 0}} ms per five-question typed-decisions request.

## Accuracy

Measured by Ollaya on an RTX 5090: {{result pplx-decider:flash public.macro_accuracy 3}} macro accuracy on Bespoke Labs' public benchmark (3,880 human-labeled questions, pooled ECE {{result pplx-decider:flash public.pooled_ece 3}}) and {{result pplx-decider:flash typed.accuracy 3}} on the 400 typed-decisions states. Its public score is the highest of any model Ollaya or Ollama 0.40 serves, Ollama's 27B pplx-decider included ({{result ollama/pplx-decider public.macro_accuracy 3}} public, {{result ollama/pplx-decider typed.accuracy 3}} typed). See the [comparison with Ollama](/#vs-ollama).

## How it works

- **One sequence per request.** Every question sees the state and the other questions. The model card finds the flash model more accurate with one question per request on most multi-question tasks: send related questions together and independent ones apart.
- **Calibrated.** The readout logits are divided by the temperature Perplexity fitted on held-out requests (1.55).
- **Weights.** The files download from Hugging Face, pinned to a commit and verified by sha256, byte for byte Perplexity's. Ollaya hosts only the ONNX graph (11 MB).
- **Parity.** Ollaya's Rust runtime matches upstream `pplx_decider` in fp32: identical token sequences, the same rejections and the same decision on all 696 test questions, probabilities within 6.8e-6.
- **fp16 tag.** `pplx-decider:flash-fp16` runs the trunk in fp16 on NVIDIA GPUs ([how](https://github.com/ollaya-dev/ollaya/blob/main/docs/decisions/0008-fp16-decoder-tags.md)): {{result pplx-decider:flash-fp16 gpu.p50_ms 0}} ms for five questions on the RTX 5090 instead of {{result pplx-decider:flash gpu.p50_ms 0}} ms, {{result pplx-decider:flash-fp16 parity.cuda.decisions_same 0}} of {{result pplx-decider:flash-fp16 parity.cuda.questions 0}} decisions the same as `pplx-decider:flash` on the parity set, and {{result pplx-decider:flash-fp16 typed.accuracy 3}} on typed decisions. On a CPU use `pplx-decider:flash`.

## Limits

- **Context.** 8,192 tokens per sequence. A longer request is split by its questions, every part keeping the whole state; a question that does not fit alone is rejected.
- **Not ported.** Images (upstream accepts them).
