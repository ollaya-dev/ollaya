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

- **RTX 4090, in the runtime:** five-question requests of about 470 tokens take 288 ms at the median (446 ms at p95). All questions share one forward pass.
- **RTX 5090, over HTTP:** 232 ms per question at the median on Bespoke Labs' public benchmark, 380 ms per five-question typed-decisions request.

## Accuracy

Measured by Ollaya on an RTX 5090: 0.824 macro accuracy on Bespoke Labs' public benchmark (3,880 human-labeled questions, pooled ECE 0.016) and 0.726 on the 400 typed-decisions states. Its public score is the highest of any model Ollaya or Ollama 0.40 serves, Ollama's 27B pplx-decider included (0.816 public, 0.702 typed). See the [comparison with Ollama](/#vs-ollama).

## How it works

- **One sequence per request.** Every question reads the state and the other questions: the model's full-attention layers see the whole request, not only what comes before. The model card finds the flash model more accurate with one question per request on most multi-question tasks, so send related questions together and independent ones apart.
- **Calibrated.** The readout logits are divided by the temperature Perplexity fitted on held-out requests (1.55).
- **Weights.** The files download from Hugging Face, pinned to a commit and verified by sha256, byte for byte Perplexity's. Ollaya hosts only the ONNX graph (11 MB).
- **Parity.** Ollaya's Rust runtime matches upstream `pplx_decider` in fp32: identical token sequences, the same rejections and the same decision on all 696 test questions, probabilities within 6.8e-6.

## Limits

- **Context.** 8,192 tokens per sequence. A longer request is split by its questions, every part keeping the whole state; a question that does not fit alone is rejected.
- **Not ported.** Images (upstream accepts them).
