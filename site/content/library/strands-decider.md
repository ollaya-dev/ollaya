Strands Decider is a decision model by [Strands Agents](https://huggingface.co/StrandsAgents): a LoRA adapter on a base decoder (Qwen3.5-2B base, or Gemma 4 E2B and E4B) plus a small pointer head, released under Apache-2.0 with its code at [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider). For each question it lays out the state, the question and one line per option, then scores every option at the end of its own line against the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Checkpoint | Params | JevBench public (upstream) |
|---|---|---|---|---|
| `strands-decider:latest`, `strands-decider:2b` | Qwen3.5-2B base | v1, 2026-10 (`strands-decider-2B-qwen3.5-v1-2610`) | 2B | 180 / 231 tasks, ECE 0.072 |
| `strands-decider:e2b` | Gemma 4 E2B (instruction-tuned) | v1, 2026-10-09 (`strands-decider-E2B-gemma4-v1-2610`) | 4.6B weights read (effective 2B) | 180 / 231 tasks, ECE 0.053 |
| `strands-decider:e4b` | Gemma 4 E4B (instruction-tuned) | v1, 2026-10-09 (`strands-decider-E4B-gemma4-v1-2610`) | 8.9B weights read (effective 4B) | 187 / 231 tasks, ECE 0.037 |

The JevBench numbers are from the model cards, served at a 4,096-token window. Half of a Gemma 4 model's weights are its two embedding tables, which the model only looks up, hence "effective" sizes. Measured by Ollaya on an RTX 5090: {{result strands-decider:2b public.macro_accuracy 3}} macro accuracy on Bespoke Labs' public benchmark (3,880 questions, pooled ECE {{result strands-decider:2b public.pooled_ece 3}}, {{result strands-decider:2b public.median_ms 0}} ms per question) and {{result strands-decider:2b typed.accuracy 3}} on the 400 typed-decisions states.

## Usage

```shell
ollaya run strands-decider --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `strands-decider`.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** five-question triage requests take {{result strands-decider:2b gpu.p50_ms 0}} ms at the median.
- **State length:** every question reads the whole state again, so the cost grows with the number of questions times the state length.

## How it works

- **One row per question.** The state comes first, then the question with a header for its type and one numbered line per option, then `<answer>`. The pointer head compares the last token of each option's line with the last token of the row.
- **Calibrated.** The scores are divided by the temperature the authors fitted for each question type: 0.86 / 1.83 / 1.07 (choice / score / noul) for `2b`, 0.56 / 1.48 / 1.07 for `e2b`, 0.77 / 0.86 / 0.59 for `e4b`.
- **Weights.** From Hugging Face, pinned to a commit and verified by sha256: the base from Qwen or Google, the adapter and `head.safetensors` from Strands Agents. The adapter is applied at run time, not merged; Ollaya hosts only the ONNX graph (10 MB). The Gemma 4 tags read only the text decoder of the checkpoint (its vision and audio towers stay untouched) and keep the weights in memory as BF16, about 11 GB for `e2b` and 20 GB for `e4b`.
- **Parity.** Ollaya's Rust runtime matches upstream `strands_decider` in fp32: identical token rows and option positions, the same rejections and the same decision on all 650 test questions, probabilities within 2.8e-6 for `2b`.; for the Gemma tags the same 650 decisions, probabilities within 1.2e-5 for `e2b` and 5.2e-6 for `e4b`.
- **fp16 tag.** `strands-decider:2b-fp16` runs the trunk in fp16 on NVIDIA GPUs ([how](https://github.com/ollaya-dev/ollaya/blob/main/docs/decisions/0008-fp16-decoder-tags.md)): {{result strands-decider:2b-fp16 gpu.p50_ms 0}} ms for five questions on the RTX 5090 instead of {{result strands-decider:2b gpu.p50_ms 0}} ms, {{result strands-decider:2b-fp16 parity.cuda.decisions_same 0}} of {{result strands-decider:2b-fp16 parity.cuda.questions 0}} decisions the same as `strands-decider:2b` on the parity set, and {{result strands-decider:2b-fp16 typed.accuracy 3}} on typed decisions. On a CPU use `strands-decider:2b`.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of 2 to 255 options (`{"refund": null, ...}`). A plain list is rejected, as upstream does. Scores take 2 to 10 levels.
- **Context.** A row is fitted to 4,096 tokens: a long question keeps its end, a long state its beginning. A request whose cut would remove an option is rejected.
- **Not ported.** Images (upstream's `--vision` path); the 12B and 26B-A4B Gemma 4 variants; fp16 tags of the Gemma 4 models.
