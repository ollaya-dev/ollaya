Strands Decider is a decision model by [Strands Agents](https://huggingface.co/StrandsAgents): a LoRA adapter on Qwen3.5-2B base plus a small pointer head, released under Apache-2.0 with its code at [strands-labs/strands-decider](https://github.com/strands-labs/strands-decider). For each question it lays out the state, the question and one line per option, then scores every option at the end of its own line against the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Checkpoint | Params | JevBench public (upstream) |
|---|---|---|---|---|
| `strands-decider:latest`, `strands-decider:2b` | Qwen3.5-2B base | v1, 2026-10 (`strands-decider-2B-qwen3.5-v1-2610`) | 2B | 180 / 231 tasks, ECE 0.072 |

The JevBench number is from the model card, served at a 4,096-token window. Measured by Ollaya on an RTX 5090: {{result strands-decider:2b public.macro_accuracy 3}} macro accuracy on Bespoke Labs' public benchmark (3,880 questions, pooled ECE {{result strands-decider:2b public.pooled_ece 3}}, {{result strands-decider:2b public.median_ms 0}} ms per question) and {{result strands-decider:2b typed.accuracy 3}} on the 400 typed-decisions states.

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
- **Calibrated.** The scores are divided by the temperature the authors fitted for each question type: 0.86 for choice, 1.83 for score and 1.07 for noul.
- **Weights.** The files download from Hugging Face, pinned to a commit and verified by sha256: the base weights from Qwen's repository, the adapter and `head.safetensors` from Strands Agents'. The adapter is applied at run time, not merged, so every file stays byte for byte the authors'. Ollaya hosts only the ONNX graph (10 MB).
- **Parity.** Ollaya's Rust runtime matches upstream `strands_decider` in fp32: identical token rows and option positions, the same rejections and the same decision on all 650 test questions, probabilities within 2.8e-6.
- **fp16 tag.** `strands-decider:2b-fp16` computes the trunk in fp16 on NVIDIA GPUs, with the norms, the recurrent state, the attention softmax and the heads in fp32 ([how](https://github.com/ollaya-dev/ollaya/blob/main/docs/decisions/0008-fp16-decoder-tags.md)): {{result strands-decider:2b-fp16 gpu.p50_ms 0}} ms for five questions on the RTX 5090 instead of {{result strands-decider:2b gpu.p50_ms 0}} ms, {{result strands-decider:2b-fp16 parity.cuda.decisions_same 0}} of {{result strands-decider:2b-fp16 parity.cuda.questions 0}} decisions the same as `strands-decider:2b` on the parity set, and {{result strands-decider:2b-fp16 typed.accuracy 3}} on typed decisions. On a CPU use `strands-decider:2b`.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of 2 to 255 options (`{"refund": null, ...}`). A plain list is rejected, as upstream does. Scores take 2 to 10 levels.
- **Context.** A row is fitted to 4,096 tokens: a long question keeps its end, a long state its beginning. A request whose cut would remove an option is rejected.
- **Not ported.** Images (upstream's `--vision` path).
