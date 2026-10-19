Nimble is a decision model by [Bespoke Labs](https://huggingface.co/bespokelabs): a LoRA adapter on Qwen3.5-9B, trained on contrastive pairs (two examples that differ by one fact that flips the answer), released under Apache-2.0. For each question it reads the whole request as a JSON schema and scores every option by the next-token logit of its code (`A`, `B`, ...). It never generates text.

> Needs Ollaya 0.8.0 or newer, the first release that runs the `nimble-codes-v1` layout.

## Models

| Tag | Base | Checkpoint | Params | Typed-decisions accuracy |
|---|---|---|---|---|
| `nimble:latest`, `nimble:9b` | Qwen3.5-9B | Bespoke-Nimble-9B-v2 (2026-09-23) | 9B | {{result nimble:9b typed.accuracy 3}} |
| `nimble:9b-v3` | Qwen3.5-9B | Bespoke-Nimble-9B-v3 (2026-10) | 9B | {{result nimble:9b-v3 typed.accuracy 3}} |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya. `nimble:9b-v3`'s adapter is licensed CC BY-NC 4.0: it may not be used commercially. It runs as the author's Decision Index engine serves it: options without a description get none, rows run to 32,768 tokens, and probabilities are the plain softmax the author reads (no temperature is shipped). On Bespoke Labs' public benchmark (13 human-labeled datasets, 3,880 questions), see the [comparison with Ollama](/#vs-ollama).

## Usage

```shell
ollaya run nimble --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `nimble`.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** the triage preset (five questions) on a short message takes {{result nimble:9b gpu.p50_ms 0}} ms at the median (about 3,000 input tokens: each of the five rows carries the whole request).
- **State length:** every question's row repeats the whole request (the state and every question's schema), so the cost grows with the number of questions times the request length.
- **CPU:** a 9B model belongs on a GPU.

## How it works

- **Prompt.** Ollaya builds Nimble's prompt exactly as the author's code does (`serving_schema.prepare_prompts` in the model repository), and maps TypeSafe questions to Nimble's schema the way the author's own `/v1/systemone` server does: noul becomes a true/false field, choice an enum of its labels, score an enum of its levels.
- **Calibrated.** Probabilities use the author's recommended temperature, 2.179 (`temperature_config.json`). A plain softmax of the code logits, what Ollama returns, is overconfident.
- **Up to 255 options.** Questions with more than 26 options use the author's serving extension: two-letter codes and the "short code" prompt.
- **Weights.** Qwen's Qwen3.5-9B shards and Bespoke Labs' adapter download from Hugging Face, pinned to a commit and verified by sha256. The adapter is applied at run time, not merged, so every file stays byte for byte the authors'. Ollaya hosts only the ONNX graph (14 MB).
- **Parity.** Ollaya's Rust runtime matches the author's code in fp32 on CUDA: identical token rows, the same rejections and the same decision on all 492 test questions, probabilities within 6.5e-6. `nimble:9b-v3` matches the author's Decision Index engine the same way on 498 questions, probabilities within 1.1e-5.
- **fp16 tag.** `nimble:9b-fp16` computes the trunk in fp16 on NVIDIA GPUs, with the norms, the recurrent state, the attention softmax and the heads in fp32 ([how](https://github.com/ollaya-dev/ollaya/blob/main/docs/decisions/0008-fp16-decoder-tags.md)): {{result nimble:9b-fp16 gpu.p50_ms 0}} ms for five questions on the RTX 5090 instead of {{result nimble:9b gpu.p50_ms 0}} ms, {{result nimble:9b-fp16 parity.cuda.decisions_same 0}} of {{result nimble:9b-fp16 parity.cuda.questions 0}} decisions the same as `nimble:9b` on the parity set, and {{result nimble:9b-fp16 typed.accuracy 3}} on typed decisions. On a CPU use `nimble:9b`.

## Limits

- **Context.** Each question's row, the whole request included, can be up to 8,192 tokens; a longer one is rejected, never cut.
- **State.** The state must not be empty; instructions and choice labels must not be blank.
- **Memory.** The weights stay BF16 in memory: about 18 GB, so a 24 GB GPU. With `nimble:9b-v3`, a row of 10,000 tokens overflows a 24 GB GPU: it runs, slowly, and slows the model until it is unloaded.
