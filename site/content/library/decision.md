Decision 1.0 is a family of decision models by the [vLLM Semantic Router](https://huggingface.co/llm-semantic-router) contributors, released under Apache-2.0. Each model is a fully fine-tuned Qwen3.5 text backbone plus a small endpoint head. For each question it lays out the context, the question and one segment per option, then scores every option at its last token against the end of the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Params | Decision Index 0.2 | ECE |
|---|---|---|---|---|
| `decision:latest`, `decision:eos` | Qwen3.5-0.8B | 0.75B | 17.49 | 0.083 |

[Decision Index 0.2](https://huggingface.co/spaces/multimodalart/jev-decision-index) is an independent comparison of open decision models with Jev. The score is its balanced skill across 43 benchmarks, where Jev scores 51.67, and ECE is its calibration error. For comparison, `kev:0.8b` scores 13.26 and `decider:2b` 26.11 there. The larger members, Nox (4B, 31.06) and Lux (9B, 38.98), use the same layout; see the limits below.

## Usage

```shell
ollaya run decision --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `decision`.

## Speed

- **RTX 4090, end to end:** a five-question request with a short message (700 tokens in all) takes about 217 ms at the median, close to `kev` and `decider:0.8b`.
- **CPU:** the same request takes about 0.85 s on a Mac mini (M4 Pro), through the daemon.
- **State length:** every question reads the whole context again, so the cost grows with the number of questions times the context length. Five typed-decisions questions about a JSON state (about 1,050 tokens in all) take about 0.2 s on the RTX 4090, 0.84 s on the Mac mini and 2.1 s on a 24-core x86 CPU, measured in the runtime without HTTP.

## How it works

- **One row per question.** The context comes first, then the question type and the question, then one `<option>` segment per option, then a fixed decision prompt. Each segment is tokenized on its own, exactly as upstream does.
- **Endpoint head.** The head compares each option's last token with the row's last token, through a bilinear term and a small MLP, in fp32.
- **Calibrated.** The logits are divided by the temperature the model ships (1.039 for Eos).
- **Weights.** Two files download from Hugging Face, pinned to a commit and verified by sha256: the BF16 backbone and the FP32 head, both from the author's repository and read in place. Ollaya hosts only the ONNX graph.
- **Parity.** Ollaya's Rust runtime matches the author's code run in fp32: identical token rows and option positions, the same decision on every test question, and probabilities within 3.2e-6, on CPU (x86 and Apple silicon) and CUDA.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of option to description, with 2 to 255 options. A plain list is rejected, as upstream does.
- **Score.** 2 to 10 levels, as a list.
- **Noul criteria.** Only the keys `true` and `false`, spelled exactly.
- **Context.** A question's row, context included, can be up to 16,384 tokens. Nothing is truncated: a longer row rejects the request. Every question reads the whole context again.
- **Calibration.** The author notes that probabilities can be overconfident.
- **Larger models.** Nox (4B) and Lux (9B) use the same layout, but Ollaya computes in fp32 and their weights alone then take about 17 GB and 32 GB, so they are not in the library.
