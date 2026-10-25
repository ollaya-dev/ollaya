Decision 2.0 is the vLLM Semantic Router contributors' second generation of decision models, released under Apache-2.0 with their code in each model repository. Each is a fully fine-tuned text backbone (Qwen3 for Kai, Qwen3.5 for the others) with an endpoint head: every option is scored at its own last token against the question, in one forward pass per question. It never generates text.

## Models

| Tag | Base | Params | Context | Ollaya Bench score |
|---|---|---|---|---|
| `decision2:kai` | Qwen3-0.6B-Base | 0.60B | 8,192 | {{result decision2:kai score 1}} |
| `decision2:eos` | Decision 1.0 Eos (Qwen3.5-0.8B) | 0.75B | 16,384 | {{result decision2:eos score 1}} |
| `decision2:sol` | Decision 1.0 Sol (Qwen3.5-2B) | 1.88B | 16,384 | {{result decision2:sol score 1}} |
| `decision2:nox` | Qwen3.5-4B-Base | 4.21B | 16,384 | {{result decision2:nox score 1}} |
| `decision2:lux` | Decision 1.0 Lux (Qwen3.5-9B) | 7.94B | 16,384 | {{result decision2:lux score 1}} |

The 27B Vega uses the same layout but is not in the library.

## Usage

```shell
ollaya run decision2 --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `decision2`.

## Accuracy

Measured by Ollaya on an RTX 5090, zero-shot as far as the model cards say (they do not list the training sets):

| Tag | typed-decisions (400 states) | Public benchmark, macro accuracy | Pooled ECE |
|---|---|---|---|
| `decision2:kai` | {{result decision2:kai typed.accuracy 3}} | {{result decision2:kai public.macro_accuracy 3}} | {{result decision2:kai public.pooled_ece 3}} |
| `decision2:eos` | {{result decision2:eos typed.accuracy 3}} | {{result decision2:eos public.macro_accuracy 3}} | {{result decision2:eos public.pooled_ece 3}} |
| `decision2:sol` | {{result decision2:sol typed.accuracy 3}} | {{result decision2:sol public.macro_accuracy 3}} | {{result decision2:sol public.pooled_ece 3}} |
| `decision2:nox` | {{result decision2:nox typed.accuracy 3}} | {{result decision2:nox public.macro_accuracy 3}} | {{result decision2:nox public.pooled_ece 3}} |
| `decision2:lux` | {{result decision2:lux typed.accuracy 3}} | {{result decision2:lux public.macro_accuracy 3}} | {{result decision2:lux public.pooled_ece 3}} |

## Speed

Five triage questions in one request, end to end over HTTP (Ollaya Bench), median:

| Tag | RTX 5090 | CPU (Threadripper 3960X) | GPU memory |
|---|---|---|---|
| `decision2:kai` | {{result decision2:kai gpu.p50_ms 0}} ms | {{result decision2:kai cpu.p50_ms 0}} ms | {{result decision2:kai gpu.vram_mb 0}} MB |
| `decision2:eos` | {{result decision2:eos gpu.p50_ms 0}} ms | {{result decision2:eos cpu.p50_ms 0}} ms | {{result decision2:eos gpu.vram_mb 0}} MB |
| `decision2:sol` | {{result decision2:sol gpu.p50_ms 0}} ms | {{result decision2:sol cpu.p50_ms 0}} ms | {{result decision2:sol gpu.vram_mb 0}} MB |
| `decision2:nox` | {{result decision2:nox gpu.p50_ms 0}} ms | {{result decision2:nox cpu.p50_ms 0}} ms | {{result decision2:nox gpu.vram_mb 0}} MB |
| `decision2:lux` | {{result decision2:lux gpu.p50_ms 0}} ms | {{result decision2:lux cpu.p50_ms 0}} ms | {{result decision2:lux gpu.vram_mb 0}} MB |

Every question is a forward pass of its own, so a request costs about one question's time per question.

## How it works

- **One row per question.** The state, the question and each option in its own segment; the head reads every option's last token against the row's last token. Questions do not see each other.
- **Temperature 1.** No package ships a calibration file, so the probabilities are the softmax of the option logits, as upstream computes them. Kai adds the author's fixed offsets to five-level score logits first.
- **Weights.** The files download from Hugging Face, pinned to a commit and verified by sha256, byte for byte the authors'. Ollaya hosts only the ONNX graph. The 4B and 9B models keep their BF16 weights in BF16 in memory.
- **Parity.** Ollaya's Rust runtime matches the author's own code in fp32 for every tag: identical token rows, the same rejections and the same decision on every test question, on CUDA and on the CPU.

## Limits

- **Context.** 8,192 tokens per question for Kai, 16,384 for the others, the state included; a longer question is rejected, never truncated.
- **Options.** 2 to 255 choice options, 2 to 10 score levels.
- **Not ported.** `share_context` (one forward pass for all of a request's questions) and the GGUF builds.
