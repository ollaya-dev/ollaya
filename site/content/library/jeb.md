Jebadiah (Jeb) is a family of decision models by Jason Brashear, built with [AINode](https://github.com/getainode/ainode) and released under Apache-2.0: rank-16 LoRAs merged into Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B. The authors publish them as GGUF files, and Ollaya runs those files as they are, on llama.cpp. Jeb reads a question as AINode's decision prompt (the state, the question and lettered options) and returns the probability of each option label as the next token. It never generates text.

> Needs Ollaya 0.8.0 or newer, the first release that runs the `jebadiah-v1` prompt.

## Models

| Tag | Base | Weights | Five questions, RTX 5090 |
|---|---|---|---|
| `jeb:4b` | Qwen3.5-4B, v2 | Q8_0 GGUF, 4.5 GB | {{result jeb:4b gpu.p50_ms 0}} ms |
| `jeb:latest`, `jeb:9b` | Qwen3.5-9B, v2.1 | Q8_0 GGUF, 9.8 GB | {{result jeb:9b gpu.p50_ms 0}} ms |
| `jeb:27b` | Qwen3.8-27B, v2.1 | Q4_K_M GGUF, 17 GB | {{result jeb:27b gpu.p50_ms 0}} ms |

The 9B and 27B are v2.1, which the authors now name as the version to use, trained on about 90 sources; there is no 4B v2.1, so `jeb:4b` stays on v2. The 4B and 9B files are Q8_0 (for v2 the authors checked them against the bf16 weights: 256 and 257 of 260 held-out answers the same). The 27B is Q4_K_M so that it fits a 24 GB GPU. Until October 2026, `jeb:9b` and `jeb:27b` were v2. Latency is the triage preset (five questions) on a short message through the HTTP API, at the median of 15 warm requests.

On typed-decisions Jeb scores {{result jeb:4b typed.accuracy 3}} to {{result jeb:27b typed.accuracy 3}}, but its training data includes the typed-decisions train split, so that number is not comparable with the zero-shot models and the model list leaves it out. v2.1's sources also include civil_comments, HelpSteer2 and MASSIVE, which Ollaya Bench leaves out of its score. The authors publish their results in [getainode/jebadiah](https://github.com/getainode/jebadiah); their v2.1 numbers use a different recipe (the payload repeated in the prompt, choices read in both orders), which their own standalone scripts and Ollaya do not run.

## Usage

```shell
ollaya run jeb --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `jeb`, `jeb:4b` or `jeb:27b`.

## How it works

- **Prompt.** Ollaya builds the prompt exactly as the authors' renderer does (`scripts/jebadiah_prompt.py`, which is AINode's own `/v1/systemone` renderer): identical on all 1,349 test prompts, for v2 and v2.1.
- **Options.** A yes/no question reads `true` as `A` and `false` as `B`; a choice reads its options in order; a score its levels. Labels run `A`..`Z`, `AA`, `AB`, ..., up to 255 options.
- **Calibration.** The authors' fitted temperature per question type (`temperatures.json`).
- **One pass per question.** Qwen3.5's recurrent layers cannot share a cached state prefix, so each question is evaluated from the start. The same request always returns the same probabilities.
- **Engine.** llama.cpp v0.5.0, ggml-org's own build, on an NVIDIA GPU (CUDA), an Apple silicon GPU (Metal) or the CPU.
- **Parity.** Ollaya's runner matches stock llama.cpp (`llama-server` of the same build, on the same files) on CUDA: the same decision on all 494 test questions for each model, probabilities within 2.8e-6.

## Limits

- **Prompt.** Up to 2,048 tokens per question, the state included, the budget the authors serve with. A longer prompt is rejected; the authors' renderer cuts the state instead.
- **Instructions.** Must be a non-empty string.
- **Control tokens.** Text you send can never become one of Qwen's control tokens.
