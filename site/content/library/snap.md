snap1-2b by [logitlab](https://huggingface.co/logitlab/snap1-2b-GGUF) is MiniCPM5-2B fine-tuned for one job: a typed decision read from the option letters after one prompt. It is the model of [snap](https://github.com/emnlmn/snap), emnlmn's decision engine, and it is trained on snap's exact prompt. Ollaya reads the probability of each option letter as the next token. It never generates text.

> Needs Ollaya 0.10.0 or newer, the first release that runs the `snap-v1` prompt.

## Models

| Tag | Weights | Typed-decisions accuracy | Five questions, RTX 5090 |
|---|---|---|---|
| `snap:latest`, `snap:2b` | snap1-2b (MiniCPM5-2B), Q8_0 GGUF, 2.7 GB | {{result snap:2b typed.accuracy 3}} | {{result snap:2b gpu.p50_ms 0}} ms |
| `snap:2b-q4_k_m` | the same model, Q4_K_M GGUF (snap's own default file), 1.6 GB | {{result snap:2b-q4_k_m typed.accuracy 3}} | {{result snap:2b-q4_k_m gpu.p50_ms 0}} ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya, with a calibration error (ECE) of {{result snap:2b typed.ece 3}} on snap's raw probabilities ({{result snap:2b-q4_k_m typed.ece 3}} for Q4_K_M; the two files are a few questions apart either way, as the author also found). The time is the median five-question request of Ollaya Bench through the HTTP API. The author reports 0.655 with snap itself on the same file: the prompts are identical, and snap reads the answer letters slightly differently (below). The author also reports that the checkpoint never trained on typed-decisions.

## Usage

```shell
ollaya run snap --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `snap`.

## How it works

- **Prompt.** Built exactly as snap 0.5.0 builds it, token for token on all 573 test prompts. A structured state is written in TOON, snap's compact JSON; the question goes before or after the state, as snap decides for the whole request.
- **Options.** Each option is a letter, up to 26 per question, read in one pass. A yes/no question reads `A) Yes`, `B) No`.
- **Calibration.** None: snap1-2b ships raw probabilities, and so does Ollaya.
- **Engine.** llama.cpp v0.5.0, ggml-org's own build, on an NVIDIA GPU (CUDA), an Apple silicon GPU (Metal) or the CPU. About 3 GB of memory.
- **Parity.** Ollaya's runner matches stock llama.cpp (`llama-server` of the same build, on the same file, with snap's own prompt tokens) on the CPU and on CUDA: the same decision on all 573 test questions, probabilities within 1.9e-6.

## Limits

- **Options.** Up to 26 per question. snap reads a wider choice as one yes/no question per option; Ollaya answers it with `TOO_MANY_OPTIONS`.
- **Readout.** Ollaya reads each letter's own token in one pass per question; snap also pools the letter's space- and newline-prefixed tokens and reads the state once for all questions.
- **Context.** 8,192 tokens per question; a longer prompt is rejected, not cut.
- **Languages.** English and Italian.
