Winnow is a pair of open decision models by EldanRing, fine-tuned from Google DeepMind's Gemma 4 and released under Apache-2.0. The author publishes them as GGUF files, and Ollaya runs those files as they are, on llama.cpp. Winnow puts the state and one question into its own prompt, labels the options `A`, `B`, `C`, and reads the probability of each label as the next token. It never generates text.

## Models

| Tag | Base | Weights | JevBench public (231) | Kev v9 clean (1,046) |
|---|---|---|---|---|
| `winnow:latest`, `winnow:12b` | Gemma 4 12B IT | Q8_0 GGUF, 12.7 GB | 85.7 % | 81.5 % |
| `winnow:e4b` | Gemma 4 E4B IT | Q8_0 GGUF, 8.0 GB | 80.5 % | 72.7 % |

The accuracies are the author's, measured with the author's server on the same Q8_0 files (model cards of [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) and [Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B)). For comparison, the author reports 85.7 % and 87.0 % for Jev 1.13 on the same two sets. On typed-decisions (all 400 states, argmax against the majority label), measured by Ollaya, `winnow:12b` scores 0.702 and `winnow:e4b` 0.722.

## Usage

```shell
ollaya run winnow --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `winnow` or `winnow:e4b`.

## Speed

- **RTX 4090, in the runner:** five questions with a short state take about 154 ms on `12b` and 96 ms on `e4b` at the median, without the HTTP layer.
- **CPU:** `e4b` takes about 5.1 s for the same request on a 24-core x86 CPU. Use a GPU.

## How it works

- **Prompt.** Ollaya builds Winnow's prompt exactly as the author's server does (`native/protocol.h` in [winnow-inference](https://github.com/EldanRing/winnow-inference)): a system turn, the state as JSON, then the question and its options, each option under a letter label. Text you send can never become one of Gemma's control tokens.
- **One question at a time.** The state is evaluated once per request, and each question after it. The split is fixed, so the same request always returns the same probabilities.
- **Labels.** Up to 64 options per question: `A` to `Z`, then two-letter labels, each a single token.
- **Calibration.** `winnow:12b` uses temperature 1, the author's default: the author fitted no calibration for it. `winnow:e4b` uses 1.2574, the temperature the author fitted for its Q8_0 file on 778 held-out questions.
- **Engine.** llama.cpp v0.5.0, ggml-org's own release build, runs inside Ollaya's runner process. It uses an NVIDIA GPU (CUDA) or an Apple silicon GPU (Metal) when the model fits, and the CPU otherwise.
- **Parity.** Ollaya's runner matches stock llama.cpp (`llama-server` of the same build, on the same file): the same decision on all 505 test questions, probabilities within 3.0e-6, on CUDA (RTX 4090, both models) and an x86-64 CPU (`e4b`). On Windows the CPU check covered 15 questions. Against the author's own server, `e4b` agrees on all 503 decisions. Not checked yet: Metal and the Apple CPU, Linux on ARM, and `12b` on the CPU.

## Limits

- **Size.** These are large language models. `winnow:12b` needs about 14 GB of GPU memory and `winnow:e4b` about 9 GB at the 8,192-token context; on the CPU they are much slower than the encoder models.
- **Context.** State, question and options share 8,192 tokens. A longer state is cut to 6,144 tokens.
- **Options.** 2 to 64 options per question and up to 256 questions per request. Probabilities are conditional on the options offered.
- **Text only.** The author's server can also read images through a vision projector; Ollaya runs the text decisions only.

## Weights and license

Apache-2.0. Winnow is developed by EldanRing ([Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B), [Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B), inference code at [github.com/EldanRing/winnow-inference](https://github.com/EldanRing/winnow-inference)), from Gemma 4 by Google DeepMind, also Apache-2.0. Ollaya downloads the GGUF from the author's repository, pinned to a commit and checked against its sha256; it never re-hosts it. llama.cpp is MIT.
