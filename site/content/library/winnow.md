Winnow is a family of open decision models by EldanRing, fine-tuned from Google DeepMind's Gemma 4 and released under Apache-2.0. The author publishes them as GGUF files, and Ollaya runs those files as they are, on llama.cpp. Winnow puts the state and one question into its own prompt, labels the options `A`, `B`, `C`, and reads the probability of each label as the next token. It never generates text.

> Needs Ollaya 0.7.0 or newer, which runs GGUF models through llama.cpp. Update first (`ollaya --version`); an older version downloads the weights and then fails to load them.

## Models

| Tag | Base | Weights | JevBench public (231) | Kev v9 clean (1,046) |
|---|---|---|---|---|
| `winnow:latest`, `winnow:12b` | Gemma 4 12B IT | Q8_0 GGUF, 12.7 GB | 85.7 % | 81.5 % |
| `winnow:e4b` | Gemma 4 E4B IT | Q8_0 GGUF, 8.0 GB | 80.5 % | 72.7 % |
| `winnow:e2b` | Gemma 4 E2B IT | Q8_0 GGUF, 5.0 GB | 75.8 % | 69.6 % |
| `winnow:e4b-vision` | Gemma 4 E4B IT, with the vision projector | Q8_0 GGUF, 8.0 GB, and a 990 MB projector | – | – |
| `winnow:12b-vision` | Gemma 4 12B IT, with the vision projector | Q8_0 GGUF, 12.7 GB, and a 175 MB projector | – | – |

The accuracies are the author's, measured with the author's server on the same Q8_0 files (model cards of [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B) and [Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B); for E2B, winnow-inference's `docs/EVALUATION.md`). The [Winnow-E2B](https://huggingface.co/EldanRing/Winnow-E2B) card leads with higher numbers from the author's adaptive client, which reasons before it answers; Ollaya does not run that client, so the table gives E2B's direct scores. For comparison, the author reports 85.7 % and 87.0 % for Jev 1.13 on the same two sets. On typed-decisions (all 400 states, argmax against the majority label), measured by Ollaya, `winnow:12b` scores {{result winnow:12b typed.accuracy 3}}, `winnow:e4b` {{result winnow:e4b typed.accuracy 3}} and `winnow:e2b` {{result winnow:e2b typed.accuracy 3}}. With its speed, that makes `winnow:e4b` Ollaya's recommended model; `winnow:e2b` is the smallest and fastest, and the least accurate.

## Usage

```shell
ollaya run winnow:e4b --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run winnow:e2b --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run winnow --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `winnow`, `winnow:e4b` or `winnow:e2b`.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** the triage preset (five questions) on a short customer message takes {{result winnow:e2b gpu.p50_ms 0}} ms on `e2b`, {{result winnow:e4b gpu.p50_ms 0}} ms on `e4b` and {{result winnow:12b gpu.p50_ms 0}} ms on `12b` at the median.
- **CPU:** `e2b` takes {{result winnow:e2b cpu.p50_ms 0}} ms and `e4b` {{result winnow:e4b cpu.p50_ms 0}} ms for the same request on the Threadripper 3960X (24 cores). Use a GPU.

## How it works

- **Prompt.** Ollaya builds Winnow's prompt exactly as the author's server does (`native/protocol.h` in [winnow-inference](https://github.com/EldanRing/winnow-inference)): a system turn, the state as JSON, then the question and its options, each option under a letter label. Text you send can never become one of Gemma's control tokens.
- **One question at a time.** The state is evaluated once per request, and each question after it. The split is fixed, so the same request always returns the same probabilities.
- **Labels.** Up to 64 options per question: `A` to `Z`, then two-letter labels, each a single token.
- **Calibration.** `winnow:12b` uses temperature 1, the author's default: the author fitted no calibration for it. `winnow:e4b` uses 1.2574, the temperature the author fitted for its Q8_0 file on 778 held-out questions. `winnow:e2b` uses temperature 1, at which the author scores it: no calibration was fitted.
- **Engine.** llama.cpp v0.5.0, ggml-org's own release build, runs inside Ollaya's runner process. It uses an NVIDIA GPU (CUDA) or an Apple silicon GPU (Metal) when the model fits, and the CPU otherwise.
- **Parity.** Ollaya's runner matches stock `llama-server` of the same build on the same file: the same decision on all 505 test questions, probabilities within 3.0e-6, on CUDA (all three models) and an x86-64 CPU (`e4b` and `e2b`). Against the author's own server, `e4b` agrees on all 503 decisions. Not checked yet: Metal, the Apple CPU, Linux on ARM and the full `12b` text suite on the CPU.

## Limits

- **Size.** `winnow:12b` needs about 14 GB of GPU memory and `winnow:e4b` about 9 GB at the 8,192-token context; on the CPU both are much slower than the encoders. `winnow:12b-vision` needs about 0.4 GB more than `winnow:12b`, so a 16 GB GPU is marginal for it.
- **Context.** State, question and options share 8,192 tokens. A longer state is cut to 6,144 tokens.
- **Options.** 2 to 64 options per question and up to 256 questions per request. Probabilities are conditional on the options offered.
- **Images.** The vision tags read images: up to 16 PNGs per request, sharing the 8,192-token context with the state. `/v1/*` and the other tags are text only.

## Image decisions

`winnow:e4b-vision` is `winnow:e4b` with the author's matching vision projector (990 MB): it answers questions about PNG images as well as the state. It needs Ollaya 0.10.0 or newer.

`winnow:12b-vision` is `winnow:12b` with its own matching projector (175 MB), read the same way. It also needs Ollaya 0.10.0 or newer, and more GPU memory than `winnow:12b` (see Limits).

```shell
ollaya run winnow:e4b-vision --image shelf.png --questions '{"blocked":{"type":"noul","instructions":"Is the aisle blocked?"}}' "A photo from the warehouse camera."
```

- **How.** Winnow's own image prompt, with the images in order before the state, tokenized and evaluated as stock llama.cpp does, and the same option-label readout as the text tags. The images and the state are evaluated once per request.
- **Parity.** On Linux, the image path matches stock llama-server of the pinned build on 21 requests and 65 questions, with option logits within 1.2e-5 (the gate is 1e-3): `winnow:e4b-vision` on an RTX 4070, an RTX 5090 and the CPU, `winnow:12b-vision` on an RTX 5090 and the CPU.
- **Not measured yet.** Image calibration (each model's text temperature is kept), visual accuracy, Metal and Windows image parity.

## Weights and license

Apache-2.0. Winnow is developed by EldanRing ([Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B), [Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B), [Winnow-E2B](https://huggingface.co/EldanRing/Winnow-E2B), inference code at [github.com/EldanRing/winnow-inference](https://github.com/EldanRing/winnow-inference)), from Gemma 4 by Google DeepMind, also Apache-2.0. Ollaya downloads the GGUF from the author's repository, pinned to a commit and checked against its sha256; it never re-hosts it. llama.cpp is MIT.
