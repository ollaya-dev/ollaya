---
title: FAQ
description: How Ollaya relates to Ollama and TypeSafe, hardware, privacy, where weights come from, and licensing.
order: 7
---

# FAQ

## What is a decision model?

A model that reads a *state* (text, an email, a ticket, a JSON object) plus typed questions, and returns a typed answer with calibrated probabilities for each question in a single forward pass. It never generates text. That makes it fast, and its output easy to act on: route the ticket, block the message, escalate when the probability is above a threshold.

## How do I install it?

The [desktop app](/download) for macOS, Windows and Linux; `curl -fsSL {{SITE_ORIGIN}}/install.sh | sh` for the command line on Linux and macOS, `irm {{SITE_ORIGIN}}/install.ps1 | iex` on Windows; or the Docker image `ghcr.io/ollaya-dev/ollaya`. See [Download](/download).

## How is Ollaya related to Ollama?

Ollaya borrows Ollama's experience (one binary, `pull`, `run`, `serve`, Modelfiles, a local REST API with the same conventions) and applies it to decision models instead of generative language models. It is an independent project, not affiliated with Ollama.

## Ollama runs decision models now. How is Ollaya different?

Ollama 0.35 (September 2026) added `/v1/systemone` for two decoder models, Bespoke Labs' Nimble and Together AI's Tev1. Both projects follow TypeSafe's wire format, and for decoder models both read the answer labels' next-token scores. Ollaya runs Nimble too, since 0.8.0. This comparison is with Ollama 0.35:

| | Ollaya | Ollama 0.35 |
|---|---|---|
| Models | 15 families: encoders (`laya`, `nli`, `gliclass`, `von`) and decoders (`winnow`, `kev`, `decider`, `nimble`, `jeb`, `jeeves`, `cygnet` and more) | Nimble (9B) and Tev1 (4B, 0.8B) |
| Encoders | Read every question in one forward pass. `laya:en` answers five questions in 8 to 10 ms on an RTX 4090 | Not supported |
| Probabilities | Calibrated with each model's fitted temperatures, which you can refit on your own data in a [Modelfile](/docs/modelfile#calibration) | Softmax of the raw label scores. Ollama documents `confidence` as uncalibrated |
| Limits | TypeSafe's: 1 to 256 questions, 2 to 255 options, 2 to 10 score levels | 1 to 64 questions, 2 to 26 options and score levels, a 64 KiB request body |
| API | TypeSafe's `/v1/systemone`, `/v1/decisions` and `/v1/models`; `/api/decide` with routing and timings; an MCP server | `/v1/systemone` |
| Routers | `laya` picks the English or the multilingual model for each request | None |
| Weights | Pulled unmodified from the author's Hugging Face repository, pinned to a commit and checked by sha256 | Converted to GGUF and served from Ollama's registry |

**Measured side by side.** On one RTX 5090, over Bespoke Labs' public benchmark (3,880 human-labeled questions from 13 datasets, scored with Bespoke's own code): `winnow:12b` on Ollaya scores 0.773 at 60 ms per question, against 0.749 at 210 ms for Nimble on Ollama, Ollama's best. On the same Nimble weights the accuracy is the same (0.748 and 0.749), the calibration error is 5.5 times lower on Ollaya (0.022 against 0.122, because Ollaya applies the author's temperature), and Ollama is faster (210 against 310 ms: it runs a Q8_0 GGUF where Ollaya computes in fp32). The [home page](/#vs-ollama) has the chart.

Ollama's advantages are real too. Tev1 is there and not here yet (Together AI has not published a license for its weights), and if you already use Ollama for language models, one daemon covers both. The two run side by side: Ollama on port 11434, Ollaya on 11435.

## How is it related to TypeSafe?

TypeSafe's closed Jev model created the decision-model category. Ollaya serves open models behind a TypeSafe-compatible API: the official TypeSafe Python SDK 0.7.1 works unchanged with `TYPESAFE_BASE_URL=http://localhost:11435` and any API key. Ollaya is not affiliated with TypeSafe. See [TypeSafe compatibility](/docs/typesafe-compatibility).

## Which models can I run?

Open decision models from these families. See [Models](/search), which compares their accuracy and speed.

- **`winnow`** from EldanRing: Winnow-E4B (`winnow:e4b`, the recommended model) and Winnow-12B (`winnow`), Gemma 4 fine-tunes published as GGUF files. Ollaya runs the author's file on llama.cpp, on an NVIDIA GPU, Apple silicon's GPU or the CPU. `winnow:e4b` scores 0.722 on typed decisions, close to Jev's 0.738, in about 90 ms on an RTX 4090.
- **`laya`** from Convai Innovations: `laya` (a router), `laya:en`, `laya:multilingual` and `laya:typed-decisions`, each also as `-fp16` and `-fp32`. `laya` sends English text to `laya:en` and other languages, Turkish for example, to `laya:multilingual`. It is the fastest.
- **`decider`** from Mapika: `decider:4b`, `decider:2b` and `decider:0.8b`, built on Qwen3.5. `decider:4b` scores 0.680 on typed decisions. `decider:2b-vision` also reads an image with the state.
- **`nli`** from Moritz Laurer: zero-shot NLI classifiers on DeBERTa-v3-large and ModernBERT-large. It is the most accurate encoder.
- **`gliclass`** from Knowledgator: an instruction-following zero-shot classifier that scores every option in one pass.
- **`kev`** from Jared Palmer: `kev:4b` (`kev`), `kev:0.8b` and `kev:9b`, a LoRA and a pointer head on Qwen3.5 that scores every option at its own span, calibrated. `kev:9b` scores 0.722 on typed decisions, as much as `winnow:e4b`, but needs a 24 GB GPU and takes about 500 ms.
- **`decision`** from the vLLM Semantic Router contributors: `decision:eos`, Decision 1.0 Eos, a fully fine-tuned Qwen3.5-0.8B with an endpoint head that scores every option at its last token, calibrated, with rows of up to 16,384 tokens.
- **`qwen3guard`** from the Qwen team: a safety guard in 119 languages. It answers its own built-in questions (safe, controversial or unsafe, and the unsafe category), so you send it only the text.
- **`von`** from Victor Hugo Panisa: Von 1.1 on ModernBERT-large, which scores every option at its own marker in one pass and reads states of up to 8,192 tokens.
- **`clm`** from Contrastive-LM: CLM-v0.1-8B, which embeds the state and each option with Qwen3-8B and picks the option closest to the state through two trained heads. Questions and options are cached, so repeated ones are nearly free. It scores 0.357 on typed decisions; its authors built it for agent, game and tool-calling states.
- **`jevk5`** from alibiserikbay: JevK5 v0.3, a Qwen3.5-4B fine-tune published as GGUF files. Ollaya runs the author's 4B Q8_0 file on llama.cpp, with up to 16 options per question.

## Where do the weights come from?

From the model authors' own Hugging Face repositories, pinned to a commit. Every file is checked against its sha256 when it is pulled. Ollaya never re-hosts weights: its registry only serves small manifests and derived files, such as the ONNX graphs, which reference the weights by URL. The same derived files are published on Hugging Face under [ollaya-dev](https://huggingface.co/ollaya-dev).

## Are the answers the same as the original model's?

Ollaya runs Laya as ONNX. Across 2,383 questions per checkpoint (`en`, `multilingual` and `typed-decisions`), the ONNX export chose the same answer as the PyTorch fp32 reference 100% of the time, with a maximum probability difference of 1.1 × 10⁻⁴. On a CUDA GPU the fp16 graph runs by default; it can differ from fp32 on near-ties. Pin a `-fp32` tag to match the reference.

## How do the models compare with Jev?

It depends on the model. On typed decisions (2,000 questions), `winnow:e4b` and `kev:9b` come closest, with 0.722 against 0.738 for Jev, and `winnow:e4b` answers five questions in about 90 ms on an RTX 4090, faster than Jev's hosted API. The small encoders, such as `laya`, are the fastest but fall well below Jev on harder questions. The [home page](/#fast) lists every model's accuracy and speed. For an independent comparison of open decision models with Jev, on accuracy and calibration, see the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index). If you have labelled data for a fixed task, a model fine-tuned on it usually beats any general one.

## How fast is it?

A decision is a single forward pass. Measured end to end through the HTTP API on an RTX 4090, the median request with five questions takes 8 ms with `laya:multilingual` and 10 ms with `laya:en` (fp16), 15 ms with `gliclass`, 20 ms with `nli` and 89 ms with `winnow:e4b`. A single question takes 8–11 ms on the encoders. The larger decoders take longer: `kev:4b` 354 ms, `decider:4b` 520 ms.

## Do I need a GPU?

No. Ollaya runs on the CPU, and on x86-64 Linux and Windows uses an NVIDIA GPU with driver R525 or newer when one is present (CUDA 13 libraries from R580 on, CUDA 12 before that). The install scripts download the CUDA libraries only when they find a GPU. The Windows and Linux desktop apps bundle no CUDA libraries: on their own they run models on the CPU, and when the command line is installed too with its GPU libraries (and is as new as the app), the app starts the server from that install, which uses the GPU. GGUF models such as `winnow` run on llama.cpp, which also uses the GPU of Apple silicon Macs (Metal); their parity has been checked on CUDA and the x86-64 CPU, not yet on Metal. They are large language models, so a GPU makes a much bigger difference for them than for the encoder models: see each model's page for measured speeds. On an RTX 30, 40 or 50 series card GGUF models need nothing more. On older or data-center cards (GTX 10 series, V100, T4, A100, H100) llama.cpp's CUDA libraries carry code the driver compiles on first use, which takes a newer driver: R570 or newer with the CUDA 12 libraries, and a driver for CUDA 13.4 or newer with the CUDA 13 libraries. With an older driver Ollaya runs GGUF models on the CPU and logs why; `ollaya llama-devices` shows each GPU's compute capability and whether the kernels run on it.

## Which platforms are supported?

- **Linux** x86-64 and ARM64 with glibc 2.38 or newer: Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer.
- **macOS** 14 or newer on Apple silicon. `laya` and `nli:modernbert-large` run on the Apple GPU through MLX, 2 to 3 times faster than on the CPU; the other models run on the CPU.
- **Docker:** `ghcr.io/ollaya-dev/ollaya` for linux/amd64 and linux/arm64, and `:cuda` for NVIDIA GPUs (`:cuda12` for host drivers older than R580). Use it on older Linux distributions too.
- **Windows** 10 and 11 on 64-bit x86 PCs: the desktop app, and `irm {{SITE_ORIGIN}}/install.ps1 | iex` for the command line, which also uses an NVIDIA GPU (install both and the app's server uses it too). WSL 2 with the Linux installer works too.
- **The desktop app** runs on all three: see [Download](/download).

## Does my data leave my machine?

No. The server listens on `127.0.0.1:11435` by default and runs the models locally. The network is used only to pull models. States and questions are never logged.

## Why port 11435?

It sits next to Ollama's default port, 11434, so both can run side by side.

## How are the probabilities calibrated?

With temperature scaling per question type and number of options, shipped with each model. For thresholds you rely on, refit the temperatures on your own labelled data and bake them in with a [Modelfile](/docs/modelfile#calibration).

## Is there an MCP server or an agent skill?

Both. `ollaya mcp` serves the local models to Claude Code, Claude Desktop, Cursor and other MCP clients (`claude mcp add ollaya -- ollaya mcp`), and the `ollaya-decisions` skill teaches agents when and how to use them. See [Agents](/docs/agents).

## How do I update it?

Run `ollaya update`. It checks the latest release and, when there is a newer one, runs the install script again into the same place, which keeps your models and the service settings. `ollaya update --check` only tells you whether an update exists. The desktop app updates as a whole: install the new version from [Download](/download). In Docker, pull the new image.

## How do I uninstall it?

On Linux, after the installer set up the service:

```shell
sudo systemctl disable --now ollaya && sudo rm /etc/systemd/system/ollaya.service
sudo rm -rf /usr/local/bin/ollaya /usr/local/lib/ollaya /usr/local/share/doc/ollaya /usr/local/share/ollaya
sudo userdel -r ollaya    # also deletes /usr/share/ollaya, including the models
```

For an install without root (in `~/.local`), delete `~/.local/bin/ollaya`, `~/.local/lib/ollaya`, `~/.local/share/doc/ollaya` and `~/.local/share/ollaya`, and the models in `~/.ollaya`.

## What is the license?

Ollaya is Apache-2.0. Models carry their own licenses: `laya`, `decider`, `kev`, `decision`, `qwen3guard`, `gliclass`, `von`, `winnow`, `jevk5`, `clm` and `nli:modernbert-large` are Apache-2.0, and `nli:deberta-v3-large` is MIT.
