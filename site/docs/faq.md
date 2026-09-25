---
title: FAQ
description: How Ollaya relates to Ollama and TypeSafe, hardware, privacy, where weights come from, and licensing.
order: 7
---

# FAQ

## What is a decision model?

A model that reads a *state* — text, an email, a ticket, a JSON object — plus typed questions, and returns a typed answer with calibrated probabilities for each question in a single forward pass. It never generates text. That makes it fast, and its output easy to act on: route the ticket, block the message, escalate when the probability is above a threshold.

## How do I install it?

The [desktop app](/download) for macOS, Windows and Linux; `curl -fsSL {{SITE_ORIGIN}}/install.sh | sh` for the command line on Linux and macOS, `irm {{SITE_ORIGIN}}/install.ps1 | iex` on Windows; or the Docker image `ghcr.io/ollaya-dev/ollaya`. See [Download](/download).

## How is Ollaya related to Ollama?

Ollaya borrows Ollama's experience — one binary, `pull`, `run`, `serve`, Modelfiles, a local REST API with the same conventions — and applies it to decision models instead of generative language models. It is an independent project, not affiliated with Ollama.

## How is it related to TypeSafe?

TypeSafe's closed Jev model created the decision-model category. Ollaya serves open models behind a TypeSafe-compatible API: the official TypeSafe Python SDK 0.7.1 works unchanged with `TYPESAFE_BASE_URL=http://localhost:11435` and any API key. Ollaya is not affiliated with TypeSafe. See [TypeSafe compatibility](/docs/typesafe-compatibility).

## Which models can I run?

Open decision models from these families. See [Models](/search).

- **`laya`** from Convai Innovations: `laya` (a router), `laya:en`, `laya:multilingual` and `laya:typed-decisions`, each also as `-fp16` and `-fp32`. `laya` sends English text to `laya:en` and other languages, Turkish for example, to `laya:multilingual`. It is the fastest.
- **`decider`** from Mapika: `decider:2b` and `decider:0.8b`, built on Qwen3.5. It is the most accurate, and the slowest.
- **`nli`** from Moritz Laurer: zero-shot NLI classifiers on DeBERTa-v3-large and ModernBERT-large. It is the most accurate encoder.
- **`gliclass`** from Knowledgator: an instruction-following zero-shot classifier that scores every option in one pass.
- **`kev`** from Jared Palmer: `kev:0.8b`, a LoRA and a pointer head on Qwen3.5-0.8B that scores every option at its own span, calibrated.
- **`qwen3guard`** from the Qwen team: a safety guard in 119 languages. It answers its own built-in questions (safe, controversial or unsafe, and the unsafe category), so you send it only the text.

## Where do the weights come from?

From the model authors' own Hugging Face repositories, pinned to a commit. Every file is checked against its sha256 when it is pulled. Ollaya never re-hosts weights: its registry only serves small manifests and derived files, such as the ONNX graphs, which reference the weights by URL. The same derived files are published on Hugging Face under [ollaya-dev](https://huggingface.co/ollaya-dev).

## Are the answers the same as the original model's?

Ollaya runs Laya as ONNX. Across 2,383 questions per checkpoint (`en`, `multilingual` and `typed-decisions`), the ONNX export chose the same answer as the PyTorch fp32 reference 100% of the time, with a maximum probability difference of 1.1 × 10⁻⁴. On a CUDA GPU the fp16 graph runs by default; it can differ from fp32 on near-ties. Pin a `-fp32` tag to match the reference.

## How fast is it?

A decision is a single forward pass. Measured end to end through the HTTP API on an RTX 4090, the median request with five questions takes 8 ms with `laya:multilingual` and 10 ms with `laya:en` (fp16), 15 ms with `gliclass` and 20 ms with `nli` (fp32). A single question takes 8–11 ms on any of them.

## Do I need a GPU?

No. Ollaya runs on the CPU, and on Linux x86-64 uses an NVIDIA GPU with driver R580 or newer (CUDA 13) when one is present. The installer downloads the CUDA libraries only when it finds a GPU.

## Which platforms are supported?

- **Linux** x86-64 and ARM64 with glibc 2.38 or newer: Ubuntu 24.04, Debian 13, Fedora 39, RHEL 10 or newer.
- **macOS** on Apple silicon.
- **Docker:** `ghcr.io/ollaya-dev/ollaya` for linux/amd64 and linux/arm64, and `:cuda` for NVIDIA GPUs. Use it on older Linux distributions too.
- **Windows** 10 and 11 on 64-bit x86 PCs, on the CPU: the desktop app, or `irm {{SITE_ORIGIN}}/install.ps1 | iex` for the command line. For an NVIDIA GPU, use WSL 2 with the Linux installer.
- **The desktop app** runs on all three: see [Download](/download).

## Does my data leave my machine?

No. The server listens on `127.0.0.1:11435` by default and runs the models locally. The network is used only to pull models. States and questions are never logged.

## Why port 11435?

It sits next to Ollama's default port, 11434, so both can run side by side.

## How are the probabilities calibrated?

With temperature scaling per question type and number of options, shipped with each model. Laya's expected calibration error is 0.081 after temperature fitting, against 0.246 for Jev. For thresholds you rely on, refit the temperatures on your own labelled data and bake them in with a [Modelfile](/docs/modelfile#calibration).

## Is there an MCP server or an agent skill?

Both. `ollaya mcp` serves the local models to Claude Code, Claude Desktop, Cursor and other MCP clients (`claude mcp add ollaya -- ollaya mcp`), and the `ollaya-decisions` skill teaches agents when and how to use them. See [Agents](/docs/agents).

## How do I uninstall it?

On Linux, after the installer set up the service:

```shell
sudo systemctl disable --now ollaya && sudo rm /etc/systemd/system/ollaya.service
sudo rm -rf /usr/local/bin/ollaya /usr/local/lib/ollaya /usr/local/share/doc/ollaya /usr/local/share/ollaya
sudo userdel -r ollaya    # also deletes /usr/share/ollaya, including the models
```

For an install without root (in `~/.local`), delete `~/.local/bin/ollaya`, `~/.local/lib/ollaya`, `~/.local/share/doc/ollaya` and `~/.local/share/ollaya`, and the models in `~/.ollaya`.

## What is the license?

Ollaya is Apache-2.0. Models carry their own licenses: `laya`, `decider`, `kev`, `qwen3guard`, `gliclass` and `nli:modernbert-large` are Apache-2.0, and `nli:deberta-v3-large` is MIT.
