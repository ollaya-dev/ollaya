---
title: FAQ
description: Release status, how Ollaya relates to Ollama and TypeSafe, hardware, privacy and licensing.
order: 6
---

# FAQ

## Can I download Ollaya today?

Not yet. Ollaya is pre-release: the runtime is being built right now and nothing is downloadable. Star or watch the [GitHub repository](https://github.com/cobanov/ollaya) to hear when the first version ships.

## What is a decision model?

A model that reads a *state* — text, an email, a ticket, a JSON object — plus typed questions, and returns a typed answer with calibrated probabilities for each question in a single forward pass. It never generates text. That makes it fast and its output easy to act on: route the ticket, block the message, escalate when the probability is above a threshold.

## How is Ollaya related to Ollama?

Ollaya borrows Ollama's experience — one binary, `pull`, `run`, `serve`, Modelfiles, a local REST API — and applies it to decision models instead of generative language models. It is an independent project, not affiliated with Ollama.

## How is it related to TypeSafe?

TypeSafe's closed Jev model created the decision-model category. Ollaya serves open models behind a TypeSafe-compatible API, so existing TypeSafe SDKs work by setting `TYPESAFE_BASE_URL`. Ollaya is not affiliated with TypeSafe. See [TypeSafe compatibility](/docs/typesafe-compatibility).

## Which models can I run?

The Laya family from Convai Innovations: `laya` (a router), `laya:en`, `laya:multilingual` and `laya:typed-decisions`. See [Models](/search).

More open decision models are planned: von, GLiClass, NLI zero-shot classifiers, and GGUF LLM-based decision models via llama.cpp.

## Do I need a GPU?

No. Ollaya runs models with ONNX Runtime on the CPU, and uses CUDA on NVIDIA GPUs and Core ML on macOS when available. Models without a precision suffix use fp16 on a GPU and fp32 on CPU.

## Which platforms are supported?

Linux and macOS come first, plus a Docker image.

## Does my data leave my machine?

No. The server listens on `127.0.0.1:11435` by default and runs the model locally. The network is only used to pull models from the registry.

## Why port 11435?

It sits next to Ollama's default port, 11434, so both can run side by side.

## How are the probabilities calibrated?

With temperature scaling per question type and number of options. Laya's expected calibration error is 0.081 after temperature fitting, against 0.246 for Jev. The raw checkpoints are over-confident, so for thresholds you rely on, refit calibration on your own labelled data — see [Modelfile](/docs/modelfile#calibration).

## What is the license?

Ollaya is Apache-2.0. Models carry their own licenses; the Laya family is Apache-2.0 too.
