---
title: Open-source Jev alternative: which open model on which hardware
nav: Jev alternative
description: Replace TypeSafe's hosted Jev with an open decision model on your own machine, behind the same API. Which model to run on a 4, 8 or 12 GB GPU or a CPU, with measured accuracy, calibration and speed.
order: 5.5
---

# Open-source Jev alternative

TypeSafe's Jev is a closed model behind a hosted API. Open decision models answer the same typed questions on your own machine, and Ollaya serves them behind TypeSafe's API, so code written for Jev switches over with three environment variables. The closest open models come within about a point of Jev on typed decisions, and you choose one by the memory you have.

## Which model on which hardware

| Your machine | Model | Typed decisions | Calibration error | Five questions | GPU memory |
|---|---|---|---|---|---|
| A 12 GB GPU or more | [`pplx-decider:flash`](/library/pplx-decider) | {{result pplx-decider:flash typed.accuracy 3}} | {{result pplx-decider:flash typed.ece 3}} | {{result pplx-decider:flash gpu.p50_ms 0}} ms | {{result pplx-decider:flash gpu.vram_mb 0}} MB |
| An 8 GB GPU | [`winnow:e4b`](/library/winnow), the recommended model | {{result winnow:e4b typed.accuracy 3}} | {{result winnow:e4b typed.ece 3}} | {{result winnow:e4b gpu.p50_ms 0}} ms | {{result winnow:e4b gpu.vram_mb 0}} MB |
| A 6 GB GPU | [`d1:3b`](/library/d1) | {{result d1:3b typed.accuracy 3}} | {{result d1:3b typed.ece 3}} | {{result d1:3b gpu.p50_ms 0}} ms | {{result d1:3b gpu.vram_mb 0}} MB |
| A 4 GB GPU | [`snap:2b-q4_k_m`](/library/snap) | {{result snap:2b-q4_k_m typed.accuracy 3}} | {{result snap:2b-q4_k_m typed.ece 3}} | {{result snap:2b-q4_k_m gpu.p50_ms 0}} ms | {{result snap:2b-q4_k_m gpu.vram_mb 0}} MB |

TypeSafe reports 0.738 for Jev on typed decisions. Accuracy is on typed decisions' 2,000 questions, and calibration error is its expected calibration error (lower means the probabilities can be trusted as they are). Times are medians through the HTTP API on Ollaya Bench's RTX 5090. `d1:3b` is under the LFM Open License, which limits commercial use by larger companies: read it before you ship it.

**On a CPU only**, the two small models answer five questions in {{result snap:2b-q4_k_m cpu.p50_ms 0}} ms (`snap:2b-q4_k_m`) and {{result d1:3b cpu.p50_ms 0}} ms (`d1:3b`) on a Threadripper 3960X, and `winnow:e4b` in {{result winnow:e4b cpu.p50_ms 0}} ms. For answers in under a second, with less accuracy, use an encoder: see [classification on a CPU](/docs/cpu-classification).

Models trained on typed decisions' own train split, such as `jeb` and `laya:typed-decisions`, score higher on it, but on questions like the ones they learned from, so they are left out here. [Models](/search) compares all of them.

## Switch

Pull the model, then point the TypeSafe SDK, or any Jev client, at Ollaya:

```shell
ollaya pull winnow:e4b
export TYPESAFE_BASE_URL=http://localhost:11435
export TYPESAFE_API_KEY=local              # any value, unless the server sets OLLAYA_API_KEY
export TYPESAFE_DEFAULT_MODEL=winnow:e4b
```

Your requests and response parsing stay as they are. [TypeSafe compatibility](/docs/typesafe-compatibility) has the details, and [Integrations](/docs/integrations) covers LiteLLM and LangChain.

## What changes

- **Your data stays on your machine.** States never leave it, and there is no per-request price.
- **The model does not change under you.** Weights come from the author's Hugging Face repository, pinned to a commit and checked against their sha256, and Ollaya's runtime is checked for parity with the author's code.
- **Shorter context and fewer options.** An open model reads fewer tokens than Jev, and choice questions with many options (more than about 20) are weaker. See [what is different](/docs/typesafe-compatibility#what-is-different).
- **Quality differs by task.** Measure on your own data before you move production traffic.

Ollaya is an independent open-source project, not affiliated with or endorsed by TypeSafe.
