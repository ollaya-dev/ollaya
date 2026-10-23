---
title: coba: Ollaya's own decision model, in your browser
date: 2026-10-23
summary: Ollaya 0.17.0 ships coba, the first decision model we trained ourselves: a fast encoder with the best typed-decisions score among encoders under 1B parameters not trained on that task, small enough to run in a browser tab.
---

Ollaya 0.17.0 ships coba, the first decision model we trained ourselves. Like every decision model, it reads a state and typed questions (yes/no, choice, score) and answers each with probabilities in a single pass, without generating text.

coba is an encoder, so it is fast: `coba:400m` answers five questions in {{result coba:400m gpu.p50_ms 0}} ms on an RTX 5090. LLM-based decision models around 0.8B take {{result decider:0.8b gpu.p50_ms 0}}–{{result decision:eos gpu.p50_ms 0}} ms. Its ternary sizes are small enough for a browser, and <a href="{{SITE_ORIGIN}}/try">{{SITE_HOST}}/try</a> runs `coba:150m-ternary` in the page: what you type never leaves it.

## Where it stands

On Ollaya Bench 2026.10, `coba:400m` has the best typed-decisions score of any encoder under 1B parameters not trained on that dataset, and `coba:1b` is higher still. It is not the most accurate decision model overall: LLM-based models around 0.8B score higher on the whole benchmark, though not on typed decisions.

| Model | Overall score | Typed decisions | Five questions, RTX 5090 / CPU |
|---|---|---|---|
| `coba:1b` | {{result coba:1b score 1}} | {{result coba:1b tasks.typed-decisions 1}} | {{result coba:1b gpu.p50_ms 0}} / {{result coba:1b cpu.p50_ms 0}} ms |
| `coba:400m` | {{result coba:400m score 1}} | {{result coba:400m tasks.typed-decisions 1}} | {{result coba:400m gpu.p50_ms 0}} / {{result coba:400m cpu.p50_ms 0}} ms |
| `coba:150m-ternary`, in-domain | – | {{result coba:150m-ternary tasks.typed-decisions 1}} | {{result coba:150m-ternary gpu.p50_ms 0}} / {{result coba:150m-ternary cpu.p50_ms 0}} ms |
| `coba:68m-ternary`, in-domain | – | {{result coba:68m-ternary tasks.typed-decisions 1}} | {{result coba:68m-ternary gpu.p50_ms 0}} / {{result coba:68m-ternary cpu.p50_ms 0}} ms |
| `laya:en` | {{result laya:en score 1}} | {{result laya:en tasks.typed-decisions 1}} | {{result laya:en gpu.p50_ms 0}} / {{result laya:en cpu.p50_ms 0}} ms |
| `decider:0.8b`, an LLM | {{result decider:0.8b score 1}} | {{result decider:0.8b tasks.typed-decisions 1}} | {{result decider:0.8b gpu.p50_ms 0}} / {{result decider:0.8b cpu.p50_ms 0}} ms |

Scores are chance-corrected accuracy: 0 is chance, 100 is every answer right. The overall score is the mean over 14 tasks, 13 public ones and typed decisions. The ternary tags are in-domain: they were trained on the train splits of most public tasks, so they have no comparable overall score. Typed decisions is not among those tasks. CPU times are on a Threadripper 3960X. Every number, with its raw data, is on [Ollaya Bench](/results) and the [coba page](/library/coba).

## How it is made

coba runs Laya's decision network, which Ollaya already ran, on Ettin encoders. It learned from `pplx-decider:flash`'s probabilities, on permissively licensed data only, and never saw typed decisions' test set. The ternary tags are trained quantization-aware from the full-precision model of the same size. coba is English only.

## Try it

- **In your browser:** <a href="{{SITE_ORIGIN}}/try">{{SITE_HOST}}/try</a>
- **On your machine,** with Ollaya 0.17.0: `ollaya run coba:400m --ask "Is the customer asking for a refund?" "I was charged twice for order A-104."`
- **In your own page:** `npm install @ollaya/web` runs the ternary tags with onnxruntime-web.
- **The weights:** [huggingface.co/ollaya-dev](https://huggingface.co/ollaya-dev)

## Next

sura is Ollaya's decoder family, built on Qwen3.5 and distilled from `pplx-decider`, for when accuracy matters more than speed. It is in training: [sura-0.8b](https://huggingface.co/ollaya-dev/sura-0.8b) and [sura-2b](https://huggingface.co/ollaya-dev/sura-2b).
