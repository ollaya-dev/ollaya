---
title: Zero-shot classification on a CPU
nav: Classification on a CPU
description: Classify text into labels you choose per request, on a CPU with no GPU, in under a second. Which open models to run, and how fast and accurate they are, measured.
order: 9.5
---

# Zero-shot classification on a CPU

For classification on a CPU, use an encoder decision model: `laya:multilingual` answers five questions in {{result laya:multilingual cpu.p50_ms 0}} ms and `decima:small` in {{result decima:small cpu.p50_ms 0}} ms, with labels you write in each request and no training. The LLM-based decision models are more accurate, but take seconds on a CPU.

## Which model

| Model | Typed decisions | Public benchmark | Five questions, CPU | Memory |
|---|---|---|---|---|
| [`laya:multilingual`](/library/laya) | {{result laya:multilingual typed.accuracy 3}} | {{result laya:multilingual public.macro_accuracy 3}} | {{result laya:multilingual cpu.p50_ms 0}} ms | {{result laya:multilingual cpu.ram_peak_mb 0}} MB |
| [`decima:small`](/library/decima) | {{result decima:small typed.accuracy 3}} | {{result decima:small public.macro_accuracy 3}} | {{result decima:small cpu.p50_ms 0}} ms | {{result decima:small cpu.ram_peak_mb 0}} MB |
| [`laya:en`](/library/laya) | {{result laya:en typed.accuracy 3}} | {{result laya:en public.macro_accuracy 3}} | {{result laya:en cpu.p50_ms 0}} ms | {{result laya:en cpu.ram_peak_mb 0}} MB |
| [`decima:base`](/library/decima) | {{result decima:base typed.accuracy 3}} | {{result decima:base public.macro_accuracy 3}} | {{result decima:base cpu.p50_ms 0}} ms | {{result decima:base cpu.ram_peak_mb 0}} MB |
| [`nli:deberta-v3-large`](/library/nli) | {{result nli:deberta-v3-large typed.accuracy 3}} | {{result nli:deberta-v3-large public.macro_accuracy 3}} | {{result nli:deberta-v3-large cpu.p50_ms 0}} ms | {{result nli:deberta-v3-large cpu.ram_peak_mb 0}} MB |
| [`gliclass:large`](/library/gliclass) | {{result gliclass:large typed.accuracy 3}} | {{result gliclass:large public.macro_accuracy 3}} | {{result gliclass:large cpu.p50_ms 0}} ms | {{result gliclass:large cpu.ram_peak_mb 0}} MB |
| [`snap:2b-q4_k_m`](/library/snap), an LLM | {{result snap:2b-q4_k_m typed.accuracy 3}} | {{result snap:2b-q4_k_m public.macro_accuracy 3}} | {{result snap:2b-q4_k_m cpu.p50_ms 0}} ms | {{result snap:2b-q4_k_m cpu.ram_peak_mb 0}} MB |

Medians through the HTTP API, five triage questions per request, on Ollaya Bench's CPU, a 24-core Threadripper 3960X; memory is the peak. Typed decisions is 2,000 typed questions; the public benchmark is Bespoke Labs', scored as macro accuracy.

- **Speed:** `laya:multilingual` and `decima:small`, both multilingual.
- **English:** `laya:en` is trained for guardrails and email triage.
- **Accuracy among encoders:** `nli:deberta-v3-large` scores highest on typed decisions, and takes {{result nli:deberta-v3-large cpu.p50_ms 0}} ms.
- **Many labels:** the time of `nli` and `gliclass` grows with the number of labels ([gliclass](/library/gliclass) has the measurements).
- **Hard questions:** the encoders fall well below the LLM-based models. If a second or two per request is fine, `snap:2b-q4_k_m` is far more accurate; see [Jev alternative](/docs/jev-alternative) for the GPU models.

Times depend on your CPU. `/api/decide` reports `eval_duration` with every answer, so you can measure on yours.

## Classify

Labels go in the request, as the `criteria` of a choice question:

```shell
ollaya pull laya:multilingual
ollaya run laya:multilingual \
  --questions '{"topic":{"type":"choice","instructions":"What is the message about?","criteria":["billing","login","shipping","other"]}}' \
  "Mi pedido lleva dos semanas sin llegar."
```

The same through the API:

```shell
curl http://localhost:11435/v1/systemone -H "Authorization: Bearer local" -d '{
  "model": "laya:multilingual",
  "state": "Mi pedido lleva dos semanas sin llegar.",
  "questions": {"topic": {"type": "choice", "instructions": "What is the message about?", "criteria": ["billing", "login", "shipping", "other"]}}
}'
```

Each label gets a probability, so you can send the uncertain answers to a person or a larger model. Labels can carry a description too: `"criteria": {"shipping": "Late, lost or damaged deliveries", ...}`.

To replace an LLM prompt that classifies today, see [Replace LLM classifiers](/docs/replace-llm-classifiers), which has a script to measure agreement with your own labels.
