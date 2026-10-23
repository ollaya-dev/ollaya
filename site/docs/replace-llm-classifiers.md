---
title: Replacing LLM classification calls with a local model
nav: Replacing LLM classifiers
description: Move the classify, triage and yes/no prompts you send to a hosted LLM onto a local decision model, with typed answers, calibrated probabilities and no per-token cost. Nothing leaves your machine.
order: 9
---

# Replacing LLM classification calls

Many LLM calls in production aren't asking for text. They ask for a label: which team gets this ticket, is this email spam, does this message ask for a refund. The prompt says "answer with one word", the code parses the reply, and every call costs tokens, a round trip and a copy of the data on someone else's server.

A decision model answers the same question as a typed answer with a probability for every option, in one forward pass, in milliseconds, on your own machine. It never generates text, so there is nothing to parse.

## Before and after

Before: a hosted LLM writes a word, and the code hopes it is one of the three.

```python
reply = client.chat.completions.create(
    model=LLM,
    messages=[
        {"role": "system", "content": "Classify the ticket as billing, technical or account. Answer with one word."},
        {"role": "user", "content": ticket},
    ],
)
team = reply.choices[0].message.content.strip().lower()
```

After: the same question, typed, to Ollaya.

```python
import requests

answers = requests.post("http://localhost:11435/api/decide", json={
    "model": "winnow:e4b",
    "state": ticket,
    "questions": {
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, invoices and refunds",
                "technical": "Bugs, errors and outages",
                "account": "Login, profile and settings",
            },
        },
    },
}).json()["answers"]
team = answers["team"]["choice"]              # always one of the three
confidence = answers["team"]["confidence"]
```

Code written for TypeSafe's Jev runs unchanged against `/v1/systemone`; see [TypeSafe compatibility](/docs/typesafe-compatibility).

## From prompt to question

| The prompt asks | Question type | The answer |
|---|---|---|
| "Pick one of these labels" | `choice` | The option, its confidence, and a probability for every option |
| "Is this X? Answer yes or no" | `noul` | The probability of yes |
| "Rate it from 1 to 5" | `score` | The expected level, and a probability for every level |

One request can ask several questions at once. Describe each option in `criteria` the way you would to a new colleague; the [API reference](/docs/api#decide) has every field.

## Check it on your data

Before you switch, run the examples you already have through both: past LLM answers, or labels from your team. A few hundred is enough to see the agreement.

```python
import json, requests

rows = [json.loads(line) for line in open("labeled.jsonl")]   # {"text": ..., "label": ...}
hits = 0
for row in rows:
    answers = requests.post("http://localhost:11435/api/decide", json={
        "model": "winnow:e4b", "state": row["text"], "questions": QUESTIONS,
    }).json()["answers"]
    hits += answers["team"]["choice"] == row["label"]
print(f"{hits / len(rows):.3f} agreement on {len(rows)} examples")
```

Where the model is unsure, use the probability: act on a confident answer, and send the rest to the LLM or a person.

## Which model

- **With an NVIDIA GPU:** `winnow:e4b` scores {{result winnow:e4b typed.accuracy 3}} on typed decisions (TypeSafe reports 0.738 for its Jev) and answers five questions in {{result winnow:e4b gpu.p50_ms 0}} ms on an RTX 5090. `pplx-decider:flash` is first on [Ollaya Bench](/results), at {{result pplx-decider:flash gpu.p50_ms 0}} ms.
- **On a CPU:** `laya` answers five questions in about half a second ({{result laya:multilingual cpu.p50_ms 0}} ms for `laya:multilingual` on a Threadripper 3960X). It is much less accurate than the 4B models on typed decisions, so the check above matters more.

## What stays with the LLM

A decision model picks between options you give it. Keep the LLM for anything that needs text written: summaries, replies, explanations, extracting a value such as an order number. Each model also reads a limited state; the Limits section of its page says how much.
