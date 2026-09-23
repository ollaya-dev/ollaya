---
title: TypeSafe compatibility
nav: TypeSafe compatibility
description: Use existing TypeSafe SDKs and code with local, open models by changing one environment variable.
order: 5
---

# TypeSafe compatibility

TypeSafe's closed Jev model created the "System One" category of decision models. Ollaya is wire-compatible with TypeSafe's API, so code written for it can run against open models on your own machine.

## Point your SDK at Ollaya

TypeSafe SDKs read their base URL from `TYPESAFE_BASE_URL`. Set it to your local server:

```shell
export TYPESAFE_BASE_URL=http://localhost:11435
```

Then set `model` in your requests to a local model such as `laya`, and run your code as usual.

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/systemone` | Decide: same request and response shape as TypeSafe's endpoint |
| `POST /v1/decisions` | Alias of `/v1/systemone` |
| `GET /v1/models` | The models available on this machine |

## Request and response

The request carries the model, the state and the typed questions:

```shell
curl http://localhost:11435/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "state": "Can I get an invoice for last month?",
    "questions": {
      "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
          "invoice": "Needs an invoice or receipt",
          "refund": "Wants money back",
          "other": "Anything else"
        }
      }
    }
  }'
```

```json
{
  "model": "laya",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.9,
      "probabilities": { "invoice": 0.933, "refund": 0.021, "other": 0.046 }
    }
  },
  "usage": { "input_tokens": 52, "output_tokens": 0 }
}
```

All three question types — `choice`, `score` and `noul` — are supported with the same fields. `/v1/systemone` returns exactly TypeSafe's shape. The native [`/api/decide`](/docs/api#decide) takes the same body (plus an optional `keep_alive`) and adds `routing` and timing fields to the response.

## What is different

Compatibility covers the API, not the model. Answers come from open models, so quality differs from Jev's by task:

- `laya:typed-decisions` scores 0.766 on typed-decisions, against 0.727 published for Jev 1.13.
- The base Laya checkpoints are near chance zero-shot on typed-decisions (0.362).
- Choice questions with many options (more than ~20) are weaker: 0.425 on Banking77, against 0.870 for Jev.

Measure on your own data before switching production traffic. The [Laya readme](/library/laya) has the details.

## Not affiliated

Ollaya is an independent open-source project. It is not affiliated with or endorsed by TypeSafe.
