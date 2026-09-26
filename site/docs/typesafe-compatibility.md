---
title: TypeSafe compatibility
nav: TypeSafe compatibility
description: Use the official TypeSafe SDK and existing code with local, open models by changing environment variables.
order: 5
---

# TypeSafe compatibility

TypeSafe's closed Jev model created the "System One" category of decision models. Ollaya's `/v1/*` API is wire-identical to TypeSafe's, as defined by the wire schema and error handling of `typesafe-sdk` 0.7.1, so code written for TypeSafe runs against open models on your own machine.

## Point the SDK at Ollaya

The official TypeSafe Python SDK 0.7.1 works unchanged. Set these environment variables:

```shell
export TYPESAFE_BASE_URL=http://localhost:11435
export TYPESAFE_API_KEY=local           # the SDK needs a non-empty key; any value works
export TYPESAFE_DEFAULT_MODEL=laya      # otherwise the SDK sends its default, "jev-latest"
export NO_PROXY=localhost,127.0.0.1    # keep local requests off any system proxy
```

- **API key.** Ollaya accepts any key, unless the server sets `OLLAYA_API_KEY`; then the SDK's key must match it.
- **Request IDs.** Every response carries `x-typesafe-request-id`, so `response.request_id` works.
- **Retries.** The SDK times out after 10 s and retries. The first request to a model waits while it loads, and a load that outlives the request continues, so the retry finds the model warm.
- **System proxies.** On a Mac with a system HTTP proxy, the TypeSafe SDK (like `httpx`) sends requests for `localhost` through the proxy too, ignoring the system's exception list. Your states then pass through the proxy, and while Ollaya is down the SDK reports `502 status code (no body)` instead of a refused connection. Set `NO_PROXY=localhost,127.0.0.1` next to `TYPESAFE_BASE_URL`.
- **Warm-up and timings.** To load a model before the first request, send `{"model": "laya", "keep_alive": -1}` to [`/api/decide`](/docs/api#decide) (no `state`). `/v1/*` responses carry no timings, as TypeSafe's don't; `/api/decide` reports `total_duration`, `load_duration` and `eval_duration`.

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/systemone` | Decide. Request: `model`, `state` (required) and `questions`. Response: exactly `model`, `answers` and `usage`. |
| `POST /v1/decisions` | Alias of `/v1/systemone` |
| `GET /v1/models` | The models on this machine: `name`, `description`, `release_date` |

## Request and response

```shell
curl http://localhost:11435/v1/systemone \
  -H "Authorization: Bearer local" \
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
  "model": "laya:en",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.9547,
      "probabilities": {"invoice": 0.9698, "refund": 0.0172, "other": 0.013}
    }
  },
  "usage": {"input_tokens": 43, "output_tokens": 0}
}
```

`model` in the response is the checkpoint that answered: `laya` is a router, and this English request went to `laya:en`. TypeSafe's schema allows this ("may differ from the alias supplied in the request"). Values have 4 decimals, and `probabilities` follow the order of `criteria`.

```shell
curl http://localhost:11435/v1/models -H "Authorization: Bearer local"
```

```json
{
  "models": [
    {
      "name": "laya:en",
      "description": "English decision model (ModernBERT-large): guardrails, email and ticket triage.",
      "release_date": "2026-09-23"
    },
    {
      "name": "laya:latest",
      "description": "Routes each request to laya:en or laya:multilingual by the text's script and language.",
      "release_date": "2026-09-23"
    },
    {
      "name": "laya:multilingual",
      "description": "Decision model for 100+ languages (mmBERT-base).",
      "release_date": "2026-09-23"
    }
  ]
}
```

`/v1/models` lists the models pulled to this machine, routers included; it doesn't list the registry.

## Errors

Errors carry TypeSafe's status codes, and a body the SDK reads correctly: a string `error` (which the SDK shows), a machine-readable `code`, and on `422` TypeSafe's `detail` list of validation issues. See [errors](/docs/api#errors) for every code.

```json
{"error": "model \"jev-latest:latest\" not found, try pulling it first", "code": "MODEL_NOT_FOUND"}
```

## What is different

Compatibility covers the API, not the model:

- **Model names** are Ollaya's (`laya`, `laya:en`), so set `TYPESAFE_DEFAULT_MODEL` or pass `model`.
- **Missing `instructions`.** When a question has none, the model reads the question id in its place, so name questions descriptively (`is_urgent`, `tone`).
- **Limits.** At most 256 questions per request, 2–255 choices and 2–10 score levels. Each model also has an option budget: about 125 options for `laya:en`, 250 for `laya:multilingual`.
- **Long states.** TypeSafe reads up to 65,536 tokens; an open model's context is shorter (512 tokens for `laya:en`, 1,024 for `laya:multilingual`, including the questions). When a state doesn't fit, `/v1/*` returns `422 STATE_TRUNCATED` rather than answering from part of it. Use a model with a longer context, shorten the state, or call [`/api/decide`](/docs/api#decide), which truncates and reports `state_truncated`.
- **`/v1/*` stays pure.** Native fields such as `keep_alive` and `extras` are ignored there; routing, timings and truncation are reported on [`/api/decide`](/docs/api#decide).
- **Quality** comes from open models, so it differs from Jev's by task:
  - `laya:typed-decisions` scores 0.766 on typed-decisions, against 0.727 published for Jev 1.13.
  - The base Laya checkpoints are near chance zero-shot on typed-decisions (0.362).
  - Choice questions with many options (more than ~20) are weaker: 0.425 on Banking77, against 0.870 for Jev.

Measure on your own data before switching production traffic. The [Laya page](/library/laya) has the details.

## Not affiliated

Ollaya is an independent open-source project. It is not affiliated with or endorsed by TypeSafe.
