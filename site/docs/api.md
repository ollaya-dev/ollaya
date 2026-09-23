---
title: API reference
nav: API
description: The native /api endpoints, the TypeSafe-compatible /v1 endpoints, streaming pulls and errors.
order: 3
---

# API reference

`ollaya serve` exposes two APIs on `http://localhost:11435`:

- the **native API** under `/api/*`, modelled on Ollama's, for decisions and model management;
- the **TypeSafe-compatible API** under `/v1/*`, so existing TypeSafe SDKs work unchanged. See [TypeSafe compatibility](/docs/typesafe-compatibility).

## Conventions

- Request and response bodies are JSON. Send `Content-Type: application/json`.
- Model names use the `name:tag` format. A missing tag means `latest`.
- Streaming endpoints return [newline-delimited JSON](https://github.com/ndjson/ndjson-spec) (NDJSON): one JSON object per line.
- Errors return a non-2xx status code and a body with a single `error` field:

```json
{ "error": "model 'laya:xl' not found" }
```

## Decide

```
POST /api/decide
```

Answers typed questions about a state in a single forward pass. The body is the same as for `/v1/systemone`, plus an optional `keep_alive`. The response is TypeSafe's response plus `routing` and timings.

### Request

| Field | Type | Description |
|---|---|---|
| `model` | string | Model name, e.g. `laya` or `laya:multilingual` |
| `state` | string or object | The text, email, ticket or JSON to decide about |
| `questions` | object | Question id → [question](#questions) |
| `keep_alive` | string | Optional. How long to keep the model loaded after the request (default `5m`) |

### Questions

Every question has a `type`, `instructions`, and — depending on the type — `criteria`.

| Type | `criteria` | Notes |
|---|---|---|
| `choice` | Object of option → description | Up to 255 options |
| `score` | Array of level descriptions, lowest first | 2–10 levels |
| `noul` | Optional `{"true": "…", "false": "…"}` | Binary; the answer is P(true) |

### Example

```shell
curl http://localhost:11435/api/decide \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "state": "I was charged twice for my subscription this month.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoices and refunds",
          "technical": "Bugs, errors and outages",
          "account": "Login, profile and settings"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Not urgent", "Normal", "Urgent"]
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?",
        "criteria": { "true": "Asks for money back", "false": "Does not ask for money back" }
      }
    }
  }'
```

### Response

```json
{
  "model": "laya",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.78,
      "probabilities": { "billing": 0.852, "account": 0.087, "technical": 0.061 }
    },
    "urgency": {
      "type": "score",
      "score": 1.2,
      "confidence": 0.34,
      "legend": { "0": "Not urgent", "1": "Normal", "2": "Urgent" },
      "probabilities": { "0": 0.12, "1": 0.56, "2": 0.32 }
    },
    "refund": { "type": "noul", "noul": 0.91 }
  },
  "routing": { "model": "laya:en" },
  "usage": { "input_tokens": 84, "output_tokens": 0 },
  "total_duration": 41250000,
  "load_duration": 0
}
```

| Field | Description |
|---|---|
| `model` | The model you asked for |
| `answers` | Question id → answer (see below) |
| `routing` | Which checkpoint served the request when `model` is a router such as `laya` |
| `usage` | `input_tokens` read from the state and questions; `output_tokens` is always `0` |
| `total_duration` | Time spent on the request, in nanoseconds |
| `load_duration` | Time spent loading the model, in nanoseconds (`0` when it was already warm) |

| Answer type | Fields |
|---|---|
| `choice` | `choice` (the most likely option), `confidence`, `probabilities` per option |
| `score` | `score` (expected value over level indexes), `confidence`, `legend` (index → level), `probabilities` per index |
| `noul` | `noul`: probability that the answer is true |

`confidence` is the normalized top probability: (K · p<sub>max</sub> − 1) / (K − 1), where K is the number of options or levels. It is 0 when all options are equally likely and 1 when one option has all the probability.

Decision models never generate text, so `usage.output_tokens` is always `0`.

## List local models

```
GET /api/tags
```

Returns the models on this machine.

```json
{
  "models": [
    { "name": "laya:latest", "size": 1712000000, "modified_at": "2026-10-01T09:30:00Z", "digest": "sha256:…" }
  ]
}
```

## Show model details

```
POST /api/show
```

```json
{ "model": "laya:en" }
```

Returns the model's details: backbone, parameters, context length, precision, baked-in questions, calibration and license.

## Pull a model

```
POST /api/pull
```

```json
{ "model": "laya:multilingual" }
```

Streams progress as NDJSON:

```json
{"status": "pulling manifest"}
{"status": "pulling sha256:…", "digest": "sha256:…", "total": 650000000, "completed": 120000000}
{"status": "pulling sha256:…", "digest": "sha256:…", "total": 650000000, "completed": 650000000}
{"status": "verifying sha256 digest"}
{"status": "writing manifest"}
{"status": "success"}
```

Set `"stream": false` to receive a single response when the pull finishes.

## List running models

```
GET /api/ps
```

Returns the models loaded in memory and when each will be unloaded (`keep_alive`).

## Delete a model

```
DELETE /api/delete
```

```json
{ "model": "my-guardrail" }
```

## Copy a model

```
POST /api/copy
```

```json
{ "source": "laya:en", "destination": "my-guardrail" }
```

## Create a model

```
POST /api/create
```

Creates a model from a [Modelfile](/docs/modelfile).

```json
{ "model": "triage", "modelfile": "FROM laya:en\nQUESTIONS ./questions.json" }
```

Progress is streamed as NDJSON, ending with `{"status": "success"}`.

## Push a model

```
POST /api/push
```

```json
{ "model": "yourname/triage" }
```

Uploads the model to a registry. Progress is streamed as NDJSON.

## Version

```
GET /api/version
```

```json
{ "version": "0.1.0" }
```

## TypeSafe-compatible endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/systemone` | Same request and response as TypeSafe's System One endpoint |
| `POST /v1/decisions` | Alias of `/v1/systemone` |
| `GET /v1/models` | The models available on this machine |

See [TypeSafe compatibility](/docs/typesafe-compatibility) for details.
