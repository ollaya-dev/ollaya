# Ollaya HTTP API: normative contract

Status: **normative** for Ollaya 0.1.0 and later. The daemon (`crates/ollaya-server`), the CLI and
the Rust client (`crates/ollaya-api`) implement this document. Where the drafts in
`site/docs/api.md` and `site/docs/typesafe-compatibility.md` disagree with it, this document wins
(see [Appendix A](#appendix-a-changes-required-in-the-site-drafts)).

The words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

The JSON examples in this file are checked by `cargo test -p ollaya-api`: every block tagged with an
HTML comment (`<!-- json: Type -->`, `<!-- ndjson: Type -->`, `<!-- curl: Type -->`) round-trips
through the matching type in `ollaya-api`, and request examples pass boundary validation.

## Contents

1. [Endpoints](#1-endpoints)
2. [Conventions](#2-conventions)
3. [Model names and resolution](#3-model-names-and-resolution)
4. [Error model](#4-error-model)
5. [Questions and validation](#5-questions-and-validation)
6. [keep_alive](#6-keep_alive)
7. [Native API: `/api/*`](#7-native-api)
8. [TypeSafe-compatible API: `/v1/*`](#8-typesafe-compatible-api)
9. [Router models](#9-router-models)
10. [Concurrency, queueing, timeouts and cancellation](#10-concurrency-queueing-timeouts-and-cancellation)
11. [Idempotency and retries](#11-idempotency-and-retries)
12. [Versioning and evolution](#12-versioning-and-evolution)
13. [Compatibility guarantees](#13-compatibility-guarantees)
14. [Security](#14-security)
15. [Environment variables](#15-environment-variables)
16. [Verification checklist](#16-verification-checklist)
- [Appendix A: changes required in the site drafts](#appendix-a-changes-required-in-the-site-drafts)

---

## 1. Endpoints

| Method | Path | Purpose | Streams |
|---|---|---|---|
| `GET`, `HEAD` | `/` | Liveness: `Ollaya is running` | no |
| `GET` | `/api/version` | Server version | no |
| `POST` | `/api/decide` | Answer typed questions about a state; also load and unload a model | no |
| `GET` | `/api/tags` | Models on this machine | no |
| `POST` | `/api/show` | One model's details | no |
| `GET` | `/api/ps` | Models loaded in memory | no |
| `POST` | `/api/pull` | Download a model from a registry | yes (default) |
| `DELETE` | `/api/delete` | Remove a local model | no |
| `POST` | `/api/copy` | Copy a local model to a new name | no |
| `POST` | `/api/create` | Create a model from another one (Modelfile) | yes (default) |
| `POST` | `/api/push` | Reserved: `501 NOT_IMPLEMENTED` | — |
| `HEAD`, `POST` | `/api/blobs/:digest` | Reserved: `501 NOT_IMPLEMENTED` | — |
| `POST` | `/v1/systemone` | TypeSafe System One, wire-identical | no |
| `POST` | `/v1/decisions` | Alias of `/v1/systemone` | no |
| `GET` | `/v1/models` | TypeSafe model list: the local models | no |

Any other path returns `404 NOT_FOUND`. A known path with the wrong method returns
`405 METHOD_NOT_ALLOWED` with an `Allow` header. Ollama's text endpoints (`/api/generate`,
`/api/chat`, `/api/embed`, `/api/embeddings`) return `404 NOT_FOUND` with a message pointing at
`/api/decide`. The daemon never serves `/v2/`: that prefix belongs to model registries.

## 2. Conventions

- **Base URL.** `http://127.0.0.1:11435` by default; see `OLLAYA_HOST` in
  [§15](#15-environment-variables). All paths are absolute from the base URL.
- **Bodies.** Requests and responses are UTF-8 JSON objects. The server parses a request body as
  JSON **whatever its `Content-Type`** (Ollama behaviour: `curl -d` sends
  `application/x-www-form-urlencoded`). Responses carry `Content-Type: application/json`, except
  streams (`application/x-ndjson`) and `GET /` (`text/plain; charset=utf-8`).
- **Size.** A request body is at most **8 MiB** (8,388,608 bytes); larger bodies get
  `413 REQUEST_TOO_LARGE`.
- **Field names** are `snake_case` on every endpoint, as TypeSafe and Ollama spell them.
- **Unknown request fields are ignored** on every endpoint. This matches TypeSafe (pydantic
  `extra="ignore"`) and Ollama (Go JSON decoding), and lets newer clients talk to older servers.
  Unknown *values* of known enum-like fields are rejected (`422`).
- **`null` means absent** for every optional request field.
- **Response fields are always present** unless a table marks them *omitted when …*. A field with
  no value is `null`, never missing.
- **Key order.** Objects keyed by the caller (`questions`, `answers`, `probabilities`, `legend`,
  `criteria`) keep the caller's order. Top-level field order is not significant.
- **Numbers.** Probabilities, confidences, `score` and `noul` are rounded to **4 decimal places**
  (Python `round(x, 4)`), so probabilities may sum to 1 ± 0.0005. Byte sizes, token counts and
  durations are non-negative integers.
- **Durations** in responses are integers in **nanoseconds** (`*_duration`), as in Ollama.
- **Timestamps** are RFC 3339 in UTC with a `Z` suffix and 0, 3, 6 or 9 fractional digits, e.g.
  `2026-09-24T09:30:12.418Z`.
- **Digests.** Layer digests are `sha256:<64 hex>`. A *model digest* (`digest` in `/api/tags` and
  `/api/ps`) is the sha256 of the manifest bytes as **bare hex**, as Ollama prints it.
- **Streaming.** Streaming endpoints send newline-delimited JSON: one object per line, each line
  terminated by `\n` and flushed immediately. A stream ends after exactly one terminal line: either
  `{"status":"success"}` or an [error line](#43-errors-inside-a-stream). A stream that ends without
  one was cut off and MUST be treated as a failure. Send `"stream": false` to get one JSON response
  when the work is done.
- **Request IDs.** Every response carries `X-Request-Id`. The server echoes a client-sent
  `X-Request-Id` if it matches `[A-Za-z0-9._-]{1,128}`, and generates an opaque ID otherwise.
  `/v1/*` responses also carry the same value as `x-typesafe-request-id`, which the TypeSafe SDK
  exposes as `response.request_id`.
- **Authorization.** Ignored unless `OLLAYA_API_KEY` is set ([§14](#14-security)). Any
  `Authorization` header, such as the TypeSafe SDK's `Bearer <key>`, is accepted.

## 3. Model names and resolution

**Grammar.** `[host/][namespace/]model[:tag]`, as in Ollama.

| Part | Rule | Default |
|---|---|---|
| `host` | Registry host, optionally with `http://` or `https://` (development registries) | `OLLAYA_REGISTRY`, else `ollaya.cobanov.dev` |
| `namespace` | 1–80 chars, `[A-Za-z0-9_][A-Za-z0-9_.-]*` | `library` |
| `model` | 1–80 chars, `[A-Za-z0-9_][A-Za-z0-9_.-]*` | — |
| `tag` | 1–80 chars, `[A-Za-z0-9_][A-Za-z0-9_.-]*` | `latest` |

**Normalization.** Names are trimmed and compared case-insensitively. The **canonical name** is the
shortest form, with the tag always shown: `laya` → `laya:latest`, `Laya:EN` → `laya:en`,
`acme/triage` → `acme/triage:latest`, `localhost:8080/library/laya:en` stays as is. Every response
field that names a model (`model`, `name`, `routing.*`, `parent_model`) uses the canonical name.
A string that does not parse is `422 INVALID_REQUEST` with a `model_name` issue on the field.

**Resolution** (`/api/decide`, `/v1/systemone`, `/api/show`, `/api/delete`, `/api/copy` `source`,
`/api/create` `from`):

1. Parse and normalize the name.
2. Look up the local manifest. If there is none, respond `404 MODEL_NOT_FOUND` with the message
   `model "<canonical>" not found, try pulling it first`. This sentence is Ollama's, verbatim; tools
   match on it, so it is the one error message this contract freezes.
3. If the model is a router ([§9](#9-router-models)) and the endpoint answers questions, pick the
   route target and resolve it the same way. A missing target is `404 MODEL_NOT_FOUND` naming the
   target: `model "laya:en" not found, try pulling it first (routed from "laya:latest")`.

**No implicit pulls.** No endpoint downloads a model as a side effect, `/v1/systemone` included. A
pull moves hundreds of megabytes. The TypeSafe SDK times out after 10 s and retries, so an
auto-pull would start a download that no caller is waiting for. Ollama makes the same split:
`ollama run` pulls, the API does not. `ollaya run` does the same: when `/api/show` returns
`MODEL_NOT_FOUND`, it calls `/api/pull` first.

**Precision.** A model such as `laya:en` carries an fp16 and an fp32 graph that share one weights
file. The precision is chosen **when the model loads**: fp16 on a CUDA GPU, fp32 on CPU. In
`/api/tags` and `/api/show`, `details.quantization_level` lists the graphs the model carries
(`F16/F32`); `/api/ps` reports the one actually loaded (`F16` or `F32`). A precision-pinned tag
(`laya:en-fp32`, or a model created with `parameters.precision`) keeps a single graph.

## 4. Error model

### 4.1 Error body

Every error response, on every endpoint including `/v1/*`, has this body:

| Field | Type | Presence | Meaning |
|---|---|---|---|
| `error` | string | always | Human-readable message. Not stable: do not parse it (exception: [§3](#3-model-names-and-resolution)). |
| `code` | string | always | Machine-readable code from the table below, `UPPER_SNAKE_CASE`. Branch on this. |
| `detail` | array of [issues](#44-validation-issues) | omitted unless `code` is `INVALID_REQUEST`, `TOO_MANY_OPTIONS` or `INPUT_TOO_LONG` | Every validation problem found, in TypeSafe's `ValidationError` shape. |

<!-- json: ErrorBody -->
```json
{
  "error": "model \"laya:xl\" not found, try pulling it first",
  "code": "MODEL_NOT_FOUND"
}
```

**Why this shape.** Three kinds of client read these errors, and this one body works for all of
them:

- **Ollama clients** (Go `api.StatusError`, `ollama-python`, `ollama-js`) read `error` as a
  string. `{"error": {"code", "message"}}` would break them: Go fails to decode the body, and JS
  prints `[object Object]`.
- **The TypeSafe SDK** (`typesafe_sdk._core.errors.extract_message`) checks a string `error`
  first, so it shows `error` verbatim. Its `422` schema is FastAPI's `HTTPValidationError`
  (`{"detail": [ValidationError]}`). `detail` carries exactly that list, so code that inspects
  `err.body["detail"]` keeps working.
- **Programs** need machine-readable codes, which the skill's error model asks for. `code` provides
  them, as a sibling field that the first two kinds of client ignore.

### 4.2 Error codes

The `code` set is **open**: clients MUST handle an unknown code by falling back to the HTTP status.

| Code | HTTP | When | Retry? |
|---|---|---|---|
| `INVALID_JSON` | 400 | Body missing, not valid JSON, or not a JSON object | no |
| `INVALID_REQUEST` | 422 | Body fails validation (schema, limits, model name, `keep_alive`, `extras`, …); `detail` lists every issue | no |
| `TOO_MANY_OPTIONS` | 422 | A question's options do not fit the answering model's option budget ([§5.3](#53-model-specific-limits)) | no |
| `INPUT_TOO_LONG` | 422 | `state` is longer than 65,536 tokens | no |
| `UNAUTHORIZED` | 401 | `OLLAYA_API_KEY` is set and the request has no matching `Authorization: Bearer` header | no |
| `FORBIDDEN` | 403 | `Origin` or `Host` header not allowed ([§14](#14-security)) | no |
| `MODEL_NOT_FOUND` | 404 | Model (or a router's target) not on this machine; for `/api/pull`, not in the registry | no |
| `NOT_FOUND` | 404 | No such endpoint | no |
| `METHOD_NOT_ALLOWED` | 405 | Endpoint exists, method does not | no |
| `OPERATION_IN_PROGRESS` | 409 | A pull or create is writing the same model name ([§11](#11-idempotency-and-retries)) | yes, after it finishes |
| `REQUEST_TOO_LARGE` | 413 | Body over 8 MiB | no |
| `QUEUE_FULL` | 503 | `OLLAYA_MAX_QUEUE` requests are already waiting; sent with `Retry-After: 1` | yes |
| `MODEL_LOAD_FAILED` | 500 | The runner could not load the model: corrupt files, not enough memory, or no load within `OLLAYA_LOAD_TIMEOUT` | rarely |
| `INFERENCE_FAILED` | 500 | The runner crashed or failed during a decision | yes |
| `STORAGE_ERROR` | 500 | Local disk failure: disk full, permissions, I/O | no |
| `INTERNAL` | 500 | A bug. The message does not expose internals; the server log has the details under the request ID. | yes |
| `UNSUPPORTED_MODEL` | 501 | This build cannot run the model's format or engine (e.g. a `gguf` model before the llama.cpp runner ships) | no |
| `NOT_IMPLEMENTED` | 501 | Reserved endpoint (`/api/push`, `/api/blobs/:digest`) | no |
| `REGISTRY_ERROR` | 502 | The registry or blob host is unreachable, answers with an unexpected status, or serves an invalid manifest | yes |
| `DIGEST_MISMATCH` | 502 | A downloaded blob does not match its sha256; the blob is discarded | yes |

The TypeSafe SDK maps `400`, `401`, `403`, `404`, `422` and `429` to specific exception classes
and `5xx` to `TypeSafeInternalServerError`. By default it retries `408`, `429` and `5xx`, and
honours `Retry-After`. The table is chosen so that SDK behaviour is right: requests that cannot
succeed are `4xx` and are not retried; overload is `503` with `Retry-After`.

**Status choice for validation.** `400` is reserved for bodies that cannot be parsed at all.
Everything that parses but is semantically invalid is `422`. That covers schema violations and
limits that depend on the model: the request is well-formed, but this model cannot process it.
FastAPI, and therefore TypeSafe, uses `422` for the same class of error.

### 4.3 Errors inside a stream

Once a stream has started, the status is already `200`. A failure is then sent as one final line
in the error-body shape, and the stream ends:

<!-- ndjson: ProgressResponse -->
```json
{"status":"pulling manifest"}
{"status":"pulling 8d32a80bb199","digest":"sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2","total":841114235,"completed":120000000}
{"error":"blob sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2 does not match its digest; the download was discarded","code":"DIGEST_MISMATCH"}
```

A client MUST check each line for `error` before reading it as progress. Errors detected **before**
the first line is written are ordinary HTTP errors, not stream lines. For `/api/pull`, that covers
an invalid name, a manifest that does not exist and an unreachable registry
([§7.6](#76-post-apipull)).

### 4.4 Validation issues

`detail` entries use TypeSafe's (FastAPI's) `ValidationError` shape:

| Field | Type | Presence | Meaning |
|---|---|---|---|
| `loc` | array of string \| integer | always | Path to the bad value: `"body"`, then keys and array indexes. Inside a question, the question's `type` follows its id, as FastAPI does for discriminated unions: `["body","questions","urgency","score","criteria"]`. |
| `msg` | string | always | Human-readable; pydantic's wording where pydantic has one. Not stable. |
| `type` | string | always | Machine-readable issue type (open set, see below) |
| `ctx` | object | omitted when there is no limit to report | The limit and the actual value, e.g. `{"min_length": 2, "actual_length": 1}` |

TypeSafe's schema also allows an `input` field. Ollaya never sends it, because it would echo user
data back.

`error` is built from `detail` exactly as the TypeSafe SDK's `extract_message` would build it. Each
issue becomes `<loc without "body", joined with ".">: <msg>`, and issues are joined with `"; "`.
TypeSafe SDK users therefore see the same text from Ollaya as from TypeSafe.

| `type` | Meaning | `ctx` |
|---|---|---|
| `missing` | Required field absent (`msg`: `Field required`) | — |
| `string_type`, `bool_type`, `dict_type`, `list_type`, `float_type` | Wrong JSON type | — |
| `state_type` | `state` is not a string, object or array | — |
| `json_type` | `instructions` or a criterion is not a string, object, array (or `null` where allowed) | — |
| `union_tag_not_found` | Question without a string `type` | `{"discriminator": "'type'"}` |
| `union_tag_invalid` | Unknown question `type` | `{"discriminator": "'type'", "tag": "<given>", "expected_tags": "'choice', 'score', 'noul'"}` |
| `too_short`, `too_long` | Count outside a limit | `{"field_type": "List" \| "Dictionary", "min_length" \| "max_length": n, "actual_length": n}` |
| `string_too_short` | Empty string where a name is required | `{"min_length": 1}` |
| `model_name` | `model` / `source` / `destination` / `from` does not parse ([§3](#3-model-names-and-resolution)) | — |
| `keep_alive` | `keep_alive` does not parse ([§6](#6-keep_alive)) | — |
| `enum` | Unknown value in a closed set, e.g. `extras` | `{"expected": "'laya'"}` |
| `stream_unsupported` | `"stream": true` on `/api/decide` | — |
| `parameter` | Unknown or invalid `/api/create` parameter | — |
| `calibration` | `/api/create` calibration key that is not `<type>:<bucket>` | — |
| `too_many_options` | Code `TOO_MANY_OPTIONS` | `{"options": n, "model": "<canonical>"}` |
| `input_too_long` | Code `INPUT_TOO_LONG` | `{"max_tokens": 65536, "tokens": n}` |

Example: a `/v1/systemone` body without `state` and with a one-level score. Both issues are
reported:

<!-- curl: SystemOneRequest invalid -->
```shell
curl http://localhost:11435/v1/systemone -d '{
  "model": "laya",
  "questions": {
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["Can wait"]}
  }
}'
```

<!-- json: ErrorBody -->
```json
{
  "error": "state: Field required; questions.urgency.score.criteria: List should have at least 2 items after validation, not 1",
  "code": "INVALID_REQUEST",
  "detail": [
    {"loc": ["body", "state"], "msg": "Field required", "type": "missing"},
    {
      "loc": ["body", "questions", "urgency", "score", "criteria"],
      "msg": "List should have at least 2 items after validation, not 1",
      "type": "too_short",
      "ctx": {"field_type": "List", "min_length": 2, "actual_length": 1}
    }
  ]
}
```

## 5. Questions and validation

`/api/decide`, `/v1/systemone`, `/v1/decisions` and `/api/create` share **one** question schema
and **one** validator (`ollaya_api::validate`). It is TypeSafe's wire schema
(`typesafe_sdk/_schemas/models.py`), with the limits below. All issues are collected and reported
together.

### 5.1 Request body shared by the decision endpoints

| Field | Type | Required | Rules |
|---|---|---|---|
| `model` | string | yes | Non-empty; a valid name ([§3](#3-model-names-and-resolution)) |
| `state` | string \| object \| array | yes on `/v1/*` (see [§7.3](#73-post-apidecide) for `/api/decide`) | Any JSON string, object or array; `""` is allowed. Numbers, booleans and `null` are `state_type` issues. At most **65,536 tokens** by the answering model's tokenizer (`INPUT_TOO_LONG`). |
| `questions` | object: question id → [question](#52-question-schema) | yes, unless the model has embedded questions | **1–256** questions. Ids are any strings, in the caller's order. If present, they **replace** the model's embedded questions entirely. |

### 5.2 Question schema

Every question is an object with a `type` and, depending on it, `instructions` and `criteria`.
Unknown fields in a question are ignored.

| `type` | `instructions` | `criteria` | Options | Answer |
|---|---|---|---|---|
| `choice` | optional: string \| object \| array \| `null` | **required**: object label → description (string \| object \| array \| `null`); or, as an Ollaya extension inherited from laya, an array of label strings (duplicates collapse onto the first) | **2–255** labels | `choice`, `confidence`, `probabilities` |
| `score` | optional, as above | **required**: array of level descriptions (string \| object \| array), level 0 first | **2–10** levels | `score`, `confidence`, `legend`, `probabilities` |
| `noul` | optional, as above | optional: `null` or object with optional `true` and `false` descriptions (string \| object \| array \| `null`). Keys are exact; other keys are ignored, as in TypeSafe. | always 2 | `noul` |

**Missing instructions.** TypeSafe makes `instructions` optional; laya was trained with them always
present. When `instructions` is absent or `null`, the model reads the **question id** in its place.
For example, `{"tone": {"type": "choice", "criteria": {...}}}` is read as if its instructions were
`"tone"`. Descriptive ids (`tone`, `is_spam`, `urgency`) are what a TypeSafe user naturally writes,
and they carry the question's meaning, which an empty string would not. An explicit string,
including `""`, is used verbatim. `ollaya_api::decide::engine_questions` applies this rule, and
`ollaya_decision::parse_questions` receives its output.

**Why 2–10 levels when TypeSafe's schema says `min_length=1`, and why at least 2 choices.** Jev's
documented rules are at most 255 choices and 2–10 score levels. The open `decider` replica of Jev
enforces 2–255 choices and 2–10 levels. The OpenAPI `min_length=1` is only the schema-level floor.
A one-option question carries no decision: its confidence formula divides by K − 1 = 0. There is
also an asymmetry between the two directions of change. Relaxing a limit later is additive;
tightening one would break clients. Starting strict keeps both directions open.

**Types follow TypeSafe, not laya.** laya also accepts numbers and booleans as criteria and
instructions, rendering them with `json.dumps`. TypeSafe rejects them, and so does this API
(`json_type`). The engine keeps laya's rendering for parity; the boundary is the stricter of the
two.

### 5.3 Model-specific limits

Some limits depend on the answering model and are checked where the model tokenizes the request.
For a router, the target's limits apply, so the same request can pass for one route and fail for
another.

| Limit | Code | Status | `loc` |
|---|---|---|---|
| Options that do not fit the model's option budget. laya-markers-v1 places every option's `[MASK]` inside `max_len`: about 125 options for `laya:en` (512 tokens) and 250 for `laya:multilingual` (1,024 tokens). | `TOO_MANY_OPTIONS` | 422 | `["body","questions",<id>,<type>,"criteria"]` |
| `state` over 65,536 tokens (TypeSafe's limit) | `INPUT_TOO_LONG` | 422 | `["body","state"]` |

A state that is shorter than 65,536 tokens but longer than the model's context is **truncated** to
fit, as laya does. `/api/decide` reports this in `state_truncated`; `/v1/*` cannot, because its
shape is TypeSafe's.

<!-- json: ErrorBody -->
```json
{
  "error": "questions.intent.choice.criteria: 140 options do not fit the option budget of laya:en",
  "code": "TOO_MANY_OPTIONS",
  "detail": [
    {
      "loc": ["body", "questions", "intent", "choice", "criteria"],
      "msg": "140 options do not fit the option budget of laya:en",
      "type": "too_many_options",
      "ctx": {"options": 140, "model": "laya:en"}
    }
  ]
}
```

### 5.4 Answer shapes

These are TypeSafe's shapes, field for field and in this order. `/v1/*` returns exactly these;
`/api/decide` MAY add the per-answer `laya` object ([§7.3](#73-post-apidecide)).

| `type` | Fields |
|---|---|
| `choice` | `choice` (string): the label with the highest probability. `confidence` (number). `probabilities` (object label → number, in criteria order). |
| `score` | `score` (number): expected level Σ i·pᵢ, which can fall between levels. `confidence` (number). `legend` (object `"0"`… → the level's description, verbatim). `probabilities` (object `"0"`… → number). |
| `noul` | `noul` (number): the probability that the statement holds. No `confidence`, as in TypeSafe. |

`confidence` is TypeSafe's normalized top probability over K options:
**(K·p<sub>max</sub> − 1) / (K − 1)**, clamped to [0, 1]. It is 0 when every option is equally
likely and 1 when one option has all the probability. It is the same formula for every model, so a
threshold transfers across models.

Probabilities are calibrated with the model's temperatures. On CUDA hosts the default variant is
fp16, whose answers can differ from the fp32 reference on near-ties (top-2 gap below 0.01). Measured
agreement is 99.1–99.6%.

## 6. keep_alive

`keep_alive` says how long a model stays loaded after a request finishes. It uses Ollama's
semantics:

| Value | Meaning |
|---|---|
| duration string: `"5m"`, `"1h30m"`, `"300ms"`, `"1.5h"` | Stay loaded this long after the request |
| number of seconds: `300`, `1.5` | Same, in seconds (fractions allowed) |
| numeric string: `"300"` | Same as the number. An Ollaya extension: Ollama accepts this form only in `OLLAMA_KEEP_ALIVE`. |
| `0`, `"0"`, `"0s"` | Unload as soon as the request finishes |
| any negative value: `-1`, `"-1"`, `"-5m"` | Keep loaded until the server stops or an explicit unload |
| absent or `null` | `OLLAYA_KEEP_ALIVE`, default `5m` |

**Grammar.** Durations use Go's `time.ParseDuration` syntax: an optional sign, then one or more
`<decimal><unit>` groups. Units are `ns`, `us` (`µs`, `μs`), `ms`, `s`, `m` and `h`. `"0"` is
valid alone. A value that does not parse, a non-finite number, or one above 2⁶⁴ − 1 ns (about 584
years) is a `keep_alive` issue (`422`). `ollaya_api::KeepAlive` implements this grammar.

**Rules.**
- The timer starts when a request finishes. While requests are running, a model is never unloaded.
- The most recent request's `keep_alive` replaces the timer, as in Ollama.
- `keep_alive` applies to the model that answered: the route target for a router. A load or unload
  request for a router ([§7.3](#73-post-apidecide)) applies to every target.
- `/v1/*` does not read `keep_alive` (TypeSafe has no such field); those requests use the default.
- When `OLLAYA_MAX_LOADED_MODELS` is reached, an idle model can be unloaded before its timer runs
  out to make room ([§10](#10-concurrency-queueing-timeouts-and-cancellation)).

---

## 7. Native API

### 7.1 `GET /`, `HEAD /`

Liveness. Always `200`, even when `OLLAYA_API_KEY` is set. `GET` returns the text below; `HEAD`
returns no body. The client's `heartbeat()` sends `HEAD /`.

```shell
curl http://localhost:11435/
```

```text
Ollaya is running
```

### 7.2 `GET /api/version`

The server's version, a SemVer string. No request body.

| Response field | Type | Meaning |
|---|---|---|
| `version` | string | Server version, e.g. `0.1.0` |

```shell
curl http://localhost:11435/api/version
```

<!-- json: VersionResponse -->
```json
{"version": "0.1.0"}
```

Errors: none beyond the global ones (`401`, `403`). Safe to retry.

### 7.3 `POST /api/decide`

Answers typed questions about a state in one forward pass. This is the native inference endpoint.
Its body is the [`/v1/systemone` body](#51-request-body-shared-by-the-decision-endpoints) plus native
options, and its response is TypeSafe's response plus native fields. **Every native addition is
additive**, so a TypeSafe client can parse an `/api/decide` response as a `SystemOneResponse`.

The same endpoint loads and unloads models, as Ollama's `/api/generate` does: a request **without
`state`** does not decide anything ([below](#load-and-unload)).

#### Request

| Field | Type | Required | Default | Rules |
|---|---|---|---|---|
| `model` | string | yes | — | [§3](#3-model-names-and-resolution) |
| `state` | string \| object \| array | no | — | As in [§5.1](#51-request-body-shared-by-the-decision-endpoints). Absent or `null`: a load or unload request. |
| `questions` | object | with `state`, unless the model has embedded questions | the model's embedded questions | [§5.2](#52-question-schema). Not allowed without `state` (`missing` issue on `state`). |
| `keep_alive` | string \| number | no | `OLLAYA_KEEP_ALIVE` (`5m`) | [§6](#6-keep_alive) |
| `extras` | array of string | no | `[]` | Closed set: `"laya"`. Each value adds a same-named object to every answer. Unknown values are `enum` issues. |
| `stream` | boolean | no | `false` | Reserved. `/api/decide` does not stream, and `true` is a `stream_unsupported` issue. It is rejected rather than ignored so that a streaming mode can be added later without changing what existing `stream: true` callers receive. |

Native options are deliberately few. `keep_alive` is Ollama's lifecycle control. `extras` is how
model-family outputs are exposed without redefining a TypeSafe field. There is no `options` object
(Ollama's sampling parameters do not apply to decision models); one can be added later as an
optional field.

#### Response

| Field | Type | Meaning |
|---|---|---|
| `model` | string | Canonical name of the model **that answered**: the route target for a router (`laya:en` for a `laya` request). TypeSafe documents that this "may differ from the alias supplied in the request". |
| `answers` | object id → answer | [§5.4](#54-answer-shapes), in question order. `{}` for load and unload. |
| `usage.input_tokens` | integer | Encoder tokens read, summed over questions, special tokens included (laya's count). `0` for load and unload. |
| `usage.output_tokens` | integer | Tokens generated. Always `0`: decision models do not generate. |
| `routing` | object \| `null` | The routing decision for a router ([§9](#9-router-models)); `null` otherwise, and for load and unload |
| `state_truncated` | boolean | `true` if, for at least one question, part of `state` was dropped to fit the model's context |
| `done_reason` | string | `"decide"`, `"load"` or `"unload"` |
| `created_at` | string | When the response was produced ([timestamps](#2-conventions)) |
| `total_duration` | integer, ns | From receiving the request to producing the response, queueing included |
| `load_duration` | integer, ns | Time this request waited for the model to load; `0` when it was warm |
| `eval_duration` | integer, ns | Time in the runner: tokenization, forward pass, calibration. `0` for load and unload. |

With `"extras": ["laya"]`, every answer also carries:

| Field | Type | Meaning |
|---|---|---|
| `laya.confidence` | number | laya's confidence: 1 − H(p)/ln K for `choice` and `score`, max(p, 1 − p) for `noul`. Namespaced so that it never redefines TypeSafe's `confidence`. |
| `laya.act_probability` | number \| `null` | Probability that acting on this answer is appropriate, from the act head; `null` for models without one (`capabilities` lacks `act`) |

#### Example

<!-- curl: DecideRequest -->
```shell
curl http://localhost:11435/api/decide -d '{
  "model": "laya",
  "state": "I was charged twice for my subscription this month. Please refund the second charge.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "technical": "Bugs, errors and outages",
        "account": "Login, profile and settings"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
    },
    "refund": {
      "type": "noul",
      "instructions": "The customer asks for money back.",
      "criteria": {"true": "Asks for a refund", "false": "Does not ask for a refund"}
    }
  },
  "keep_alive": "10m"
}'
```

<!-- json: DecideResponse -->
```json
{
  "model": "laya:en",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.7781,
      "probabilities": {"billing": 0.8521, "technical": 0.0611, "account": 0.0868}
    },
    "urgency": {
      "type": "score",
      "score": 1.1982,
      "confidence": 0.3418,
      "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
      "probabilities": {"0": 0.1203, "1": 0.5612, "2": 0.3185}
    },
    "refund": {"type": "noul", "noul": 0.9127}
  },
  "usage": {"input_tokens": 118, "output_tokens": 0},
  "routing": {
    "router": "laya:latest",
    "model": "laya:en",
    "route": "english",
    "reason": "English Latin text"
  },
  "state_truncated": false,
  "done_reason": "decide",
  "created_at": "2026-09-24T09:30:12.418Z",
  "total_duration": 18734512,
  "load_duration": 0,
  "eval_duration": 16302117
}
```

The same request with `"extras": ["laya"]` against `laya:en` directly (no router, so `routing` is
`null`):

<!-- json: DecideResponse -->
```json
{
  "model": "laya:en",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.7781,
      "probabilities": {"billing": 0.8521, "technical": 0.0611, "account": 0.0868},
      "laya": {"confidence": 0.5273, "act_probability": 0.8841}
    },
    "urgency": {
      "type": "score",
      "score": 1.1982,
      "confidence": 0.3418,
      "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
      "probabilities": {"0": 0.1203, "1": 0.5612, "2": 0.3185},
      "laya": {"confidence": 0.1413, "act_probability": 0.6120}
    },
    "refund": {
      "type": "noul",
      "noul": 0.9127,
      "laya": {"confidence": 0.9127, "act_probability": 0.9310}
    }
  },
  "usage": {"input_tokens": 118, "output_tokens": 0},
  "routing": null,
  "state_truncated": false,
  "done_reason": "decide",
  "created_at": "2026-09-24T09:30:13.002Z",
  "total_duration": 17210044,
  "load_duration": 0,
  "eval_duration": 16011382
}
```

#### Load and unload

A request without `state` (absent or `null`) and without `questions` loads or unloads the model,
and never decides:

| `keep_alive` | Effect | `done_reason` |
|---|---|---|
| absent, positive or negative | Load the model now (for a router, every target), then apply `keep_alive` | `"load"` |
| `0` | Unload the model (for a router, every target) once its in-flight requests finish. Unloading a model that is not loaded succeeds. | `"unload"` |

The CLI uses these requests: `ollaya run` preloads, and `ollaya stop` sends `keep_alive: 0`. The
model must exist locally (`404 MODEL_NOT_FOUND` otherwise). The response has the
[decide shape](#response) with `answers: {}`, zero `usage`, `routing: null`,
`state_truncated: false` and `eval_duration: 0`.

<!-- curl: DecideRequest -->
```shell
curl http://localhost:11435/api/decide -d '{"model": "laya:en", "keep_alive": -1}'
```

<!-- json: DecideResponse -->
```json
{
  "model": "laya:en",
  "answers": {},
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "routing": null,
  "state_truncated": false,
  "done_reason": "load",
  "created_at": "2026-09-24T09:29:58.120Z",
  "total_duration": 2204518310,
  "load_duration": 2204112034,
  "eval_duration": 0
}
```

<!-- curl: DecideRequest -->
```shell
curl http://localhost:11435/api/decide -d '{"model": "laya:en", "keep_alive": 0}'
```

<!-- json: DecideResponse -->
```json
{
  "model": "laya:en",
  "answers": {},
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "routing": null,
  "state_truncated": false,
  "done_reason": "unload",
  "created_at": "2026-09-24T09:45:01.531Z",
  "total_duration": 48210,
  "load_duration": 0,
  "eval_duration": 0
}
```

#### Errors

`400 INVALID_JSON`, `422 INVALID_REQUEST`, `422 TOO_MANY_OPTIONS`, `422 INPUT_TOO_LONG`,
`404 MODEL_NOT_FOUND`, `413 REQUEST_TOO_LARGE`, `503 QUEUE_FULL`, `500 MODEL_LOAD_FAILED`,
`500 INFERENCE_FAILED`, `501 UNSUPPORTED_MODEL`, plus the global `401`, `403` and `500 INTERNAL`.

<!-- curl: DecideRequest invalid -->
```shell
curl http://localhost:11435/api/decide -d '{"model": "laya", "questions": {"q": {"type": "noul"}}, "stream": true}'
```

<!-- json: ErrorBody -->
```json
{
  "error": "state: Field required; stream: /api/decide does not stream; omit \"stream\" or set it to false",
  "code": "INVALID_REQUEST",
  "detail": [
    {"loc": ["body", "state"], "msg": "Field required", "type": "missing"},
    {
      "loc": ["body", "stream"],
      "msg": "/api/decide does not stream; omit \"stream\" or set it to false",
      "type": "stream_unsupported"
    }
  ]
}
```

#### Idempotency

A decision has no side effect on stored data. The only state it touches is the model's residency
(load and the `keep_alive` timer), which converges no matter how often the request is repeated.
**Safe to retry.**

### 7.4 `GET /api/tags`

Lists the models on this machine, newest `modified_at` first (ties by name). No request body. The
list is **not paginated** (see [§16](#16-verification-checklist), "List endpoints support
pagination").

| Response field | Type | Meaning |
|---|---|---|
| `models[].name` | string | Canonical name |
| `models[].model` | string | Same as `name` (Ollama sends both) |
| `models[].modified_at` | string | When this name was last pulled, created or copied |
| `models[].size` | integer | Bytes of the model's own blobs. A router's size excludes its targets, which are listed separately. |
| `models[].digest` | string | Model digest, bare hex ([§2](#2-conventions)) |
| `models[].details` | [details](#model-details) | |

#### Model details

| Field | Type | Meaning |
|---|---|---|
| `parent_model` | string | `from` of a model made with `/api/create`; `""` otherwise |
| `format` | string | Open set: `onnx`, `gguf`, `router` |
| `family` | string | e.g. `laya`, `von` |
| `families` | array of string | `[family]` today; always an array, never `null` |
| `parameter_size` | string | e.g. `421M`; `""` for routers |
| `quantization_level` | string | Precisions the model carries, GPU preference first: `F16/F32`, or one of `F32`, `F16`, `INT8` for a single graph (GGUF names for GGUF models); `""` for routers. In `/api/ps`: the precision loaded. |

```shell
curl http://localhost:11435/api/tags
```

<!-- json: TagsResponse -->
```json
{
  "models": [
    {
      "name": "laya:latest",
      "model": "laya:latest",
      "modified_at": "2026-09-24T08:12:40.551Z",
      "size": 11862,
      "digest": "5060b6e565647fc9718ef0b190cce61a6c0a90294f5bcfcc37d718a3657b85c8",
      "details": {
        "parent_model": "",
        "format": "router",
        "family": "laya",
        "families": ["laya"],
        "parameter_size": "",
        "quantization_level": ""
      }
    },
    {
      "name": "laya:multilingual",
      "model": "laya:multilingual",
      "modified_at": "2026-09-24T08:12:39.902Z",
      "size": 687214992,
      "digest": "d26c30144bb3c5decfa624594549ba8479fa0874bb94cf68416ad1d0d7e850fa",
      "details": {
        "parent_model": "",
        "format": "onnx",
        "family": "laya",
        "families": ["laya"],
        "parameter_size": "322M",
        "quantization_level": "F16"
      }
    },
    {
      "name": "laya:en",
      "model": "laya:en",
      "modified_at": "2026-09-24T08:11:02.117Z",
      "size": 845897529,
      "digest": "e1b74e2bbeed9fcbbfc7e6ad44a5cbb38d66f7f24f321f699119b56cc0f7158a",
      "details": {
        "parent_model": "",
        "format": "onnx",
        "family": "laya",
        "families": ["laya"],
        "parameter_size": "421M",
        "quantization_level": "F16"
      }
    }
  ]
}
```

Errors: `500 STORAGE_ERROR`. Safe to retry.

### 7.5 `POST /api/show`

Details of one local model.

| Request field | Type | Required | Rules |
|---|---|---|---|
| `model` | string | yes | [§3](#3-model-names-and-resolution); not resolved through a router (shows the router itself) |

| Response field | Type | Meaning |
|---|---|---|
| `license` | string | License text; `""` if the model has no license layer |
| `modelfile` | string | A Modelfile that recreates this model. The text is informative, not for parsing. |
| `parameters` | string | Parameters set on the model, one `name value` per line (`precision fp32`); `""` if none |
| `questions` | object \| `null` | Embedded question schema ([§5.2](#52-question-schema)), or `null` |
| `router` | object \| `null` | For a router: `strategy` (string, e.g. `script`), `default` (route name) and `routes` (route name → canonical model name). `null` otherwise. |
| `details` | [details](#model-details) | |
| `model_info` | object | Namespaced facts. `general.*` keys are stable: `general.architecture` (string), `general.languages` (array of string), `general.source` (string). `<family>.*` keys are family-specific (e.g. `laya.context_length`, `laya.head_max_len`, `laya.encoder`, `laya.layout`). A router has `general.*` keys only. |
| `capabilities` | array of string | Open set: `choice`, `score`, `noul` (question types), `act` (has an act head: `laya.act_probability` is non-null). For a router, the intersection of its targets' capabilities. |
| `modified_at` | string | As in `/api/tags` |

```shell
curl http://localhost:11435/api/show -d '{"model": "laya:en"}'
```

<!-- json: ShowResponse -->
```json
{
  "license": "Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/licenses/\n",
  "modelfile": "# Modelfile generated by \"ollaya show\"\n# To build a new Modelfile based on this, replace FROM with:\n# FROM laya:en\n\nFROM laya:en\n",
  "parameters": "",
  "questions": null,
  "router": null,
  "details": {
    "parent_model": "",
    "format": "onnx",
    "family": "laya",
    "families": ["laya"],
    "parameter_size": "421M",
    "quantization_level": "F16"
  },
  "model_info": {
    "general.architecture": "laya",
    "general.languages": ["en"],
    "general.source": "huggingface.co/convaiinnovations/laya@6f0c2a1",
    "laya.encoder": "answerdotai/ModernBERT-large",
    "laya.layout": "laya-markers-v1",
    "laya.context_length": 512,
    "laya.head_max_len": 192
  },
  "capabilities": ["choice", "score", "noul", "act"],
  "modified_at": "2026-09-24T08:11:02.117Z"
}
```

For the router `laya:latest`, `router` is:

<!-- json: RouterInfo -->
```json
{
  "strategy": "script",
  "default": "english",
  "routes": {"english": "laya:en", "multilingual": "laya:multilingual"}
}
```

Errors: `404 MODEL_NOT_FOUND`, `422 INVALID_REQUEST`, `500 STORAGE_ERROR`. Safe to retry.

### 7.6 `POST /api/pull`

Downloads a model from its registry into the local store. Pulling a **router also pulls every model
it routes to**, because a router cannot answer without them.

| Request field | Type | Required | Default | Rules |
|---|---|---|---|---|
| `model` | string | yes | — | [§3](#3-model-names-and-resolution) |
| `insecure` | boolean | no | `false` | Accepted for Ollama compatibility; has no effect. A development registry is addressed with an explicit `http://` host in the name (`http://127.0.0.1:8123/library/laya:en`). |
| `stream` | boolean | no | `true` | `false`: one response when done |

**Streaming response** (`application/x-ndjson`), one `ProgressResponse` per line:

| Field | Type | Presence | Meaning |
|---|---|---|---|
| `status` | string | always | Open set; the values below |
| `digest` | string | layer lines only | `sha256:<hex>` of the layer |
| `total` | integer | layer lines only | Layer size in bytes |
| `completed` | integer | layer lines only | Bytes present so far |

Status lines, in order. The strings are Ollama's:

1. `pulling manifest`
2. For each blob of the model (config included), in manifest order: `pulling <first 12 hex of the
   digest>` with `digest`, `total` and `completed`. It repeats as bytes arrive; a blob already in
   the store is reported once with `completed == total`. Blobs shared by several models download
   once.
3. `verifying sha256 digest`
4. For a router only: each route target's own steps 1–3 and 5, in route order. There is no
   `success` line in between.
5. `writing manifest`: the manifest is written atomically, and only now does the model appear in
   `/api/tags`.
6. `success`: exactly once, as the last line.

Only the layers this host needs are downloaded (the precision variant chosen at pull time,
[§3](#3-model-names-and-resolution)). Interrupted downloads resume from where they stopped.

**Errors before the stream starts are ordinary HTTP errors.** The server fetches the manifest
before sending the response head. So a name that does not parse (`422`), a model that is not in the
registry (`404 MODEL_NOT_FOUND`: `model "laya:xl" not found in registry ollaya.cobanov.dev`) and an
unreachable registry (`502 REGISTRY_ERROR`) have real status codes, and `curl --fail` works. Ollama
instead streams these as error lines after `200`. Status codes matter to scripts, and Ollama
clients handle both forms.

**Errors after the stream starts** are an error line ([§4.3](#43-errors-inside-a-stream)):
`DIGEST_MISMATCH`, `REGISTRY_ERROR`, `STORAGE_ERROR`.

**Non-streaming** (`"stream": false`): `200 {"status":"success"}` when done, or an ordinary HTTP
error with the codes above.

```shell
curl http://localhost:11435/api/pull -d '{"model": "laya:en"}'
```

<!-- ndjson: ProgressResponse -->
```json
{"status":"pulling manifest"}
{"status":"pulling 2409643934fa","digest":"sha256:2409643934fa5fa03f823921d7cb76a413143606f826946316d5ce4f5e2d15d5","total":318,"completed":318}
{"status":"pulling 59d09cddf064","digest":"sha256:59d09cddf0644c2ceff24ee5afc82839752fd0deb5bf102899f6ecea83f4b616","total":1187205,"completed":1187205}
{"status":"pulling 8d32a80bb199","digest":"sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2","total":841114235,"completed":0}
{"status":"pulling 8d32a80bb199","digest":"sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2","total":841114235,"completed":420557117}
{"status":"pulling 8d32a80bb199","digest":"sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2","total":841114235,"completed":841114235}
{"status":"pulling 3da1a3bd1aa2","digest":"sha256:3da1a3bd1aa2dd93a4d5a50b8f6b00f1a8453cc17cc16df75e1159476c71eed3","total":3583371,"completed":3583371}
{"status":"pulling 39a9e0ae1223","digest":"sha256:39a9e0ae1223400122965307ee5a126b8bd925c5023a2cb3654066f5fd998bdb","total":612,"completed":612}
{"status":"pulling 3a1f403956d8","digest":"sha256:3a1f403956d84101d442241a098093f9e93d6aa827ca1301b8847d926bf9da83","total":431,"completed":431}
{"status":"pulling 9cdd00311420","digest":"sha256:9cdd0031142d8dc81dce467c26538cf25e9dfa6f2ba59b54d64f5c5a62fe59bf","total":11357,"completed":11357}
{"status":"verifying sha256 digest"}
{"status":"writing manifest"}
{"status":"success"}
```

<!-- curl: PullRequest -->
```shell
curl http://localhost:11435/api/pull -d '{"model": "laya:en", "stream": false}'
```

<!-- json: ProgressResponse -->
```json
{"status": "success"}
```

**Idempotency.** A pull converges on one state: the model at the registry's current manifest is
present locally. Repeating it downloads only missing blobs, and it picks up a moved tag, which is
intended. **Safe to retry.** A **concurrent duplicate** pull (same canonical name) **joins** the
pull in flight: the second caller gets every later line and the same terminal line. A pull of a name
that `/api/create` is writing gets `409 OPERATION_IN_PROGRESS`. See
[§11](#11-idempotency-and-retries).

### 7.7 `DELETE /api/delete`

Removes a local model name. Blobs no other model references are deleted too. A loaded model is
unloaded once its in-flight requests finish. Deleting a router removes only the router; its targets
stay.

| Request field | Type | Required | Rules |
|---|---|---|---|
| `model` | string | yes | [§3](#3-model-names-and-resolution) |

Response: `200` with an empty body, as in Ollama.

<!-- curl: DeleteRequest -->
```shell
curl -X DELETE http://localhost:11435/api/delete -d '{"model": "triage"}'
```

Errors: `404 MODEL_NOT_FOUND` if the name does not exist, `409 OPERATION_IN_PROGRESS` while a pull
or create writes it, `422 INVALID_REQUEST`, `500 STORAGE_ERROR`.

**Idempotency.** The effect is idempotent: after one or many calls the model is gone. The *response*
is not: a repeat gets `404`, as in Ollama. That `404` tells `ollaya rm lyaa` that the name was a
typo, instead of pretending that a model was removed. A client retrying a delete after a timeout
MUST treat `404 MODEL_NOT_FOUND` as success. This deliberately deviates from the skill's "succeeds
even if already deleted".

### 7.8 `POST /api/copy`

Copies a local model to a new name. The destination manifest is written atomically. If the
destination already exists, it is **overwritten**, as in Ollama.

| Request field | Type | Required | Rules |
|---|---|---|---|
| `source` | string | yes | [§3](#3-model-names-and-resolution); must exist |
| `destination` | string | yes | [§3](#3-model-names-and-resolution) |

Response: `200` with an empty body.

<!-- curl: CopyRequest -->
```shell
curl http://localhost:11435/api/copy -d '{"source": "laya:en", "destination": "my-guardrail"}'
```

Errors: `404 MODEL_NOT_FOUND` (source), `409 OPERATION_IN_PROGRESS` (a pull or create writes the
destination), `422 INVALID_REQUEST`, `500 STORAGE_ERROR`.

**Idempotency.** The final state depends only on the request, so repeats converge. **Safe to
retry.** Copying a model onto itself is a no-op `200`.

### 7.9 `POST /api/create`

Creates a model from a local model plus an embedded question schema, calibration, parameters and
license. It is the API behind `ollaya create -f Modelfile`: the CLI reads the Modelfile and the files
it names, and sends their contents as JSON.

| Request field | Type | Required | Default | Rules |
|---|---|---|---|---|
| `model` | string | yes | — | Name to create ([§3](#3-model-names-and-resolution)) |
| `from` | string | yes | — | Local base model, possibly a router. It must exist (`404`); it is never pulled. |
| `questions` | object | no | inherited | Validated as in [§5.2](#52-question-schema) (1–256 questions); becomes the embedded schema |
| `calibration` | object | no | inherited | `temperature`: array of up to 3 numbers (choice, score, noul). `temperature_by_options`: object of `"<type>:<2\|3-5\|6-10\|11+>"` → number. |
| `parameters` | object | no | inherited | Closed set: `precision` (`"fp16"` or `"fp32"`) pins one graph. Other keys are `parameter` issues. |
| `license` | string \| array of string | no | inherited | License text(s); several are joined with a blank line |
| `description` | string | no | inherited | One line, shown by `/v1/models` and `ollaya show` |
| `stream` | boolean | no | `true` | As in `/api/pull` |

Unlike Ollama, there is no `files` map and no `/api/blobs` upload. A decision model's own layers are
small JSON documents, so they travel inline and a two-step upload protocol is not needed. Weights
always come from a registry.

**Streaming response.** `ProgressResponse` lines with `status`: `using existing layer sha256:<hex>`
(one per inherited layer), `creating new layer sha256:<hex>` (one per new layer), then
`writing manifest` and `success`. Errors follow [§4.3](#43-errors-inside-a-stream); validation and
`from` errors come before the stream, as ordinary HTTP errors.

<!-- curl: CreateRequest -->
```shell
curl http://localhost:11435/api/create -d '{
  "model": "triage",
  "from": "laya:en",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": ["billing", "technical", "account"]
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
    }
  },
  "parameters": {"precision": "fp32"},
  "license": "Apache-2.0",
  "description": "Support ticket triage"
}'
```

<!-- ndjson: ProgressResponse -->
```json
{"status":"using existing layer sha256:59d09cddf0644c2ceff24ee5afc82839752fd0deb5bf102899f6ecea83f4b616"}
{"status":"using existing layer sha256:8d32a80bb199bcd4ff10abc28d651fe576fb59f86b24039402e49be9e01578c2"}
{"status":"using existing layer sha256:3da1a3bd1aa2dd93a4d5a50b8f6b00f1a8453cc17cc16df75e1159476c71eed3"}
{"status":"using existing layer sha256:39a9e0ae1223400122965307ee5a126b8bd925c5023a2cb3654066f5fd998bdb"}
{"status":"using existing layer sha256:3a1f403956d84101d442241a098093f9e93d6aa827ca1301b8847d926bf9da83"}
{"status":"creating new layer sha256:0afd81868e80e6f7d47be3d5f87ee7bd27c74acdee4d755d092e49284c34f5f8"}
{"status":"creating new layer sha256:b30d43f3a6156239c92db95a7aeedbd4f5331e96a7e2aa4b429be1df858e4104"}
{"status":"creating new layer sha256:6c7aaa8f2a7669fd4a4a400f0bde4753a8f5ec84d46ff5ed35a69fc278d15def"}
{"status":"writing manifest"}
{"status":"success"}
```

Errors: `404 MODEL_NOT_FOUND` (`from`), `409 OPERATION_IN_PROGRESS`, `422 INVALID_REQUEST`,
`500 STORAGE_ERROR`.

**Idempotency.** Layers are content-addressed, so the same request produces the same manifest.
Repeats converge, and it is **safe to retry**. A concurrent create or pull of the same name gets
`409 OPERATION_IN_PROGRESS`: two definitions racing for one name are conflicting intents, and
joining one would silently serve the other caller the wrong result ([§11](#11-idempotency-and-retries)).

### 7.10 `GET /api/ps`

Models loaded in memory, sorted by name. Routers never appear; their loaded targets do. Models that
share a digest share one runner and appear once, under the name they were last loaded by. No
request body. Not paginated: the list is bounded by `OLLAYA_MAX_LOADED_MODELS`.

| Response field | Type | Meaning |
|---|---|---|
| `models[].name`, `models[].model` | string | Canonical name |
| `models[].size` | integer | Estimated memory of the loaded model, RAM plus VRAM, in bytes |
| `models[].digest` | string | Model digest, bare hex |
| `models[].details` | [details](#model-details) | |
| `models[].expires_at` | string \| `null` | When the model unloads if no further request arrives. While a request is running, the current time: the timer starts when the last request finishes. `null` when kept forever (`keep_alive` < 0). Ollama sends a far-future date instead; `null` is explicit. |
| `models[].size_vram` | integer | Part of `size` in GPU memory; `0` on CPU |
| `models[].context_length` | integer | The model's `max_len` in tokens |
| `models[].device` | string | Where the runner computes: `cpu`, `cuda:0`, … |

```shell
curl http://localhost:11435/api/ps
```

<!-- json: PsResponse -->
```json
{
  "models": [
    {
      "name": "laya:en",
      "model": "laya:en",
      "size": 1203765248,
      "digest": "e1b74e2bbeed9fcbbfc7e6ad44a5cbb38d66f7f24f321f699119b56cc0f7158a",
      "details": {
        "parent_model": "",
        "format": "onnx",
        "family": "laya",
        "families": ["laya"],
        "parameter_size": "421M",
        "quantization_level": "F16"
      },
      "expires_at": "2026-09-24T09:40:12.418Z",
      "size_vram": 1203765248,
      "context_length": 512,
      "device": "cuda:0"
    },
    {
      "name": "laya:multilingual",
      "model": "laya:multilingual",
      "size": 980418560,
      "digest": "d26c30144bb3c5decfa624594549ba8479fa0874bb94cf68416ad1d0d7e850fa",
      "details": {
        "parent_model": "",
        "format": "onnx",
        "family": "laya",
        "families": ["laya"],
        "parameter_size": "322M",
        "quantization_level": "F16"
      },
      "expires_at": null,
      "size_vram": 980418560,
      "context_length": 1024,
      "device": "cuda:0"
    }
  ]
}
```

Safe to retry.

### 7.11 Reserved endpoints

`POST /api/push`, `HEAD /api/blobs/:digest` and `POST /api/blobs/:digest` return
`501 NOT_IMPLEMENTED`. The registry is static (manifests are files published with the site), so
there is nothing to push to yet. The paths are reserved with Ollama's semantics, so adding them
later changes nothing else.

---

## 8. TypeSafe-compatible API

`/v1/*` is **wire-identical to TypeSafe's API**, as defined by the wire schema and error parsing
of `typesafe-sdk` 0.7.1. The official SDKs work unchanged:

```shell
export TYPESAFE_BASE_URL=http://localhost:11435
export TYPESAFE_API_KEY=local           # the SDK requires a non-empty key; any value works unless OLLAYA_API_KEY is set
export TYPESAFE_DEFAULT_MODEL=laya      # otherwise the SDK's default "jev-latest" gets 404 MODEL_NOT_FOUND
```

Rules for `/v1/*`:
- Request: the shared body of [§5.1](#51-request-body-shared-by-the-decision-endpoints) and nothing
  else. Every other field is ignored, native ones (`keep_alive`, `extras`) included.
- Response: **exactly** `model`, `answers` and `usage`, with the answer shapes of
  [§5.4](#54-answer-shapes). No native fields: Ollaya-only data goes on `/api/*`.
- Errors: the [error body](#41-error-body). `extract_message` shows `error`, and `422` carries
  TypeSafe's `detail` list.
- Headers: `x-typesafe-request-id` on every response.

### 8.1 `POST /v1/systemone`

Request: [§5.1](#51-request-body-shared-by-the-decision-endpoints). `state` is required (a missing
`state` is a `missing` issue, as in FastAPI).

| Response field | Type | Meaning |
|---|---|---|
| `model` | string | Canonical name of the model that answered (route target for a router) |
| `answers` | object id → answer | [§5.4](#54-answer-shapes), in question order |
| `usage.input_tokens` | integer | As in `/api/decide` |
| `usage.output_tokens` | integer | `0` |

<!-- curl: SystemOneRequest -->
```shell
curl http://localhost:11435/v1/systemone \
  -H "Authorization: Bearer local" \
  -H "Content-Type: application/json" \
  -d '{
  "model": "laya",
  "state": {"subject": "Invoice", "message": "Can I get an invoice for last month?"},
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "What does the customer want?",
      "criteria": {
        "invoice": "Needs an invoice or receipt",
        "refund": "Wants money back",
        "other": "Anything else"
      }
    },
    "billing": {"type": "noul", "instructions": "Is this message about billing?"}
  }
}'
```

<!-- json: SystemOneResponse -->
```json
{
  "model": "laya:en",
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "invoice",
      "confidence": 0.8995,
      "probabilities": {"invoice": 0.933, "refund": 0.021, "other": 0.046}
    },
    "billing": {"type": "noul", "noul": 0.9641}
  },
  "usage": {"input_tokens": 71, "output_tokens": 0}
}
```

Errors: as `/api/decide`. Safe to retry: the SDK retries `5xx` and `503` with `Retry-After`, which
suits an idempotent decision.

### 8.2 `POST /v1/decisions`

An exact alias of `/v1/systemone`: same body, same response, same errors.

### 8.3 `GET /v1/models`

The local models, in TypeSafe's `ModelMetadataList` shape, sorted by name. Routers are included
(their names are valid `model` values). Not paginated: TypeSafe's shape has no pagination.

| Response field | Type | Meaning |
|---|---|---|
| `models[].name` | string | Canonical name, usable as `model` |
| `models[].description` | string | From the model's config; `""` if none |
| `models[].release_date` | string | `YYYY-MM-DD`: the config's `release_date`, else the UTC date it was pulled or created |

```shell
curl http://localhost:11435/v1/models -H "Authorization: Bearer local"
```

<!-- json: ModelList -->
```json
{
  "models": [
    {
      "name": "laya:en",
      "description": "Laya decision model for English (ModernBERT-large, 512 tokens)",
      "release_date": "2026-09-20"
    },
    {
      "name": "laya:latest",
      "description": "Routes each request to laya:en or laya:multilingual by script and language",
      "release_date": "2026-09-20"
    },
    {
      "name": "laya:multilingual",
      "description": "Laya decision model for 100+ languages (mmBERT-base, 1024 tokens)",
      "release_date": "2026-09-20"
    }
  ]
}
```

---

## 9. Router models

A router (`format: "router"`, e.g. `laya` = `laya:latest`) has no weights. For each request it
picks one of its targets (`laya:en` or `laya:multilingual`), which then answers. Routers are one
level deep: a target is never itself a router.

**What the response reports.**
- `model` (`/api/decide` and `/v1/*`): the target that answered, e.g. `laya:en`. TypeSafe's own
  schema says `model` "may differ from the alias supplied in the request". Reporting the checkpoint
  lets a client see which model produced the numbers, which matters because thresholds and quality
  differ between checkpoints.
- `routing` (`/api/decide` only):

| Field | Type | Meaning |
|---|---|---|
| `router` | string | Canonical name of the router that was requested (`laya:latest`) |
| `model` | string | Canonical name of the chosen target (equals the top-level `model`) |
| `route` | string | Route key from the router's `routes` (`english`, `multilingual`). Stable: branch on this. |
| `reason` | string | Why, in words. Informative only: the text may change in any release. |

**Strategy `script`** (`laya:latest`) is ported from `laya/router.py` and `laya/lang.py`, and golden
tests in `crates/ollaya-lang` check the port. The reasons, verbatim:

| Condition on `state` | `route` | `reason` |
|---|---|---|
| No letters | the router's `default` | `no letters detected in state; using default (english)` |
| Mostly non-Latin script | `multilingual` | `non-Latin script (<script>, <pct>% of letters); the English checkpoint cannot read it` |
| Latin script, identified non-English language | `multilingual` | `Latin script but language looks like '<lang>', not English` |
| Latin script, unidentified language with non-English letters | `multilingual` | `Latin script, language not identified but <pct>% non-English letters; not safe for the English checkpoint` |
| English | `english` | `English Latin text` |

`<pct>` is rounded to a whole number (`%.0f`), and `<lang>` is quoted as Python's `%r` quotes it.
laya's typed-decisions workflow detection is opt-in in laya and is not part of `laya:latest`;
request `laya:typed-decisions` directly.

Routing reads `state` only. It never inspects questions, and it costs microseconds, so a router
adds no measurable latency. Model-specific limits ([§5.3](#53-model-specific-limits)) and
`keep_alive` apply to the target.

---

## 10. Concurrency, queueing, timeouts and cancellation

- **Capacity.** A loaded model's runner computes one forward pass at a time; each pass answers
  every question of a request. At most `OLLAYA_MAX_LOADED_MODELS` models are loaded (default `3`).
- **Queue.** Decision requests (`/api/decide`, `/v1/*`) that cannot run immediately wait: for a busy
  runner, for a model that is loading, or for a slot when the load limit is reached. At most
  `OLLAYA_MAX_QUEUE` decision requests (default `512`) are in flight, waiting or running, across
  all models. A request beyond that gets
  `503 QUEUE_FULL` with `Retry-After: 1`, immediately and without being queued. `503` rather than
  `429` because this is server overload, not a per-client rate limit. The TypeSafe SDK retries both.
- **Loading.** One model loads at a time. Requests for a model that is loading wait for that load;
  they do not start a second one. When the load limit is reached, the least-recently-used idle
  model is unloaded first. If every loaded model is busy, the request waits in the queue.
- **Load timeout.** A load that does not finish within `OLLAYA_LOAD_TIMEOUT` (default `5m`) fails.
  Every request waiting for it gets `500 MODEL_LOAD_FAILED`, and the next request retries the
  load.
- **No inference timeout** on the server, as in Ollama. The client owns its deadline.
- **Client cancellation** (the connection closes):

| When | Effect |
|---|---|
| Waiting (for a load or a busy runner) | The load **continues** and the model stays loaded under the request's `keep_alive`; the decision itself is skipped. The TypeSafe SDK times out after 10 s and retries, so its retry finds the model warm instead of starting over. |
| Running | The forward pass completes (a batch cannot be interrupted), the result is discarded and `keep_alive` is applied |
| Pull (either mode) | Detached from the pull. When the last attached client leaves, the pull stops; partial blobs stay on disk and the next pull resumes them. |
| Create | Stops. Nothing becomes visible, because the manifest is written last. |

## 11. Idempotency and retries

No endpoint accepts an `Idempotency-Key`. Every state-changing endpoint is **naturally idempotent**:
it declares a target state (this model present, this name gone, this name equal to that model), and
repeating it converges on the same state. None creates a resource with a server-generated identity,
so there is no key for a client to derive and nothing for the server to remember. That is the case
where the skill's key machinery is needed, and it does not arise here.

| Endpoint | Effect idempotent | Repeat response | In-flight duplicate | Safe to retry |
|---|---|---|---|---|
| `POST /api/decide`, `/v1/*` | yes (no stored effect; residency converges) | same shape; numbers equal for the same model and precision | runs again | yes |
| `POST /api/pull` | yes (converges on the registry's current manifest) | `success` | same name: **joins** the in-flight pull. During a create of that name: `409` | yes |
| `DELETE /api/delete` | yes | `404 MODEL_NOT_FOUND`; treat as success on retry | `409` while a pull or create writes it | yes, with that rule |
| `POST /api/copy` | yes (last write wins) | `200` | `409` if a pull or create writes the destination | yes |
| `POST /api/create` | yes (content-addressed) | same manifest | `409 OPERATION_IN_PROGRESS` | yes |

**How the guards are built** (the skill's "claim atomically" rule, applied to the in-flight table):

- **One table.** The daemon keeps one in-flight table keyed by **canonical model name**. A pull or
  create claims its name with a single insert-if-absent under one lock. There is no separate check
  followed by an act, so no TOCTOU window.
- **Joining is guarded by payload.** A pull joins only a pull: its payload is the name alone, so
  the intent is identical. A create of the same name has a different payload and gets `409`, instead
  of silently receiving another request's result.
- **Shared blobs.** A blob already in the store is never downloaded again, so models that share
  weights (`laya:en` and `laya:en-fp32`) download them once.
- **Crash safety.** Blobs are written to a temporary file, verified against their sha256, then
  renamed into place. Manifests are written last, atomically. A crash therefore leaves either the
  old state or the new one, never a half-written model. Partial downloads are kept, and the next
  pull resumes them.
- **Retention.** The table lives only as long as the operation; there is no key to retain and no
  replay path (no queue, no dead-letter) that could deliver an operation twice.

## 12. Versioning and evolution

- **One version at a time** (the One-Version rule). `/api/*` has no version prefix, as in Ollama.
  `/v1/*` is TypeSafe's namespace: if TypeSafe publishes `/v2`, Ollaya adds it alongside and keeps
  `/v1`.
- **`GET /api/version`** reports the server's SemVer version. Clients that need a newer feature
  compare versions. Each field added after 0.1.0 is marked *since x.y.z* in this document.
- **Additive changes** can ship in any release:
  - new endpoints;
  - new optional request fields;
  - new response fields;
  - new values in **open** sets: error `code`, issue `type`, pull and create `status`,
    `capabilities`, `details.format`, `quantization_level`, `routing.route`;
  - new values in closed **request** sets (`extras`);
  - relaxed validation, such as accepting more options.
- **Breaking changes** are these. They are avoided; if one is unavoidable before 1.0 it bumps the
  minor version and is listed under "Breaking" in the release notes, and after 1.0 it needs a new
  major version.
  - removing or renaming a field;
  - changing a field's type, unit or meaning (for example redefining `confidence`);
  - making an optional request field required;
  - tightening validation;
  - changing the code or status for an existing condition;
  - changing a default (`keep_alive`, `stream`);
  - changing a documented status string;
  - changing the answer shape on `/v1/*` in any way;
  - adding an unsolicited new answer `type`. A new question type may add its own answer type,
    because only callers who ask the new question receive it.
- **Deprecation.** A deprecated field keeps working for at least one minor release. Its
  documentation says so, and it is removed only under the breaking-change rule above.
- **Clients MUST:**
  - ignore unknown response fields and tolerate unknown values in open sets;
  - branch on `code`, `route` and `done_reason`, never on `error`, `msg` or `reason` text;
  - not depend on top-level field order. Order *is* guaranteed inside `answers`,
    `probabilities`, `legend` and `questions`.

## 13. Compatibility guarantees

### TypeSafe (`/v1/*`)

Guaranteed. `crates/ollaya-api/tests/typesafe_schema.rs` checks these against the schema
examples and the `extract_message` logic of `typesafe-sdk` 0.7.1. Running the Python SDK itself
against a live daemon is part of the end-to-end verification.

- **Requests.** Every body the SDK can send is accepted with TypeSafe's meaning, within
  [§5](#5-questions-and-validation). That covers `instructions` of any allowed type or absent,
  noul `criteria` absent, and `extra_body` fields, which are ignored.
- **Responses** are exactly `{model, answers, usage}`:
  - the answer shapes, and their field order;
  - `noul` has no `confidence`;
  - `confidence` uses TypeSafe's formula;
  - values are rounded to 4 decimals;
  - keys are in criteria order.
- **Status codes** map onto the SDK's exception classes as described in [§4.2](#42-error-codes).
  Every error body gives a readable message through `extract_message`, and `422` bodies are valid
  `HTTPValidationError`s.
- **Auth.** Any API key is accepted, unless `OLLAYA_API_KEY` is set.
- **Headers.** `x-typesafe-request-id` is sent.

Documented differences, which are not bugs:

- **Model names.** They are Ollaya's (`laya`, `laya:en`), so set `TYPESAFE_DEFAULT_MODEL`.
- **Answering model.** `model` in the response is the checkpoint that answered.
- **Missing `instructions`.** The model reads the question id ([§5.2](#52-question-schema)).
- **Limits.** Ollaya enforces 256 questions per request, and some models have their own limits
  ([§5.3](#53-model-specific-limits)). TypeSafe's questions-per-request limit is not published.
- **Quality.** The answers come from open models, so quality differs by task.

### Ollama (`/api/*`)

Mirrored as Ollama has them:

- **Transport.** Paths and methods (including the verb paths `/api/pull`, `/api/show` and
  `DELETE /api/delete`), `snake_case` names, and bodies parsed as JSON whatever their
  `Content-Type`.
- **Errors and streaming.** `{"error": "<message>"}` bodies, and NDJSON streaming on by default
  with `"stream": false`.
- **Pull progress.** The shape and status strings.
- **Responses and timing.** `GET /` text, `/api/version`, the `/api/tags` and `/api/ps` shapes
  (`name`, `model`, `modified_at`, `size`, bare-hex `digest`, `details`), `keep_alive`, load and
  unload through the inference endpoint, and nanosecond durations.
- **Messages.** The `not found, try pulling it first` message.
- **Browser access.** CORS defaults and the `Host` allowlist.

Deviations, each needed because decision models differ from LLMs or because the API is safer this
way:

| Deviation | Reason |
|---|---|
| `/api/decide` replaces `generate`/`chat`/`embed` | Decision models answer typed questions; they never generate text |
| Error bodies add `code` (and `detail`) | Machine-readable errors; additive, so Ollama clients are unaffected |
| `/api/pull` reports pre-stream failures as HTTP errors | Real status codes for scripts; streamed errors are still handled |
| `/api/create` takes structured JSON, no `files`/blobs | Decision-model layers are small JSON documents |
| `expires_at: null` for "forever" | Explicit, instead of a sentinel date |
| `/api/ps` never lists routers | A router holds no memory; its targets do |
| `show` adds `questions`, `router`; drops `template` | Decision models have question schemas and routing, not prompt templates |
| `OLLAYA_API_KEY` | laya-app's deployment is protected by a bearer token, and replacing it must not remove that protection ([§14](#14-security)) |

## 14. Security

**No authentication by default.** Like Ollama, the daemon binds to `127.0.0.1:11435` and trusts
local callers.

**`OLLAYA_HOST=0.0.0.0` (or any non-loopback address) is a risk.** Everyone who can reach the port
can:
- run decisions;
- pull arbitrary models, filling the disk and using the bandwidth;
- delete, copy and create models;
- keep models loaded with `keep_alive: -1`.

Therefore:

- **`OLLAYA_API_KEY`.** When set, every request except `GET /`, `HEAD /` and CORS preflight
  (`OPTIONS`) must carry `Authorization: Bearer <key>`. Otherwise the response is
  `401 UNAUTHORIZED` with `WWW-Authenticate: Bearer`. The key is compared in constant time. The
  TypeSafe SDK already sends `Bearer $TYPESAFE_API_KEY`, and the `ollaya` CLI and `ollaya-api`
  client send `$OLLAYA_API_KEY`. This keeps the protection laya-app has today (`LAYA_TOKEN`) for
  the shared GPU deployment.
- **Startup warning.** When the bind address is not loopback and `OLLAYA_API_KEY` is unset, the
  server logs a warning at startup. It does not refuse to start: containers bind `0.0.0.0` behind
  port mapping, as Ollama's image does.
- **TLS** is not terminated by the daemon. Put a reverse proxy in front for remote access.

**Browsers (CORS and DNS rebinding)**, mirroring Ollama:

- **No `Origin` header.** Requests without one (curl, SDKs, the CLI) are allowed.
- **Allowed origins.** A request with an `Origin` header is allowed only if the origin matches the
  allowlist:
  - the defaults: `http(s)://localhost`, `http(s)://127.0.0.1`, `http(s)://0.0.0.0` and
    `http(s)://[::1]`, each with any port or none, plus `app://*`, `file://*`, `tauri://*`,
    `vscode-webview://*` and `vscode-file://*`;
  - the entries in `OLLAYA_ORIGINS`: comma-separated, with `*` as a wildcard, and a bare `*`
    allows every origin.

  Any other origin gets `403 FORBIDDEN`.
- **CORS headers.**
  - Preflight `OPTIONS` answers `204`.
  - Allowed request headers: `Authorization`, `Content-Type`, `Accept`, `User-Agent`,
    `X-Requested-With`, `X-Request-Id`, `X-TypeSafe-SDK`, `X-TypeSafe-Runtime` and
    `X-TypeSafe-Retry-Count`.
  - Exposed headers: `X-Request-Id`, `x-typesafe-request-id` and `Retry-After`.
- **Host check.** When bound to a loopback address, the server rejects a `Host` header that is not
  one of these (`403 FORBIDDEN`):
  - empty;
  - `localhost`, or a name ending in `.localhost`, `.local` or `.internal`;
  - the machine's hostname;
  - a loopback, private or unspecified IP.

  This blocks DNS-rebinding attacks from web pages.

**Input and data handling.**
- **Limits.** The body limit (8 MiB), the question and option limits and the 65,536-token state
  limit bound the work one request can cause.
- **Registry data is untrusted.** Every blob is verified against its sha256 before use, and
  manifests and JSON layers are validated when read.
- **User data stays out of logs and errors.** `state` and question text are never logged, and are
  not echoed in errors (no `input` in issues).
- **Internal errors** (`500 INTERNAL`) never include paths, stack traces or runner output. The
  server log has them under the request ID.

## 15. Environment variables

These are the variables that change API behaviour.

| Variable | Default | Effect |
|---|---|---|
| `OLLAYA_HOST` | `127.0.0.1:11435` | Server bind address; the client's target. `http://` is assumed if no scheme is given. A missing port means `11435` for `http` and `443` for `https`; an empty host means `127.0.0.1`. A path is kept as a prefix. |
| `OLLAYA_API_KEY` | unset | Require `Authorization: Bearer <key>` ([§14](#14-security)); the client sends it |
| `OLLAYA_ORIGINS` | unset | Extra allowed browser origins |
| `OLLAYA_KEEP_ALIVE` | `5m` | Default `keep_alive` ([§6](#6-keep_alive)) |
| `OLLAYA_MAX_LOADED_MODELS` | `3` | Loaded-model limit |
| `OLLAYA_DEVICE` | `auto` | Runner device: `auto` (CUDA if available, else CPU), `cpu`, `cuda`, `cuda:<n>` |
| `OLLAYA_MAX_QUEUE` | `512` | Queue bound before `503 QUEUE_FULL` |
| `OLLAYA_LOAD_TIMEOUT` | `5m` | Load deadline before `500 MODEL_LOAD_FAILED` |
| `OLLAYA_MODELS` | `~/.ollaya/models` | Model store |
| `OLLAYA_REGISTRY` | `ollaya.cobanov.dev` | Default registry host in names |

## 16. Verification checklist

The checklist from the `api-and-interface-design` skill, answered item by item. Every "deliberate"
answer has its reason next to it; `.claude/skills/PROJECT_NOTES.md` records the project-level
overrides.

- [x] **Every endpoint has typed input and output schemas.** Each endpoint in §7–§8 has request
  and response tables. `crates/ollaya-api` has a serde type for each, and the tests round-trip
  every example in this file through those types.
- [x] **Error responses follow a single consistent format.** One body
  (`error` + `code` + optional `detail`) on every endpoint, `/v1/*` included. Mid-stream failures
  are sent as a line in the same shape.
- [x] **Validation happens at system boundaries only.**
  - HTTP input is validated once, by `ollaya_api::validate`, at the handler.
  - Model-specific limits are checked where the request becomes model input (tokenization), which
    is the boundary of the model.
  - Registry data (third-party) is validated on pull: digests, manifests and JSON layers.
  - Internal code trusts the typed values.
- [ ] **List endpoints support pagination.** *Deliberate deviation.*
  - The lists are small and local: models on one disk (tens), loaded models (at most
    `OLLAYA_MAX_LOADED_MODELS`), and the same set again in `/v1/models`.
  - Ollama's `/api/tags` and TypeSafe's `ModelMetadataList` are fixed unpaginated shapes. A `data` +
    `pagination` envelope would break both (Hyrum's Law); `PROJECT_NOTES.md` requires unpaginated
    `/api/tags`.
  - If a list ever needs paging, it will be added additively: optional `limit` and `cursor` query
    parameters and an optional `next_cursor` field, and the full list without them. That is a
    non-breaking change by §12.
- [x] **New fields are additive and optional.** §12 defines additive and breaking changes. Native
  additions to TypeSafe's response live only on `/api/decide`, and never redefine a TypeSafe field
  (laya's confidence is namespaced under `laya`).
- [x] **Naming follows consistent conventions across all endpoints**, with two deliberate
  deviations from the skill's table.
  - **Consistent:**
    - `snake_case` fields everywhere;
    - `UPPER_SNAKE` error codes;
    - lower-case enum values on the wire;
    - canonical model names in every model-naming field.
  - **Deviations:**
    - Verb paths (`/api/pull`, `/api/show`, `/api/delete`) and `snake_case` rather than camelCase
      mirror Ollama and TypeSafe, whose clients must work unchanged.
    - Booleans keep Ollama's names (`stream`, `insecure`) without `is_`/`has_` prefixes, and the
      new boolean `state_truncated` follows the same style.
- [x] **API documentation or types are committed alongside the implementation.** This file and
  `crates/ollaya-api`. Their tests fail if an example in this file stops matching the types.
- [x] **State-changing endpoints either honour an idempotency key or are documented as unsafe to
  retry.** Neither is needed: every state-changing endpoint is naturally idempotent and documented
  as safe to retry (§11). `DELETE` has an explicit retry rule (a `404` on retry means success).
- [x] **The key is claimed in one atomic operation, guarded by a unique constraint.** There are no
  keys (§11). The in-flight table has the same property: one insert-if-absent under one lock, keyed
  by canonical name.
- [x] **A reused key with a different payload fails loudly rather than replaying the wrong
  response.** For the in-flight table: a pull joins only a pull of the same name, whose payload
  is the name alone. A create, or any write to a name being written, gets
  `409 OPERATION_IN_PROGRESS`.
- [x] **The in-flight-duplicate response is a deliberate choice.**
  - Pull: **wait** (join), because the caller needs the result and the intent is identical.
  - Create, and delete or copy against a name being written: **409**, because the intents may
    conflict.
- [x] **Key retention outlives the longest retry path, including dead-letter replay.** Not
  applicable: nothing is keyed or retained, and nothing replays operations (no queue or
  dead-letter path). Retries converge because every effect is idempotent, not because anything is
  remembered.

---

## Appendix A: changes required in the site drafts

The drafts were written before this contract. They MUST change as follows.

**`site/docs/api.md`**

1. **Errors.** Document `code` (and `detail` on `422`) next to `error`, add the code table, and
   add mid-stream error lines. The draft's message `model 'laya:xl' not found` becomes
   `model "laya:xl" not found, try pulling it first`.
2. **Decide request.**
   - `keep_alive` also accepts numbers, `0` and negative values.
   - Add `extras`, and `stream` (reserved).
   - `instructions` is optional; "every question has instructions" is wrong.
   - The limits are 2–255 choices ("up to 255" is wrong), 2–10 levels, 1–256 questions and a
     65,536-token state.
   - `choice` criteria also accept a list of labels.
3. **Decide response.**
   - `model` is the answering model (`laya:en`), not `laya`.
   - `routing` is `{router, model, route, reason}`, or `null`.
   - Add `state_truncated`, `done_reason`, `created_at` and `eval_duration`.
   - `probabilities` are in criteria order, not sorted by value, and every value has 4 decimals.
     The draft's example is internally inconsistent: `confidence` does not match its
     probabilities.
   - Document load and unload through `/api/decide`.
4. **`/api/tags`.** Entries have `name`, `model`, `modified_at`, `size`, `digest` (bare hex, not
   `sha256:…`) and `details`.
5. **`/api/show` and `/api/ps`.** Replace the prose with the field tables (`questions`, `router`,
   `model_info`, `capabilities`; `expires_at` may be `null`).
6. **`/api/pull`.** Pre-stream failures are HTTP errors. For routers, there is one `success`.
7. **`/api/delete`.** Returns `404` for a missing model; `/api/copy` overwrites the destination.
8. **`/api/create`.** The body is structured (`model`, `from`, `questions`, `calibration`,
   `parameters`, `license`), not a `modelfile` string.
9. **`/api/push`.** It is reserved (`501`), not available.
10. **Conventions.** `Content-Type` is not required. Add `OLLAYA_API_KEY`, `OLLAYA_ORIGINS` and
    the security notes.

**`site/docs/typesafe-compatibility.md`**

1. **SDK setup.** Also set `TYPESAFE_API_KEY` (any non-empty value; the SDK refuses to start
   without one) and `TYPESAFE_DEFAULT_MODEL=laya` (the SDK otherwise sends `jev-latest`).
2. **Response `model`.** It is the checkpoint that answered (`laya:en`), which TypeSafe's schema
   allows. The example should show that and use 4-decimal values in criteria order.
3. **Differences.** Document the missing-instructions rule, the limits, and the fact that errors
   carry `error` + `code` (with TypeSafe's `detail` on `422`).
4. **`/v1/models`.** It lists local models only (`name`, `description`, `release_date`).
5. **`x-typesafe-request-id`.** Mention that it is sent, so `response.request_id` works.
