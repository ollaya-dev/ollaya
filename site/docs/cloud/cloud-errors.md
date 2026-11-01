---
title: Ollaya Cloud errors
nav: Cloud errors
description: Every error Ollaya Cloud returns, its HTTP status, and what to do about it.
order: 22
---

# Errors

Every error has the same body as a local Ollaya, `{"error": "<message>", "code": "<CODE>"}`, and the header `X-Ollaya-Error-Code` with the same code. Branch on the code, never on the message: messages may change, codes do not. The TypeSafe SDK raises a typed exception from the status and shows `error` as its message.

Every response carries `X-Request-Id` (your own value when it matches `[A-Za-z0-9._-]{1,128}`); `/v1/*` responses carry it as `x-typesafe-request-id` too, which the SDK exposes as `request_id`. Quote it when you ask about a request.

## Codes

| Code | Status | When | What to do |
| --- | --- | --- | --- |
| `INVALID_JSON` | 400 | The body is not valid JSON | Fix the request |
| `UNAUTHORIZED` | 401 | The key is missing, malformed, unknown, revoked, or expired | Check `Authorization: Bearer oly_live_…` |
| `INSUFFICIENT_CREDITS` | 402 | The organization's balance is used up | Add credit on the **Credits** page |
| `SPEND_CAP_REACHED` | 402 | The key's or the organization's budget is used up | Raise the budget, or wait for the next month |
| `FORBIDDEN` | 403 | Your role does not allow this (management API) | Ask an Owner or Admin |
| `MODEL_NOT_ALLOWED` | 403 | The key or the organization may not call this model | Change the key's or the organization's model access |
| `IP_NOT_ALLOWED` | 403 | The key has an IP allowlist and the caller is outside it | Add the address to the key |
| `BROWSER_NOT_ALLOWED` | 403 | The request came from a browser (`Origin` header) | Call Ollaya Cloud from your server |
| `MODEL_NOT_FOUND` | 404 | The model is not in the catalog | `GET /v1/models` lists the available models |
| `NOT_FOUND` | 404 | The path does not exist | Check the path |
| `METHOD_NOT_ALLOWED` | 405 | The path exists with other methods, listed in `Allow` | Use one of those |
| `OPERATION_IN_PROGRESS` | 409 | A request with the same `Idempotency-Key` is still running (management API) | Retry later |
| `REQUEST_TOO_LARGE` | 413 | The body is larger than 8 MiB, compressed or not | Send a smaller state |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | An unknown content encoding, or more than one | Send `gzip` or `zstd`, or no encoding |
| `IDEMPOTENCY_KEY_REUSED` | 422 | The `Idempotency-Key` was used with a different request (management API) | Use a new key |
| `INVALID_REQUEST` | 422 | The body does not validate; `detail` lists each problem, as FastAPI does | Fix the fields named in `detail` |
| `RATE_LIMITED` | 429 | Over your request rate | Retry after `Retry-After`; the SDK does |
| `CONCURRENCY_LIMITED` | 429 | Over your concurrent request limit | Retry after `Retry-After`; the SDK does |
| `INTERNAL` | 500 | An error in Ollaya Cloud itself | Retry; tell us the request ID if it persists |
| `QUEUE_FULL` | 503 | The model is overloaded, or about to load and not ready yet | Retry after `Retry-After`; the SDK does |

Errors from the model itself (`TOO_MANY_OPTIONS`, `INPUT_TOO_LONG`, `STATE_TRUNCATED`, `INFERENCE_FAILED`, and the rest) pass through with Ollaya's own status and code; see [the API reference](/docs/api#errors). Only a `2xx` answer is billed: no error ever costs anything.
