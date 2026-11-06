---
title: Ollaya Cloud limits, retries, and headers
nav: Cloud limits
description: Rate and concurrency limits per tier, the headers that report them, retries, deadlines, and usage tags on Ollaya Cloud.
order: 23
---

# Limits, retries, and headers

## Limits per tier

Each organization has a request rate (with a burst on top) and a number of requests it may have running at once. The dashboard's **Limits** page shows your tier and its numbers.

| Tier | Requests per second | Burst | Concurrent requests |
|---|---|---|---|
| Free | 5 | 10 | 2 |
| Tier 1 | 50 | 100 | 10 |
| Tier 2 | 200 | 400 | 40 |
| Tier 3 | 1,000 | 2,000 | 150 |
| Enterprise | by contract | by contract | by contract |

A key can have lower limits than its organization (see [Keys](/docs/cloud-keys)). Over the rate, a request gets `429 RATE_LIMITED`; over the concurrency, `429 CONCURRENCY_LIMITED`. Both carry `Retry-After` in seconds.

## Rate-limit headers

Answers carry the IETF rate-limit fields (`draft-ietf-httpapi-ratelimit-headers`, revision 11):

```http
RateLimit-Policy: "rate";q=10;w=2, "concurrent";q=2;qu="concurrent-requests"
RateLimit: "rate";r=9;t=1
```

`RateLimit-Policy` describes your limits: the burst `q` and the seconds `w` to refill it, and the concurrent requests. `RateLimit` says what is left: `r` requests, and `t` seconds until the burst is full again.

## Overload

When a model's replicas are busy enough that a request would wait longer than your tier's limit, or longer than your deadline (below), Ollaya Cloud answers `503 QUEUE_FULL` with `Retry-After` at once instead of queueing it. The limit is two seconds of expected wait for Tier 3 and Enterprise, and a quarter, half, and three quarters of that for Free, Tier 1, and Tier 2, so under load Free traffic is refused first and paid traffic keeps its latency. A request for a model that is not loaded anywhere gets `503 QUEUE_FULL` with `Retry-After` set to the model's load time, while Ollaya Cloud loads it.

## Retries

The TypeSafe SDK retries `429` and `503` after `Retry-After`, and `5xx` answers, so most clients need nothing more. Decisions are idempotent: retrying one is always safe, and only the answer you receive is billed. Ollaya Cloud itself retries an attempt that failed in the worker once on another replica, within a retry budget, before you see an error.

## Deadlines

Send `X-Ollaya-Timeout-Ms: <milliseconds>` with a decision to say how long you will wait. Ollaya Cloud refuses at once (`503 QUEUE_FULL`) a request it could not answer in time, and cuts each attempt to that many milliseconds, instead of answering after you have given up.

## Tags

Send up to 10 tags in `X-Ollaya-Tags` to split usage by your own dimensions, such as a customer or a feature:

```http
X-Ollaya-Tags: team=support, feature=refunds
```

Tags are `key=value` pairs separated by commas: keys of 1 to 32 characters from `a-z 0-9 _ . -`, values of 1 to 64 characters from `A-Z a-z 0-9 _ . : / -`. The management API's usage report groups by them (`group_by=tag:<key>`). A bad tag never fails the request: it is left out, and the answer carries `X-Ollaya-Tags-Dropped: 1`.
