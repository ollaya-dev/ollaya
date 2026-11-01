---
title: Ollaya Cloud quickstart
nav: Cloud quickstart
description: Make your first decision on Ollaya Cloud with the TypeSafe SDK in a few minutes, with no GPU of your own.
order: 20
---

# Ollaya Cloud quickstart

Ollaya Cloud runs Ollaya's own decision models for you: the same `/v1/*` API as a local Ollaya, at `https://api.{{SITE_HOST}}`, billed per input token. Code that works against a local Ollaya or against TypeSafe works here by changing two environment variables.

## 1. Create a key

Sign in at **{{SITE_HOST}}/dashboard** with your email, Google, or GitHub. A new account gets its own organization. Open **Keys**, choose **Create key**, and copy the key: it starts with `oly_live_` and is shown only once.

## 2. Point the SDK at Ollaya Cloud

```shell
pip install typesafe-sdk
export TYPESAFE_BASE_URL=https://api.{{SITE_HOST}}
export TYPESAFE_API_KEY=oly_live_...        # the key from step 1
export TYPESAFE_DEFAULT_MODEL=coba:400m     # otherwise the SDK sends its default, "jev-latest"
```

## 3. Make a decision

```python
from typesafe_sdk import Noul, TypeSafeClient

client = TypeSafeClient()
answer = client.system_one(
    "I was charged twice for my subscription this month. Please refund one of the charges.",
    {"refund": Noul(instructions="Does the customer ask for money back?")},
)
print(answer.nouls["refund"].noul, answer.request_id)
```

The same request with curl:

```shell
curl https://api.{{SITE_HOST}}/v1/systemone \
  -H "Authorization: Bearer $TYPESAFE_API_KEY" \
  -d '{
  "model": "coba:400m",
  "state": "I was charged twice for my subscription this month. Please refund one of the charges.",
  "questions": {
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}
  }
}'
```

The answer has exactly `model`, `answers`, and `usage`, as TypeSafe's does. `usage.input_tokens` is what you pay for.

## 4. See it in the dashboard

**Usage** shows the request within a few seconds, with its cost; **Requests** lists it by its request ID (never its content, which Ollaya Cloud does not store); **Credits** shows the balance it came out of.

## Next

- [Keys](/docs/cloud-keys): scopes, budgets, allowed models, IP allowlists, and expiry.
- [Limits and headers](/docs/cloud-limits): rate and concurrency limits, retries, deadlines, and tags.
- [Errors](/docs/cloud-errors): every error code and what to do about it.
- [Models](/docs/cloud-models): the models Ollaya Cloud serves and where they run.
- [Data handling](/docs/cloud-data): what is kept, for how long, and where.
