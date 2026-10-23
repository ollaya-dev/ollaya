---
title: Routing: pick the model for each request
nav: Routing
description: Route each request to the right LLM, tool or team with a local decision model. Difficulty, domain, tool use and sensitivity come back typed and calibrated, in milliseconds.
order: 8
---

# Routing

Most requests don't need the largest model. A router decides before the expensive call: which model tier should answer, whether the request needs tools or search, whether it is sensitive enough for a stronger model or a person. A decision model answers all of these at once, in one request that takes milliseconds and no tokens, on your own machine.

## The router preset

The built-in `router` preset asks four questions about `request`:

| Question | Type | Answer |
|---|---|---|
| `difficulty` | score | 0 trivial (a lookup or one-liner), 1 easy, 2 moderate (several steps), 3 hard (long multi-step reasoning or specialist knowledge) |
| `domain` | choice | `code`, `math_or_logic`, `writing`, `factual_lookup`, `data_analysis` or `chitchat` |
| `needs_tools` | yes/no | Does answering require external tools, search or private data? |
| `is_sensitive` | yes/no | Does it involve money, legal, medical or safety consequences? |

```shell
ollaya run winnow:e4b --preset router '{"request": "Refactor this 2,000-line module into packages"}'
```

## Route from code

```python
import requests

def route(text: str) -> str:
    answers = requests.post("http://localhost:11435/api/decide", json={
        "model": "winnow:e4b",
        "preset": "router",
        "state": {"request": text},
    }).json()["answers"]
    if answers["is_sensitive"]["noul"] > 0.5 or answers["difficulty"]["score"] >= 2:
        return "large"
    return "small"
```

`score` is the expected level, between 0 and 3, and `noul` is the probability of yes. Each answer also carries `probabilities`, and choice and score answers a `confidence`; see the [API reference](/docs/api#decide). When the router isn't sure, send the request to the stronger model: a wrong "small" costs more than a wrong "large".

## Your own routes

The preset's questions are generic. To route to your own teams, tools or prompts, ask a choice question whose options are your routes, each with a description of what belongs there, and save it as a preset that the CLI, the API and the MCP server all see:

```shell
ollaya preset create support-route --questions '{
  "queue": {
    "type": "choice",
    "instructions": "Which queue should handle `request`?",
    "criteria": {
      "billing": "Charges, invoices, refunds and plans",
      "bug": "Errors, crashes and things that stopped working",
      "how_to": "Questions about how to use the product",
      "sales": "Pricing for teams, contracts and demos"
    }
  }
}'
ollaya run winnow:e4b --preset support-route '{"request": "We want 40 seats, can we get an invoice instead of a card payment?"}'
```

## Which model

A router runs on every request, so speed counts as much as accuracy.

- **With an NVIDIA GPU:** `winnow:e4b` answers five questions in {{result winnow:e4b gpu.p50_ms 0}} ms on an RTX 5090 and scores {{result winnow:e4b typed.accuracy 3}} on typed decisions. `pplx-decider:flash`, first on [Ollaya Bench](/results), takes {{result pplx-decider:flash gpu.p50_ms 0}} ms.
- **On a CPU:** `laya:multilingual` takes {{result laya:multilingual cpu.p50_ms 0}} ms and `decima:small` {{result decima:small cpu.p50_ms 0}} ms on a Threadripper 3960X. They are much less accurate than the 4B models on typed decisions ({{result laya:multilingual typed.accuracy 3}} and {{result decima:small typed.accuracy 3}}), so check them on your own requests first.

All the models and their numbers are on the [models page](/search). For agents, the `agent` preset decides whether a tool call should run, ask or be blocked; see [Agents](/docs/agents).
