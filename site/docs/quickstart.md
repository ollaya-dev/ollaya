---
title: Quickstart
description: Install Ollaya, run your first decision model and call the local API.
order: 1
---

# Quickstart

Ollaya runs open decision models on your own machine. You give a model a **state** (a message, an email, a ticket, a JSON object) plus a few **typed questions**, and it returns typed answers with calibrated probabilities in a single forward pass.

## 1. Install

On Linux and macOS:

```shell
curl -fsSL {{SITE_ORIGIN}}/install.sh | sh
```

On Windows, in PowerShell:

```powershell
irm {{SITE_ORIGIN}}/install.ps1 | iex
```

Or get the [desktop app](/download), which bundles the same command line and server.

The script downloads the latest release from GitHub and checks its sha256. On Linux and Windows it also fetches the CUDA libraries when it finds an NVIDIA GPU. On Linux, where systemd runs and it has root rights, it sets up a service that serves the API on `127.0.0.1:11435`. See [Download](/download) for requirements and the Docker images.

## 2. Run a model

```shell
ollaya run winnow:e4b "Third time this year you've double-charged me. Refund it today or I'm cancelling and moving to a competitor."
```

```text
intent            refund                                ███████████████░ 0.91
is_urgent         yes                                   ███████████████░ 0.92
frustration       2.89 / 3  very angry or using stron…  ██████████████░░ 0.86
refund_requested  yes                                   ████████████████ 0.99
churn_risk        yes                                   ████████████████ 0.99
```

`ollaya run` starts the server if it isn't running, pulls the model on first use and loads it. `winnow:e4b` is the recommended model: 0.722 accuracy on typed decisions, close to TypeSafe's Jev (0.738), and 89 ms for these five questions on an RTX 4090. It is an 8 GB download.

No NVIDIA GPU? Start with `laya`, which answers in a fraction of a second on a CPU. It is a router: it sends English text to `laya:en` and other languages, Turkish for example, to `laya:multilingual`, and pulling `laya` pulls both. [Models](/search) compares the accuracy and speed of every model.

- `--preset NAME` asks a built-in question set: `triage`, `email`, `guard`, `moderation`, `router` or `agent`. Without it, a model that has no questions of its own answers `triage`.
- `--verbose` adds every option's probability, the routing decision and timings.
- `--format json` prints the full API response.
- Without a state, `ollaya run` reads piped stdin, or opens a prompt on a terminal.

## 3. Ask your own questions

Write the questions to a file:

```json
{
  "topic": {
    "type": "choice",
    "instructions": "What is this message about?",
    "criteria": {
      "billing": "Payments, invoices and refunds",
      "access": "Login, passwords and permissions",
      "other": "Anything else"
    }
  },
  "urgency": {
    "type": "score",
    "instructions": "How urgent is this?",
    "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
  },
  "angry": {
    "type": "noul",
    "instructions": "Is the customer angry?"
  }
}
```

```shell
ollaya run winnow:e4b --questions questions.json "Hi, I cannot log in since this morning and I have a demo at 3pm."
```

Or skip the file and pass the JSON inline, the way `curl -d` does (a value that starts with `{` is JSON, not a file path):

```shell
ollaya run winnow:e4b --questions '{"angry":{"type":"noul","instructions":"Is the customer angry?"}}' "Hi, I cannot log in since this morning and I have a demo at 3pm."
```

Or send the same questions to the API:

```shell
curl http://localhost:11435/api/decide -d '{
  "model": "winnow:e4b",
  "state": "Hi, I cannot log in since this morning and I have a demo at 3pm.",
  "questions": {
    "angry": {"type": "noul", "instructions": "Is the customer angry?"}
  }
}'
```

Every answer comes back typed: a `choice` with a probability per option, a `score` as the expected level, and a `noul` as the probability that the statement holds. See the [API reference](/docs/api).

## 4. Use an existing TypeSafe client

Ollaya serves TypeSafe's API too. The official TypeSafe Python SDK works unchanged:

```shell
export TYPESAFE_BASE_URL=http://localhost:11435
export TYPESAFE_API_KEY=local        # any value; the SDK needs one
export TYPESAFE_DEFAULT_MODEL=winnow:e4b
```

See [TypeSafe compatibility](/docs/typesafe-compatibility).

## 5. Bake your questions into a model

A [Modelfile](/docs/modelfile) turns a question set into a model you can run by name:

```dockerfile
FROM laya
QUESTIONS ./questions.json
DESCRIPTION Support inbox triage
```

```shell
ollaya create inbox -f Modelfile
ollaya run inbox "Hi, I cannot log in since this morning and I have a demo at 3pm."
```

## Next steps

- [CLI reference](/docs/cli)
- [API reference](/docs/api)
- [Browse models](/search)
