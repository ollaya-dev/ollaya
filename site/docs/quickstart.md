---
title: Quickstart
description: Install Ollaya, run your first decision model and call the local API.
order: 1
---

# Quickstart

Ollaya runs open decision models on your own machine. You give a model a **state** — a message, an email, a ticket, a JSON object — plus a few **typed questions**, and it returns typed answers with calibrated probabilities in a single forward pass.

> Ollaya is pre-release. The commands below describe the first version, which is being built right now. [Watch the repository](https://github.com/cobanov/ollaya) to hear when it ships.

## 1. Install

On Linux and macOS:

```shell
curl -fsSL {{SITE_ORIGIN}}/install.sh | sh
```

Or run the Docker image — see [Download](/download) for all options.

## 2. Run a model

`ollaya run` pulls the model on first use, loads it, and answers:

```shell
ollaya run laya --preset triage "I was charged twice for my subscription this month."
```

`laya` is a router: it inspects the language of the state and runs `laya:en` or `laya:multilingual`. `--preset triage` asks a built-in set of support-triage questions.

## 3. Start the server

```shell
ollaya serve
```

The server listens on `127.0.0.1:11435`. Loaded models stay warm for five minutes after their last request (`keep_alive`), so follow-up calls skip the load.

## 4. Ask your own questions

```shell
curl http://localhost:11435/api/decide \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "state": "Hi, I cannot log in since this morning and I have a demo at 3pm.",
    "questions": {
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
        "criteria": ["Not urgent", "Normal", "Urgent"]
      },
      "angry": {
        "type": "noul",
        "instructions": "Is the customer angry?"
      }
    }
  }'
```

Every answer comes back typed: a `choice` with probabilities per option, a `score` as an expected value over the levels, and a `noul` as the probability of *true*. See the [API reference](/docs/api) for the full format.

## 5. Use an existing TypeSafe client

Ollaya also serves TypeSafe's API. Point any TypeSafe SDK at your local server:

```shell
export TYPESAFE_BASE_URL=http://localhost:11435
```

See [TypeSafe compatibility](/docs/typesafe-compatibility).

## 6. Bake your questions into a model

Write a [Modelfile](/docs/modelfile) once and run it by name:

```shell
ollaya create triage -f Modelfile
ollaya run triage "I was charged twice for my subscription this month."
```

## Next steps

- [CLI reference](/docs/cli)
- [API reference](/docs/api)
- [Browse models](/search)
