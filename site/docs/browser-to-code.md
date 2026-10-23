---
title: From the browser to your code
nav: Browser to code
description: Take the model you tried in your browser to your own code. Pull coba:150m-ternary with Ollaya and call it from curl, Python, TypeScript or a Docker container.
order: 1.5
---

# From the browser to your code

[Try it in your browser]({{SITE_ORIGIN}}/try) runs `coba:150m-ternary` inside the page. Ollaya runs the same model on your machine or a server, behind an HTTP API your code can call. This page takes the support example from that page to curl, Python, TypeScript and Docker.

## 1. Get the model

Install Ollaya ([Quickstart](/docs/quickstart), or the [desktop app](/download)) and pull the model:

```shell
ollaya pull coba:150m-ternary
```

Then start the server with `ollaya serve`, unless the desktop app or the Linux service already runs it. It listens on `localhost:11435`. Ollaya runs the same weights as the browser, stored as float32; the browser runs a 2-bit graph of them.

## 2. curl

[`POST /api/decide`](/docs/api#decide) takes the text as `state` and your questions, in the same form as the page:

```shell
curl http://localhost:11435/api/decide -d '{
  "model": "coba:150m-ternary",
  "state": "Hi, I was charged twice for my Pro plan this month (invoice #4471). Please refund the duplicate before Friday.",
  "questions": {
    "refund": {"type": "noul", "instructions": "The customer asks for money back."},
    "team": {
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
    }
  }
}'
```

Each answer is under `answers`, by the question's key:
- `refund` is a yes/no question. `noul` is the probability that it is true.
- `team` is a choice question. `choice` is the most likely option, and `probabilities` scores every option.
- `urgency` is a score question. `score` is the expected value on the scale, and `probabilities` scores each step.

The same body also works at `/v1/systemone`, so the TypeSafe SDK and the libraries in [Integrations](/docs/integrations) can call the model too ([TypeSafe compatibility](/docs/typesafe-compatibility)).

## 3. Python

The standard library is enough:

```python
import json
import urllib.request

body = {
    "model": "coba:150m-ternary",
    "state": "Hi, I was charged twice for my Pro plan this month (invoice #4471). Please refund the duplicate.",
    "questions": {
        "refund": {"type": "noul", "instructions": "The customer asks for money back."},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payments, invoices and refunds",
                "technical": "Bugs, errors and outages",
                "account": "Login, profile and settings",
            },
        },
    },
}
request = urllib.request.Request(
    "http://localhost:11435/api/decide",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request) as response:
    answers = json.load(response)["answers"]

print(answers["team"]["choice"], answers["refund"]["noul"])
```

## 4. TypeScript

`fetch` is enough. Node 22.18 or newer runs the file directly with `node decide.ts`:

```ts
const response = await fetch('http://localhost:11435/api/decide', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    model: 'coba:150m-ternary',
    state: 'Hi, I was charged twice for my Pro plan this month (invoice #4471). Please refund the duplicate.',
    questions: {
      refund: { type: 'noul', instructions: 'The customer asks for money back.' },
      team: {
        type: 'choice',
        instructions: 'Which team should handle this ticket?',
        criteria: {
          billing: 'Payments, invoices and refunds',
          technical: 'Bugs, errors and outages',
          account: 'Login, profile and settings',
        },
      },
    },
  }),
})
if (!response.ok) throw new Error(`Ollaya: ${response.status} ${await response.text()}`)
const { answers } = await response.json()
console.log(answers.team.choice, answers.refund.noul)
```

To keep the model in the browser, as the try page does, use [`@ollaya/web`](https://www.npmjs.com/package/@ollaya/web).

## 5. Docker

The image `ghcr.io/ollaya-dev/ollaya` serves the same API. Run it, then pull the model into its volume:

```shell
docker run -d --name ollaya -p 127.0.0.1:11435:11435 -v ollaya:/home/ollaya/.ollaya ghcr.io/ollaya-dev/ollaya
docker exec ollaya ollaya pull coba:150m-ternary
```

The examples above then work unchanged.
- `127.0.0.1:` keeps the port on this machine, as the native server does.
- The volume keeps the models when the container is replaced.
- For an NVIDIA GPU, use the `:cuda` image (see [Download](/download)).
- To pin a release, use its version as the tag, such as `ghcr.io/ollaya-dev/ollaya:0.16.0`.

### Docker Compose

```yaml
services:
  ollaya:
    image: ghcr.io/ollaya-dev/ollaya
    ports:
      - "127.0.0.1:11435:11435"
    environment:
      OLLAYA_API_KEY: ${OLLAYA_API_KEY:?set OLLAYA_API_KEY}
    volumes:
      - ollaya:/home/ollaya/.ollaya
    restart: unless-stopped

volumes:
  ollaya:
```

```shell
export OLLAYA_API_KEY=$(openssl rand -hex 16)
docker compose up -d
docker compose exec ollaya ollaya pull coba:150m-ternary
```

- Other services in the same file reach the server at `http://ollaya:11435`.
- Every request needs `Authorization: Bearer $OLLAYA_API_KEY`. Without the key it gets `401`. The `ollaya` command inside the container sends the key itself.
- Compose refuses to start without the key, because inside the container the server listens on every address. See [Security](/docs/api#security).
- The image has a health check, so `docker compose ps` shows when the server is up.

## Next

- [API](/docs/api): every field, presets and errors.
- [Guardrails](/docs/guardrails) and [Routing](/docs/routing): questions for agents.
- [Integrations](/docs/integrations): LiteLLM, LangChain, the Vercel AI SDK, DSPy and LlamaIndex.
