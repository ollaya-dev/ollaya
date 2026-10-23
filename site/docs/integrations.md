---
title: Integrations: LiteLLM and LangChain
nav: Integrations
description: Use Ollaya from LiteLLM and LangChain through their TypeSafe integrations, with a base URL and nothing else to install, including LangChain's model router and LiteLLM's tool-result guardrail.
order: 12
---

# Integrations

LiteLLM and LangChain ship TypeSafe integrations. Ollaya speaks TypeSafe's wire format, so every one of them runs on a local model once you point it at your server: no Ollaya package, no plugin and no API key.

| Library | What it adds | Point it at Ollaya with |
|---|---|---|
| [LiteLLM](#litellm) | `litellm.systemone` and `litellm.decisions`, the same on its proxy, and a guardrail that drops stale tool results | `api_base`, or `TYPESAFE_API_BASE` |
| [LangChain](#langchain) (Python and JS) | `TypeSafeClassifier`, a runnable, and agent middleware that picks the model or blocks risky tool calls | `base_url` / `baseUrl`, or `TYPESAFE_BASE_URL` |
| [The TypeSafe SDKs](/docs/typesafe-compatibility) | The official client | `TYPESAFE_BASE_URL` |

What they all need:

- **The server.** `ollaya serve`, or any `ollaya` command, which starts it.
- **A model.** Pull it first. `winnow:e4b` is the recommended model, and `laya` is fast on a CPU. See [which model to use](/docs/typesafe-compatibility).
- **An API key, any value.** The libraries refuse to send a request without one. Ollaya ignores it unless you set `OLLAYA_API_KEY`, and then it must match.
- **A model name.** Set it wherever you can. A library that sends TypeSafe's default, `jev-latest`, gets `404 MODEL_NOT_FOUND`, unless you give a local model that name (see [LangChain agent middleware](#agent-middleware)).

These examples were checked against Ollaya 0.16.0 with LiteLLM 1.106, `langchain-typesafe` 0.0.1a3 and `@langchain/typesafe` 0.0.2. The integrations are young, so check their changelogs when you upgrade.

## LiteLLM

The provider is `typesafe/`, followed by the Ollaya model name.

### In Python

`litellm.systemone` takes TypeSafe's request: a state, and questions keyed by id.

```python
import litellm

response = litellm.systemone(
    model="typesafe/winnow:e4b",
    api_base="http://localhost:11435",
    api_key="local",
    state="I was charged twice this month. Refund the second charge or I'm cancelling.",
    questions={
        "refund": {"type": "noul", "instructions": "The customer asks for money back."},
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"},
        },
    },
)
print(response.answers["refund"].noul, response.answers["department"].choice)
```

`litellm.decisions` takes LiteLLM's other format, `input` and a list of `predicate`, `choice` and `score` questions, and converts it for you:

```python
response = litellm.decisions(
    model="typesafe/winnow:e4b",
    api_base="http://localhost:11435",
    api_key="local",
    input="I was charged twice this month. Refund the second charge or I'm cancelling.",
    questions=[
        {"type": "predicate", "name": "refund", "instructions": "Does the customer ask for money back?"},
        {"type": "choice", "name": "department", "instructions": "Which team should handle this?",
         "choices": [{"value": "billing"}, {"value": "technical"}]},
    ],
)
print(response.answers[0].probability)
```

To set the server once instead of per call, use `TYPESAFE_API_BASE=http://localhost:11435` and `TYPESAFE_API_KEY=local`.

LiteLLM validates requests before they reach Ollaya. It needs `instructions` or `criteria` on every yes/no question, even though Ollaya would read the question id on its own.

### On the proxy

Add the model to `model_list`. Clients then call the proxy's `/v1/systemone` or `/v1/decisions` with the model name, `decide` here, and their LiteLLM key:

```yaml
model_list:
  - model_name: decide
    litellm_params:
      model: typesafe/winnow:e4b
      api_base: http://localhost:11435
      api_key: local
```

```shell
curl http://localhost:4000/v1/systemone \
  -H "Authorization: Bearer $LITELLM_KEY" -H "Content-Type: application/json" \
  -d '{"model": "decide", "state": "I was charged twice this month.", "questions": {"refund": {"type": "noul", "instructions": "The customer asks for money back."}}}'
```

### Drop stale tool results before each LLM call

LiteLLM's `typesafe` guardrail trims an agent's context. Before a chat request goes to the LLM, it asks one yes/no question per completed tool exchange: is this result still needed for the task? Results whose probability falls below `relevance_threshold` are replaced with a short note. With Ollaya the check runs on your machine, so the tool results never leave it:

```yaml
guardrails:
  - guardrail_name: compaction
    litellm_params:
      guardrail: typesafe
      mode: pre_call
      default_on: true
      api_base: http://localhost:11435
      api_key: local
      model: winnow:e4b
      optional_params:
        relevance_threshold: 0.2       # drop below this probability; LiteLLM's default
        min_chars_to_evaluate: 200     # skip short results; LiteLLM's default
```

A result the guardrail drops is gone for that request, so measure the threshold on your own agent's traces before you turn it on everywhere. If Ollaya can't be reached, the request goes through unchanged. Set `unreachable_fallback: fail_closed` to reject it instead.

## LangChain

### Classify

`TypeSafeClassifier` is a runnable. Its state can be a string, JSON, or LangChain messages, which it sends as role and content.

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_typesafe import Choice, Noul, TypeSafeClassifier

classifier = TypeSafeClassifier(model="winnow:e4b", base_url="http://localhost:11435", api_key="local")

response = classifier.invoke({
    "state": [
        HumanMessage("My card was charged twice."),
        AIMessage("Sorry about that. Which card?"),
        HumanMessage("The Visa. Refund it today or I cancel."),
    ],
    "questions": {
        "refund": Noul(instructions="Does the customer ask for a refund?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "Payments and refunds", "technical": "Bugs and outages"},
        ),
    },
})
print(response.nouls["refund"].noul, response.choices["department"].choice)
```

In JavaScript, the questions are fixed when the classifier is created, and the input is the state:

```ts
import { TypeSafeClassifier } from '@langchain/typesafe'

const classifier = new TypeSafeClassifier({
  model: 'winnow:e4b',
  baseUrl: 'http://localhost:11435',
  apiKey: 'local',
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: { billing: 'Payments and refunds', technical: 'Bugs and outages' },
    },
    urgent: { type: 'noul', instructions: 'Does this convey urgency?' },
  },
})

const result = await classifier.invoke({ message: 'Help! My payouts have been failing for 3 days.' })
console.log(result.choices.department.choice, result.nouls.urgent.noul)
```

### Agent middleware

Both packages have two experimental middlewares for `create_agent` / `createAgent`:

- **The model router** classifies the latest user message once per run and sends every model call in that run to the chosen model, a cheap one for simple requests and a strong one for hard ones.
- **Auto mode** checks the tool calls you list before they run, and blocks the ones whose probability of being risky or unauthorized reaches a threshold.

In JavaScript, pass the model and the server in `classifierOptions`:

```ts
import { createAgent } from 'langchain'
import { modelRouterMiddleware } from '@langchain/typesafe/middleware'

const router = modelRouterMiddleware({
  choices: {
    fast: { model: 'openai:gpt-5-mini', criteria: 'Greetings, lookups and other simple, well-scoped tasks.' },
    powerful: { model: 'anthropic:claude-opus-5-5', criteria: 'Multi-step reasoning, proofs and long code changes.' },
  },
  instructions: 'Choose the least costly model suited to the task.',
  classifierOptions: { model: 'winnow:e4b', baseUrl: 'http://localhost:11435', apiKey: 'local' },
})

const agent = createAgent({ model: 'openai:gpt-5-mini', middleware: [router] })
```

The Python middlewares, from `pip install "langchain-typesafe[experimental]"`, take no model or server. They read the server from the environment and always ask for `jev-latest`, so give a local model that name:

```shell
ollaya cp winnow:e4b jev-latest
export TYPESAFE_BASE_URL=http://localhost:11435 TYPESAFE_API_KEY=local
```

```python
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import ModelChoice, ModelRouterMiddleware

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(model="openai:gpt-5-mini", criteria="Greetings, lookups and other simple, well-scoped tasks."),
        "powerful": ModelChoice(model="anthropic:claude-opus-5-5", criteria="Multi-step reasoning, proofs and long code changes."),
    },
    instructions="Choose the least costly model suited to the task.",
)
agent = create_agent("openai:gpt-5-mini", middleware=[router])
```

The routing answer, with its probabilities, is kept in the agent state as `model_route` (`modelRoute` in JavaScript). For the same decisions from Claude Code, Cursor or any MCP client, see [Agents](/docs/agents).

## Anything else

Any tool that can send an HTTP request can use Ollaya: POST a state and questions, or a [preset](/docs/api#presets), to [`/api/decide`](/docs/api#decide). That includes workflow builders such as n8n and Dify through their HTTP Request nodes.

If the tool runs in a container, `localhost` is the container itself, not your machine. Bind Ollaya to an address the container can reach, such as `OLLAYA_HOST=0.0.0.0` with the container calling `host.docker.internal:11435`. Read [Security](/docs/api#security) first and set `OLLAYA_API_KEY`.
