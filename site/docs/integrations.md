---
title: Integrations: LiteLLM, LangChain, AI SDK, DSPy, LlamaIndex, n8n and Dify
nav: Integrations
description: Use Ollaya from LiteLLM, LangChain, the Vercel AI SDK, DSPy and LlamaIndex through their TypeSafe integrations, with a base URL and nothing else to install, and from n8n and Dify workflows over HTTP.
order: 12
---

# Integrations

LiteLLM, LangChain, the Vercel AI SDK and DSPy ship TypeSafe integrations, and LlamaIndex has community packages. Ollaya speaks TypeSafe's wire format, so every one of them runs on a local model once you point it at your server: no Ollaya package, no plugin and no API key.

| Library | What it adds | Point it at Ollaya with |
|---|---|---|
| [LiteLLM](#litellm) | `litellm.systemone` and `litellm.decisions`, the same on its proxy, and a guardrail that drops stale tool results | `api_base`, or `TYPESAFE_API_BASE` |
| [LangChain](#langchain) (Python and JS) | `TypeSafeClassifier`, a runnable, and agent middleware that picks the model or blocks risky tool calls | `base_url` / `baseUrl`, or `TYPESAFE_BASE_URL` |
| [Vercel AI SDK](#vercel-ai-sdk) | Decision models for `experimental_decide` | `baseURL` ending in `/v1`, or `TYPESAFE_AI_BASE_URL` |
| [DSPy](#dspy) | `Choice`, `Noul` and `Score` output fields in signatures | `base_url`, or `TYPESAFE_BASE_URL` |
| [LlamaIndex](#llamaindex) (community) | A reranker and a query-engine selector | `TYPESAFE_BASE_URL` |
| [The TypeSafe SDKs](/docs/typesafe-compatibility) | The official client | `TYPESAFE_BASE_URL` |
| [n8n](#n8n) | Decisions in a workflow, through the HTTP Request node | The node's URL |
| [Dify](#dify) | A custom tool for agents, chatflows and workflows | The tool's OpenAPI schema |

What they all need:

- **The server.** `ollaya serve`, or any `ollaya` command, which starts it.
- **A model.** Pull it first. `winnow:e4b` is the recommended model, and `laya` is fast on a CPU. See [which model to use](/docs/typesafe-compatibility).
- **An API key, any value.** The libraries refuse to send a request without one. Ollaya ignores it unless you set `OLLAYA_API_KEY`, and then it must match.
- **A model name.** Set it wherever you can. A library that sends TypeSafe's default, `jev-latest`, gets `404 MODEL_NOT_FOUND`, unless you give a local model that name (see [LangChain agent middleware](#agent-middleware)).

- **A timeout that allows for loading.** The first request to a model waits while it loads. Some integrations time out sooner than that by default; the sections below say which.

n8n and Dify have no TypeSafe integration. They call [`/api/decide`](/docs/api#decide) over HTTP, with nothing to install.

These examples were checked against Ollaya 0.16.0 with LiteLLM 1.106, `langchain-typesafe` 0.0.1a3, `@langchain/typesafe` 0.0.2, AI SDK 7.0 with `@ai-sdk/typesafe-ai` 3.0, DSPy 3.4.1, `llama-index-postprocessor-jev` and `llama-index-selectors-jev` 0.1.1, n8n 2.44 and Dify 1.18.0. n8n's workflows ran from its command line and Dify's tool through its own code and SSRF proxy, not through their web editors. The integrations are young, so check their changelogs when you upgrade.

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

## Vercel AI SDK

The AI SDK's TypeSafe provider, `@ai-sdk/typesafe-ai`, adds decision models to `experimental_decide`. Its base URL includes `/v1`:

```ts
import { createTypeSafeAi } from '@ai-sdk/typesafe-ai'
import { experimental_decide } from 'ai'

const ollaya = createTypeSafeAi({ baseURL: 'http://localhost:11435/v1', apiKey: 'local' })

const { answers } = await experimental_decide({
  model: ollaya.decisionModel('winnow:e4b'),
  state: "I was charged twice this month. Refund the second charge or I'm cancelling.",
  questions: {
    department: {
      type: 'choice',
      instructions: 'Which team should handle this?',
      criteria: { billing: 'Payments and refunds', technical: 'Bugs and outages' },
    },
    refund: { type: 'boolean', instructions: 'Does the customer ask for money back?' },
  },
})
console.log(answers.department.choice, answers.refund.probability)
```

To use the provider's default `typeSafeAi` export instead, set `TYPESAFE_AI_BASE_URL=http://localhost:11435/v1` and `TYPESAFE_AI_API_KEY=local`. Yes/no questions are `boolean` here (TypeSafe's `noul`), and each answer's confidence is in `providerMetadata.typesafe.confidence`.

## DSPy

DSPy 3.4 has an experimental TypeSafe client and three decision types for a signature's output fields: `Choice`, `Noul` and `Score`. Install it with `pip install "dspy[typesafe]"`.

```python
import dspy
from dspy.experimental import Choice, Noul, Score, TypeSafe

dspy.configure(lm=TypeSafe("winnow:e4b", base_url="http://localhost:11435", api_key="local", timeout=60))

class Triage(dspy.Signature):
    """Triage a customer message."""

    message: str = dspy.InputField()
    department: Choice[("billing", "Payments and refunds"), ("technical", "Bugs and outages")] = dspy.OutputField(
        desc="Which team should handle this?"
    )
    refund: Noul = dspy.OutputField(desc="Does the customer ask for money back?")
    urgency: Score["low", "medium", "high"] = dspy.OutputField(desc="How urgent is it?")

result = dspy.Predict(Triage)(message="I was charged twice this month. Refund the second charge or I'm cancelling.")
print(result.department.value, result.refund.probability, result.urgency.value)
```

The client also reads `TYPESAFE_BASE_URL`, `TYPESAFE_API_KEY` and `TYPESAFE_DEFAULT_MODEL`. Its default timeout is 10 seconds, so set `timeout` for a model that loads slowly. A `Noul` field's `confidence` is the distance from the threshold, not a probability: read `probability` for that. A `Score` field's `value` is the mean level, 0 to 2 here.

## LlamaIndex

LlamaIndex has no TypeSafe integration of its own. The community project [llama-index-jev](https://github.com/WiktorB2004/llama-index-jev) adds a reranker and a selector for router query engines. Both use the TypeSafe SDK, so they reach Ollaya through its environment variables:

```shell
pip install llama-index-postprocessor-jev llama-index-selectors-jev
export TYPESAFE_BASE_URL=http://localhost:11435 TYPESAFE_API_KEY=local
```

```python
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.postprocessor.jev import JevRerank
from llama_index.selectors.jev import JevSingleSelector

# Keep the five retrieved passages that best answer the query.
reranker = JevRerank(model="winnow:e4b", top_n=5, timeout_s=30)
query_engine = index.as_query_engine(node_postprocessors=[reranker])

# Send each query to the right engine.
router = RouterQueryEngine(
    selector=JevSingleSelector(model="winnow:e4b", timeout_s=30),
    query_engine_tools=[docs_tool, billing_tool],
)
```

- **Set `timeout_s`.** Its default, 2.5 seconds, is shorter than a model's first load. On a timeout the reranker keeps the retrieval order without saying so, and the selector raises.
- **One decision per passage.** The reranker asks about each passage separately, so ten retrieved passages are ten requests, sent eight at a time.

The project is independent of LlamaIndex and of TypeSafe.

## n8n

n8n calls Ollaya with its **HTTP Request** node, so there is no node to install. Set the node up like this:

- **Method:** `POST`.
- **URL:** `http://localhost:11435/api/decide`, or an address n8n can reach (see [From a container](#from-a-container)).
- **Send Body:** on.
- **Body Content Type:** `JSON`.
- **Specify Body:** `Using JSON`.
- **JSON:** switched to an expression, with this body:

```text
{
  "model": "coba",
  "state": {{ JSON.stringify($json.text) }},
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
    }
  }
}
```

`{{ JSON.stringify($json.text) }}` puts the incoming item's `text` field in as a JSON string, so quotes and line breaks in it can't break the body.

The answers are in the node's output. Later nodes, such as an If or a Switch, read them with expressions:

- `{{ $json.answers.team.choice }}`: the most likely team.
- `{{ $json.answers.refund.noul > 0.5 }}`: `true` when the refund question is more likely true than not.

With `OLLAYA_API_KEY` set, give the node a credential:

- **Authentication:** `Generic Credential Type`.
- **Generic Auth Type:** `Header Auth`.
- **Header Auth credential:** the name `Authorization` and the value `Bearer <your key>`.

## Dify

Dify calls Ollaya through a custom tool built from an OpenAPI schema. Agents, chatflows and workflows can all use it. Create one from this schema, with a server URL Dify can reach:

```yaml
openapi: 3.1.0
info:
  title: Ollaya
  description: Typed decisions from a local decision model.
  version: 1.0.0
servers:
  - url: http://host.docker.internal:11435
paths:
  /api/decide:
    post:
      operationId: decide
      summary: Answer typed questions about a text with a local decision model. Returns a probability for every answer.
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [state]
              properties:
                state:
                  type: string
                  description: The text to decide about, such as a support ticket or a user message.
                model:
                  type: string
                  description: The Ollaya model to use.
                  default: coba
                questions:
                  type: object
                  description: Leave empty to ask the default questions.
                  default:
                    refund:
                      type: noul
                      instructions: The customer asks for money back.
                    team:
                      type: choice
                      instructions: Which team should handle this ticket?
                      criteria:
                        billing: Payments, invoices and refunds
                        technical: Bugs, errors and outages
                        account: Login, profile and settings
```

- **Parameters.** Each field of the request body becomes a parameter of the tool, and the agent writes `state`.
- **Questions.** Left empty, `questions` gets its default from the schema, so put your own questions there. An agent can also pass its own as JSON text, and Dify parses it.
- **Response.** The tool returns Ollaya's response as JSON.
- **Private addresses.** Dify sends tool requests through its SSRF proxy, which refuses private, loopback and link-local addresses by default. `host.docker.internal` resolves to one of them. Add the range Ollaya is on to `SSRF_PROXY_ALLOW_PRIVATE_IPS` in Dify's `docker/.env`, such as `SSRF_PROXY_ALLOW_PRIVATE_IPS=172.21.0.0/16`, and restart Dify with `docker compose up -d`. Until then the tool fails with "blocked by SSRF protection", and the error names the address it blocked.
- **API key.** With `OLLAYA_API_KEY` set, give the tool API key authorization, sent as a Bearer token in the `Authorization` header.

## From a container

If the tool runs in a container, `localhost` is the container itself, not your machine.

- **Ollaya on your machine:** bind Ollaya to an address the container can reach, such as `OLLAYA_HOST=0.0.0.0`, and call `http://host.docker.internal:11435` from the container. Read [Security](/docs/api#security) first and set `OLLAYA_API_KEY`.
- **Ollaya in a container on the same Docker network,** such as the same Compose file: call it by its name, such as `http://ollaya:11435`.

## Anything else

Any tool that can send an HTTP request can use Ollaya: POST a state and questions, or a [preset](/docs/api#presets), to [`/api/decide`](/docs/api#decide).
