---
title: CLI reference
nav: CLI
description: Every ollaya command — run, pull, serve, list, ps, show, rm, cp, create, push and stop.
order: 2
---

# CLI reference

Ollaya is a single binary. If you have used Ollama, the commands will feel familiar.

| Command | What it does |
|---|---|
| `ollaya serve` | Start the server on `127.0.0.1:11435` |
| `ollaya run MODEL [STATE]` | Answer questions about a state; pulls the model first if needed |
| `ollaya pull MODEL` | Download a model from the registry |
| `ollaya list` | List models on this machine |
| `ollaya ps` | List models loaded in memory |
| `ollaya show MODEL` | Show a model's details, questions and license |
| `ollaya stop MODEL` | Unload a running model |
| `ollaya rm MODEL` | Delete a model |
| `ollaya cp SOURCE DESTINATION` | Copy a model under a new name |
| `ollaya create NAME -f Modelfile` | Create a model from a [Modelfile](/docs/modelfile) |
| `ollaya push MODEL` | Upload a model to a registry |

## Model names

Models are referenced as `name:tag`. When the tag is left out, `latest` is used:

```shell
ollaya run laya                 # same as laya:latest
ollaya run laya:multilingual
ollaya pull laya:en-fp32
```

Tags without a precision suffix resolve to **fp16 on a GPU** and **fp32 on CPU**. Add `-fp16` or `-fp32` to choose explicitly.

## ollaya serve

Starts the server that the CLI and your applications talk to. It listens on `127.0.0.1:11435` and serves both the native API (`/api/*`) and the TypeSafe-compatible API (`/v1/*`). See the [API reference](/docs/api).

```shell
ollaya serve
```

## ollaya run

Runs a model against a state and prints the answers. The model is pulled automatically if it is not on this machine yet.

```shell
ollaya run laya --preset triage "I was charged twice for my subscription this month."
```

`--preset NAME` uses a built-in question set such as `triage`. Models created from a Modelfile with `QUESTIONS` already know what to ask, so the state is all you pass:

```shell
ollaya run triage "My invoice shows the wrong company name."
```

After a request, the model stays loaded for `keep_alive` (five minutes by default), so the next request skips the load.

## ollaya pull

Downloads a model and verifies every layer.

```shell
ollaya pull laya
```

## ollaya list and ollaya ps

`list` shows models on disk with their size and when they were modified. `ps` shows models loaded in memory and how long they will stay loaded.

## ollaya show

Prints a model's details: backbone, context length, precision, baked-in questions, calibration and license.

```shell
ollaya show laya:en
```

## ollaya stop

Unloads a running model right away instead of waiting for `keep_alive` to expire.

## ollaya rm and ollaya cp

```shell
ollaya cp laya:en my-guardrail
ollaya rm my-guardrail
```

## ollaya create

Builds a derived model from a [Modelfile](/docs/modelfile), for example to bake a question schema or a refit calibration into a model.

```shell
ollaya create triage -f Modelfile
```

## ollaya push

Uploads a model you created to a registry, so others can `pull` it.
