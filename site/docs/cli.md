---
title: CLI reference
nav: CLI
description: Every ollaya command and flag: run, pull, serve, list, ps, show, stop, rm, cp, create and mcp.
order: 2
---

# CLI reference

Ollaya is a single binary: the CLI, the server and the model runners. If you have used Ollama, the commands will feel familiar.

| Command | What it does |
|---|---|
| `ollaya serve` | Start the server on `127.0.0.1:11435` |
| `ollaya run MODEL [STATE]` | Answer questions about a state, or open a prompt; pulls and loads the model as needed |
| `ollaya pull MODEL` | Download a model from the registry |
| `ollaya list` (`ls`) | List models on this machine |
| `ollaya ps` | List models loaded in memory |
| `ollaya show MODEL` | Show a model's details, capabilities and license |
| `ollaya stop MODEL` | Unload a running model |
| `ollaya stop` | Stop the server that the CLI started |
| `ollaya mcp [--http [ADDR]]` | Serve the models to AI agents over MCP ([Agents](/docs/agents)) |
| `ollaya rm MODEL…` | Remove one or more models |
| `ollaya cp SOURCE DESTINATION` | Copy a model under a new name |
| `ollaya create NAME [-f Modelfile]` | Create a model from a [Modelfile](/docs/modelfile) |
| `ollaya -v` | Print the server's (and the client's) version |

Every command except `serve` talks to the server at `OLLAYA_HOST`. When nothing answers there and the address is local, the CLI (except `ollaya stop` without a model) starts `ollaya serve` in the background, logging to `~/.ollaya/logs/server.log`.

## Model names

Models are referenced as `name:tag`; without a tag, `latest` is used. Names are case-insensitive.

```shell
ollaya run laya                 # same as laya:latest, a router
ollaya run laya:multilingual
ollaya pull laya:en-fp32
```

A model such as `laya:en` carries an fp16 and an fp32 graph that share one weights file. The precision is picked when the model loads: fp16 on a CUDA GPU, fp32 on the CPU. The `-fp16` and `-fp32` tags pin one.

## ollaya run

```shell
ollaya run laya --preset triage "I was charged twice for my subscription this month and want a refund."
```

`run` connects to the server (starting it if needed), pulls the model if it is not on this machine, loads it and prints one row per question: the answer, a bar and its probability.

| Flag | Effect |
|---|---|
| `--preset NAME` | Use a built-in question set: `triage`, `email`, `guard`, `moderation`, `router` or `agent` |
| `--questions FILE` | Use the questions in a JSON file (question id → question). Overrides the model's own |
| `--format text\|json` | `text` (default) prints the table; `json` prints the full [`/api/decide`](/docs/api#decide) response |
| `--keepalive DURATION` | How long to keep the model loaded afterwards: `5m`, `1h`, `0` (unload now), `-1` (keep loaded) |
| `--verbose` | Also print every option's probability, the routing decision and the timings |
| `--state-json` | Parse the state as JSON. A state that looks like a JSON object or array is detected anyway |

Where the questions come from, first match wins: `--questions`, then `--preset`, then questions built into the model (qwen3guard's, or ones added with a Modelfile), then the `triage` preset. When `run` falls back to `triage` it says so on stderr.

**The state** is the rest of the command line. Without one, `run` reads piped stdin:

```shell
cat ticket.txt | ollaya run laya --preset triage
```

On a terminal without a state, `run` opens a prompt. Type a state and press Enter; wrap several lines in `"""`. Commands:

| Command | Effect |
|---|---|
| `/preset NAME` | Switch to a built-in question set |
| `/set questions FILE` | Use the questions in a JSON file |
| `/show` | Show the model and the current questions |
| `/clear` | Clear the screen |
| `/bye` | Exit (or Ctrl+D) |
| `/?`, `/help` | Help |

## ollaya serve

Starts the server that the CLI and your applications talk to. It serves the native API (`/api/*`) and the TypeSafe-compatible API (`/v1/*`); see the [API reference](/docs/api). The Linux installer runs it as the `ollaya` systemd service.

```shell
ollaya serve
```

You rarely need it: the other commands start the server in the background when it is not running. If a server already runs at `OLLAYA_HOST`, `ollaya serve` says so and exits; stop that one with `ollaya stop` first to run the server in the foreground.

It is configured with environment variables:

| Variable | Default | Effect |
|---|---|---|
| `OLLAYA_HOST` | `127.0.0.1:11435` | Address to bind; the CLI's target |
| `OLLAYA_MODELS` | `~/.ollaya/models` | Model store |
| `OLLAYA_KEEP_ALIVE` | `5m` | How long a model stays loaded after its last request |
| `OLLAYA_MAX_LOADED_MODELS` | `3` | Models kept loaded at once |
| `OLLAYA_MAX_QUEUE` | `512` | Decision requests in flight before `503 QUEUE_FULL` |
| `OLLAYA_LOAD_TIMEOUT` | `5m` | How long a model may take to load |
| `OLLAYA_DEVICE` | `auto` | `auto` (an NVIDIA GPU when the installer added the CUDA libraries, else the CPU), `cpu`, `cuda` or `cuda:<n>` |
| `OLLAYA_API_KEY` | unset | Require `Authorization: Bearer <key>`; the CLI sends it too |
| `OLLAYA_ORIGINS` | unset | Extra browser origins to allow, comma-separated |
| `OLLAYA_REGISTRY` | `{{SITE_HOST}}` | Default registry host in model names |

For the systemd service, change them with `sudo systemctl edit ollaya` and `Environment=` lines.

## ollaya pull

Downloads a model and verifies every layer against its sha256. Pulling a router also pulls every model it routes to. Only the layers this machine needs are downloaded, and interrupted downloads resume.

```shell
ollaya pull laya
```

## ollaya list and ollaya ps

`list` shows the models on this machine with their ID, size and when they were pulled. `ps` shows the loaded models, the device they run on (`cpu`, `cuda:0`), the precision and when they will be unloaded.

```text
NAME                ID             SIZE     MODIFIED
laya:latest         b87ca1631b11   11 KB    13 seconds ago
laya:multilingual   ba7a334675b4   684 MB   13 seconds ago
laya:en             bf30e4654e94   854 MB   30 seconds ago
```

## ollaya show

Prints a model's architecture, parameters, context length, precisions, languages, capabilities and license. For a router, it prints the routes.

| Flag | Prints only |
|---|---|
| `--questions` | The questions built into the model |
| `--license` | The license |
| `--modelfile` | A Modelfile that recreates the model |
| `--parameters` | The parameters, such as a pinned precision |

## ollaya stop

`ollaya stop MODEL` unloads a running model once its in-flight requests finish, instead of waiting for the keep-alive to run out.

`ollaya stop` without a model stops the server at `OLLAYA_HOST`, which unloads every model. It stops only a server that you started, with `ollaya serve` or by running another command. It never stops another user's server, such as the Linux systemd service; stop that with `sudo systemctl stop ollaya`.

## ollaya rm and ollaya cp

```shell
ollaya cp laya:en my-guardrail
ollaya rm my-guardrail
```

`rm` also deletes the blobs no other model uses. Removing a router keeps the models it routes to. `cp` overwrites an existing destination.

## ollaya create

Builds a model from a [Modelfile](/docs/modelfile), for example to bake in a question set, a refit calibration or a pinned precision. `-f` defaults to `./Modelfile`.

```shell
ollaya create triage -f Modelfile
```
