---
title: Agents: MCP and skill
nav: Agents (MCP)
description: Give AI agents local, typed decisions: the Ollaya MCP server for Claude Code, Claude Desktop, Cursor and other clients, and an agent skill that teaches when and how to use it.
order: 6
---

# Agents: MCP and skill

A decision takes about 10 ms and costs nothing, so an agent can make one before every routing,
triage, moderation or escalation step. Ollaya gives agents two things:

- **An MCP server**, `ollaya mcp`, which exposes the local models as tools.
- **An agent skill**, `ollaya-decisions`, which teaches an agent when a typed decision beats
  reasoning in text, how to write good questions, and how to act on the probabilities.

## MCP server

`ollaya mcp` speaks the Model Context Protocol over stdio. It talks to the local server like every
other command, and starts it when it isn't running.

### Claude Code

```shell
claude mcp add ollaya -- ollaya mcp
```

### Claude Desktop

Add it to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "ollaya": { "command": "ollaya", "args": ["mcp"] }
  }
}
```

If Claude Desktop can't find `ollaya`, use its full path, for example
`/usr/local/bin/ollaya` or `~/.local/bin/ollaya`.

### Cursor, VS Code and other clients

Use the same command, `ollaya mcp`, in the client's MCP settings: `.cursor/mcp.json` for Cursor,
`.vscode/mcp.json` for VS Code (with `"type": "stdio"`).

### Over HTTP

For clients that connect over the network, serve streamable HTTP instead:

```shell
ollaya mcp --http                  # http://127.0.0.1:11436/mcp
ollaya mcp --http 127.0.0.1:9000   # another address
```

It only accepts loopback hosts by default.

### Tools

| Tool | What it does |
|---|---|
| `decide` | Answers typed questions about a state. Arguments: `state` (a string or any JSON), `questions` or `preset`, and `model` (default `laya`). The result is exactly what [`POST /v1/systemone`](/docs/api) returns. |
| `list_models` | The models on this machine, like `ollaya list`. |
| `show_model` | A model's details, capabilities, license and built-in questions. |
| `pull_model` | Downloads a model, with progress notifications. `decide` also pulls on first use. |

### Resources

| URI | Contents |
|---|---|
| `ollaya://models` | The installed models |
| `ollaya://presets/<name>` | The questions of a built-in preset: `triage`, `email`, `guard`, `moderation`, `router`, `agent` |

## Agent skill

The skill is a `SKILL.md` that agents supporting Agent Skills, such as Claude Code, load when a
task calls for it. It covers picking a model, writing `choice`, `score` and `noul` questions,
the presets, confidence thresholds for acting or escalating, and calling Ollaya through MCP, the
CLI or HTTP.

Install it into a project (or add `-g` for every project):

```shell
npx skills add ollaya-dev/ollaya --skill ollaya-decisions
```

It also ships with every release, in `share/ollaya/skills/ollaya-decisions/` next to the binary
(`/usr/local/share/ollaya/skills/` for a system install), so you can copy it into
`.claude/skills/` without the network.

The skill works on its own too: an agent with only the skill and `ollaya` installed calls the CLI.
