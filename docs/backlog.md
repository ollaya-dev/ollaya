# Backlog

Items queued for GitHub issues. Each item becomes one issue once `github.com/ollaya-dev/ollaya` exists.

---

## MCP server: expose local decision models as agent tools

**Why.** Agents need fast, calibrated, typed decisions for routing, guardrails, triage and escalation. An MCP server lets Claude Code, Claude Desktop, Cursor and other MCP clients call local decision models with no API keys, no per-token cost, and no data leaving the machine. A decision in ~10 ms is cheap enough to put in front of every agent step.

**Scope**
- An `ollaya mcp` subcommand: an MCP server over stdio, plus streamable HTTP behind `--http`. It talks to the local daemon and starts it if needed, like the CLI.
- Tools:
  - `decide`: `{model, state, questions}` → answers, same schema as `/v1/systemone`.
  - `list_models`, `show_model`, `pull_model` (reports progress).
- Resources: installed models and their question presets, so a client can discover what `triage`, `guard` and the others ask.
- Docs: a setup snippet per client, e.g. `claude mcp add ollaya -- ollaya mcp`.

**Acceptance**
- Works in Claude Code and one other MCP client.
- Tool schemas validated against the MCP spec version current at implementation time.
- `decide` output is identical to `/v1/systemone`.

---

## Agent Skill: teach agents when and how to use Ollaya

**Why.** Having an MCP tool is not enough; agents also need to know *when* a typed decision beats asking an LLM, and how to write good questions (criteria, levels, noul phrasing, confidence thresholds).

**Scope**
- An Agent Skill (`SKILL.md` + examples) named e.g. `ollaya-decisions`. It covers:
  - picking a model;
  - writing `choice` / `score` / `noul` questions;
  - using presets;
  - acting vs escalating on confidence;
  - calling through MCP, the CLI or HTTP.
- Ship it in the repo and in `ollaya` releases, installable with one command.
- Optional: a Claude Code plugin that bundles the MCP server config and the skill.

**Acceptance**
- An agent with only the skill installed and `ollaya` running completes a ticket-triage task end-to-end using typed decisions.
