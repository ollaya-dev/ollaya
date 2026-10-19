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

---

## Decoders as fast as Ollama's: a reduced-precision path for the Qwen3.5 families

**Why.** Ollama 0.40.6 serves the same Kev and Nimble weights 4 to 8 times faster than Ollaya on the same RTX 5090,
at the same accuracy (Bespoke Labs' public suite, one request at a time; `docs/measurements/2026-10-ollama-0.40.md`):
Nimble 42 ms against 328 ms, Kev 4B 34 against 187 ms, Kev 9B 40 against 227 ms. Ollama runs Q8_0 GGUFs on
llama.cpp, whose b11669 knows these models natively (kev, nimble, clef, laya and pplx-decider decision types, a
`/v1/systemone` endpoint, decision metadata in the GGUF). Ollaya computes in fp32 on the BF16 weights. Its
calibration is better (Nimble's pooled ECE 0.022 against 0.115, because Ollaya applies the author's temperature),
which a faster path must keep.

**Scope**
- Measure first: ONNX Runtime CUDA with BF16 or FP16 compute for `qwen35.py` graphs (TensorRT RTX EP included),
  against the fp32 goldens. Report the decision agreement and the probability error per family.
- If that is not enough, run the Qwen3.5 families on Ollaya's libllama runner (ADR 0003). Ollaya never re-hosts
  weights, so the GGUF has to be built on the user's machine from the author's pinned safetensors (Ollama
  quantizes at `ollama create` time, `create/create.go`), and the pointer heads (kev, strands) need custom code.
- A reduced-precision tag is a new tag with its own measured gate (decision agreement, max probability error),
  never a loosened tolerance on the fp32 tags.

**Acceptance**
- `kev:4b` and `nimble:9b` within 1.5x of Ollama's median latency on the same GPU, same decisions on at least
  99.5% of the parity questions, the authors' temperatures applied.

---

## Give memory back after an oversized request

**Why.** A session's CUDA arena keeps the size of its largest `session.run`. One 10,000-token Nimble v3 row on a
24 GB RTX 4090 pushes the working set past the GPU, the WSL driver spills into system memory, and every later
request in that process stays slow: five-question requests went from 2.4 s to 20.9 s at the median
(`docs/families/nimble.md`, v3). Kev, Clef and pplx-decider have the same exposure near their windows.

**Scope**
- After a run whose padded tokens exceed the usual budget, shrink the arena (ONNX Runtime's run option
  `memory.enable_memory_arena_shrinkage`), or run such batches in a separate allocation.
- Measure that the shrink costs nothing on ordinary requests.

**Acceptance**
- The Nimble v3 parity run, long row included, keeps the five-question p50 within 10% of a run without it.
