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

## Decoders as fast as Ollama's: Nimble and Clef

**Why.** 0.14.0 closed most of the gap with Ollama 0.40.6 on the same RTX 5090 (ADR 0008): `Split` instead of
`SplitToSequence` and the `-fp16` tags. On the typed-decisions suite (median request,
`results/runs/2026-10-19-typed-decisions-rtx5090-cuda.json` against `2026-10-18-…-ollama.json`), `kev:4b-fp16` is
within 1.2x of Ollama's Q8_0 (152 against 129 ms) and `kev:9b-fp16` within 1.5x (211 against 138 ms), with the same
decisions as fp32 on at least 99.5 % of the parity questions. Two remain slow: `nimble:9b-fp16` 675 against 141 ms
(4.8x) and `clef:flash-fp16` 160 against 72 ms (2.2x). Their calibration stays better than Ollama's (Nimble's
pooled ECE 0.022 against 0.115), which a faster path must keep.

**Scope**
- Profile `nimble:9b-fp16` per request: its prompt is longer than Kev's, so prefill dominates; check what ONNX
  Runtime runs on the CPU and whether attention takes the fused path.
- If ONNX Runtime cannot close it, run the Qwen3.5 families on Ollaya's libllama runner (ADR 0003), with the GGUF
  built on the user's machine from the author's pinned safetensors (Ollaya never re-hosts weights).
- A faster path is a new tag with its own gate, never a loosened tolerance on the existing tags.

**Acceptance**
- `nimble:9b` and `clef:flash` within 1.5x of Ollama's median latency on the same GPU, same decisions on at least
  99.5 % of the parity questions, the authors' temperatures applied.

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

---

## decider's own image path: `decider:2b` and `decider:4b` reading images (layout `decider-vision-v2`)

**Why.** From decider-2b v12 and decider-4b v3 (2026-10-20) Mapika ships the text model with a `vision/` folder (the
Qwen3.5 vision tower, 0.6 GB, and the image-text config) and serves images through the same weights (decider-ai 1.10.0,
`decider/vision_reader.py`). Their image scores (0.847 for 2b, 0.891 for 4b on 11 the_cauldron sets) are well above the
separate `Mapika/decider-2b-vision` (v5 text weights) that `decider:2b-vision` runs today. COB-559 updated the text tags
and kept `decider:2b-vision` on its own repo: the new image path is a new layout, not a weight refresh.

**Scope**
- Layout `decider-vision-v2`, ported from `vision_reader.py` + `images.py`: the row is a `decider-slots-v1` text row (all
  255 labels, isolated score levels), every asset reference in the state becomes "the image" ("image k" with several), and
  one `<|vision_start|> <|image_pad|> x n <|vision_end|>` block per image goes in front of the whole row (slot shifted);
  images resized to at most 768x768 pixels; up to 4 images; M-RoPE positions from the image-text model's
  `get_rope_index` (as `decider-vision-v1`).
- Wire: TypeSafe's Jev `assets` map (`{"<ref>": {"type": "image", "format": "png", "data_base64": ...}}`) on
  `/v1/systemone` and `/api/decide`, references named in the state; keep `images` for `decider:2b-vision`.
- Export: `vision.onnx` from `vision/visual.safetensors` (keys `model.visual.*`) and a decoder graph with
  `position_ids` and `image_embeds` inputs reading the root `model.safetensors`; one manifest layer for the vision tower.
  Decide whether the image graphs become `decider:2b-vision` / `decider:4b-vision` tags (sharing the text weights blob
  with `decider:2b` / `decider:4b`) or the text tags themselves gain the image path.
- Goldens from decider-ai 1.10.0's server path (`decider.serve._prepare_images` + `VisionReader.score`), fp32; the same
  gate as the text tags; CPU and CUDA parity; `decider-2b-vision` retired once the new tag passes.

**Acceptance**
- Identical token ids, image blocks, positions and resized pixels on the goldens; same decisions; option logits within
  1e-3; `/v1/systemone` with `assets` answers as decider-ai 1.10.0 does.
