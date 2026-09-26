<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="site/public/static/logo-white.svg">
    <img src="site/public/static/logo.svg" alt="" width="88">
  </picture>
</p>

<h1 align="center">ollaya</h1>

<p align="center"><strong>Run open decision models locally, the way Ollama runs LLMs.</strong></p>

<p align="center">
  <a href="https://ollaya.dev">Website</a> ·
  <a href="https://ollaya.dev/search">Models</a> ·
  <a href="https://ollaya.dev/docs">Docs</a> ·
  <a href="https://github.com/ollaya-dev/ollaya/releases">Releases</a> ·
  <a href="https://huggingface.co/ollaya-dev">Hugging Face</a>
</p>

A decision model reads a *state* (a message, an email, a ticket, any JSON) plus typed questions
(`choice`, `score`, `noul`) and returns calibrated probabilities in a single forward pass, in
milliseconds. It never generates text. Ollaya pulls these models by name, serves them from a
local daemon, and speaks TypeSafe's `/v1/systemone` wire format, so existing Jev clients work by
changing one environment variable.

```sh
curl -fsSL https://ollaya.dev/install.sh | sh
ollaya run laya --preset triage "I was charged twice for my subscription this month and want a refund."
```

```
intent            refund       ████████████████ 1.00
is_urgent         no           ██████████████░░ 0.88
frustration       1.76 / 3     ██████░░░░░░░░░░ 0.36
refund_requested  yes          ██████████████░░ 0.90
churn_risk        no           ██████████░░░░░░ 0.61
```

## Features

- **One binary.** `ollaya serve` runs the daemon; `ollaya run`, `pull`, `list`, `ps`, `show`, `rm`,
  `cp`, `stop` and `create` work the way they do in Ollama. If the daemon isn't running, the CLI
  starts it.
- **TypeSafe-compatible.** `POST /v1/systemone`, `/v1/decisions` and `GET /v1/models` are
  wire-identical to TypeSafe. The official SDK works unchanged when you set
  `TYPESAFE_BASE_URL=http://localhost:11435`.
- **Native API.** `/api/decide` adds routing information and timings. `/api/pull` streams
  NDJSON progress, and there are `/api/tags`, `/api/show`, `/api/ps` and more. See
  [docs/api.md](docs/api.md).
- **Weights come from their authors.** Ollaya publishes only small ONNX graphs, about 3 MB each.
  These graphs read the original weight files (usually `model.safetensors`) from the author's
  Hugging Face repository, pinned to a commit and verified by sha256. Models whose authors publish
  GGUF files (`winnow`) run that file itself on llama.cpp. Ollaya never re-hosts weights.
- **For agents.** `ollaya mcp` serves the models to Claude Code, Claude Desktop, Cursor and other
  MCP clients (`claude mcp add ollaya -- ollaya mcp`), and the
  [`ollaya-decisions` skill](skills/ollaya-decisions/SKILL.md) teaches agents when and how to use
  them (`npx skills add ollaya-dev/ollaya --skill ollaya-decisions`).
- **Routers.** `laya` detects the script and language of each request, then answers with
  `laya:en` or `laya:multilingual`.
- **Modelfiles.** You can bake a question set into your own model:
  ```
  FROM laya
  QUESTIONS ./triage.json
  PARAMETER precision fp32
  ```
  Then run `ollaya create triage -f Modelfile` and `ollaya run triage "…"`.
- **Fast and exact.**
  - **Hardware:** ONNX Runtime on CPU, and CUDA on NVIDIA GPUs. GGUF models run on llama.cpp:
    CPU, CUDA, and Metal on Apple silicon.
  - **Precision:** fp16 on GPU and fp32 on CPU, chosen when the model loads.
  - **Accuracy:** fp32 exports give the same decision as the PyTorch reference on 100% of 2,383
    questions per checkpoint.

## Models

| Model | What it is |
|---|---|
| `laya` | Router: picks `laya:en` or `laya:multilingual` by language |
| `laya:en` | English decision model (ModernBERT-large, 421M). The fastest: 8–10 ms for five questions on an RTX 4090 |
| `laya:multilingual` | 100+ languages (mmBERT-base, 322M) |
| `laya:typed-decisions` | Fine-tuned on the typed-decisions workflows |
| `decider`, `decider:0.8b` | Mapika's Qwen3.5 decoders, 2B (the default) and 0.8B: 0.591 on typed-decisions for 2B |
| `kev`, `kev:4b`, `kev:9b` | Jared Palmer's Kev: a LoRA and a pointer head on Qwen3.5 (0.8B by default, 4B, 9B), calibrated. `kev:9b` scores 0.722 on typed-decisions, the most of the models not trained on it |
| `decision` | Decision 1.0 Eos by the vLLM Semantic Router contributors: a fine-tuned Qwen3.5-0.8B with an endpoint head, 16k-token rows |
| `qwen3guard` | Qwen3Guard-Gen-0.6B safety guard with built-in questions: safe, controversial or unsafe, and the category |
| `nli`, `nli:modernbert-large` | Moritz Laurer's zero-shot NLI classifiers (DeBERTa-v3-large, ModernBERT-large) |
| `gliclass` | Knowledgator's instruction-following zero-shot classifier (DeBERTa-v3-large) |
| `von` | Victor Hugo Panisa's Von 1.1 (ModernBERT-large): every option scored at its own marker, 8k-token context |
| `winnow`, `winnow:e4b` | EldanRing's Winnow-12B and Winnow-E4B, Gemma 4 fine-tunes run from the author's Q8_0 GGUF on llama.cpp |

Browse them at [ollaya.dev/search](https://ollaya.dev/search). Laya tags ending in
`-fp32` or `-fp16` pin the precision. The derived files of every model are also published at
[huggingface.co/ollaya-dev](https://huggingface.co/ollaya-dev).

## Install

- **Linux** (x86_64 or arm64, glibc ≥ 2.38, e.g. Ubuntu 24.04+):
  `curl -fsSL https://ollaya.dev/install.sh | sh`. When an NVIDIA GPU is present
  (driver R580+), the installer adds the CUDA runtime.
- **macOS** (Apple silicon): the same command.
- **Windows** (x64): `irm https://ollaya.dev/install.ps1 | iex` in PowerShell. When an NVIDIA GPU is
  present (driver R580+), the installer adds the CUDA runtime, as on Linux.
- **Desktop app** for macOS, Windows and Linux: start and stop the server, download models and run
  them in one window. On macOS it lives in the menu bar. Get it from
  [ollaya.dev/download](https://ollaya.dev/download).
- **Docker:** `docker run -d --gpus=all -p 11435:11435 ghcr.io/ollaya-dev/ollaya:cuda`, or
  `ghcr.io/ollaya-dev/ollaya` for CPU only.

Configuration is through environment variables: `OLLAYA_HOST`, `OLLAYA_MODELS`,
`OLLAYA_KEEP_ALIVE`, `OLLAYA_DEVICE`, `OLLAYA_API_KEY` and others, listed in
[docs/api.md §15](docs/api.md).

## Repository

| Path | What |
|---|---|
| `crates/ollaya` | The binary: CLI, daemon, runner |
| `crates/ollaya-server` | HTTP API, scheduler (one runner process per model), model resolution |
| `crates/ollaya-api` | API types and client; the contract is [docs/api.md](docs/api.md) |
| `crates/ollaya-registry` | Model names, manifests, blob store, resumable pulls |
| `crates/ollaya-decision` | Question schema, sequence layouts, calibration, answers |
| `crates/ollaya-runner` | Inference engines (ONNX Runtime, and llama.cpp for GGUF models) |
| `crates/ollaya-lang` | Script and language detection for routers |
| `convert/` | Build-time Python: ONNX export, parity checks, packaging |
| `site/` | The website and the static model registry host |

## Development

```sh
cargo test --workspace
cargo build --release -p ollaya --features cuda   # CUDA build (x86-64 Linux and Windows)
```

`convert/` rebuilds models. It exports them, checks parity against the PyTorch reference,
generates golden fixtures, and packages the result into `registry/`. See the module docstrings.

## License

Apache-2.0. Each model keeps its own license: `laya` (Convai Innovations), `decider` (Mapika),
`kev` (Jared Palmer, on Qwen3.5 by the Qwen team), `decision` (the vLLM Semantic Router
contributors, on Qwen3.5), `qwen3guard` (Qwen team), `gliclass` (Knowledgator), `von` (Victor Hugo
Panisa), `winnow` (EldanRing, on Gemma 4 by Google DeepMind) and `nli:modernbert-large` are
Apache-2.0, and `nli:deberta-v3-large` (Moritz Laurer) is MIT. llama.cpp, which Ollaya ships for
GGUF models, is MIT.
