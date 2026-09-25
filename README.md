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
  These graphs read the original `model.safetensors` from the author's Hugging Face repository,
  pinned to a commit and verified by sha256. Ollaya never re-hosts weights.
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
  - **Hardware:** ONNX Runtime on CPU, and CUDA on NVIDIA GPUs.
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
| `decider`, `decider:0.8b` | Mapika's Qwen3.5 decoders, 2B and 0.8B. The most accurate: 0.591 on typed-decisions |
| `nli`, `nli:modernbert-large` | Moritz Laurer's zero-shot NLI classifiers (DeBERTa-v3-large, ModernBERT-large) |
| `gliclass` | Knowledgator's instruction-following zero-shot classifier (DeBERTa-v3-large) |

Browse them at [ollaya.dev/search](https://ollaya.dev/search). Laya tags ending in
`-fp32` or `-fp16` pin the precision. The derived files of every model are also published at
[huggingface.co/ollaya-dev](https://huggingface.co/ollaya-dev).

## Install

- **Linux** (x86_64 or arm64, glibc ≥ 2.38, e.g. Ubuntu 24.04+):
  `curl -fsSL https://ollaya.dev/install.sh | sh`. When an NVIDIA GPU is present
  (driver R580+), the installer adds the CUDA runtime.
- **macOS** (Apple silicon): the same command.
- **Windows** (x64, CPU): `irm https://ollaya.dev/install.ps1 | iex` in PowerShell. For an NVIDIA GPU,
  use WSL 2 with the Linux command.
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
| `crates/ollaya-runner` | Inference engines (ONNX Runtime) |
| `crates/ollaya-lang` | Script and language detection for routers |
| `convert/` | Build-time Python: ONNX export, parity checks, packaging |
| `site/` | The website and the static model registry host |

## Development

```sh
cargo test --workspace
cargo build --release -p ollaya --features cuda   # CUDA build (Linux x86_64)
```

`convert/` rebuilds models. It exports them, checks parity against the PyTorch reference,
generates golden fixtures, and packages the result into `registry/`. See the module docstrings.

## License

Apache-2.0. Each model keeps its own license: `laya` (Convai Innovations), `decider` (Mapika),
`gliclass` (Knowledgator) and `nli:modernbert-large` are Apache-2.0, and `nli:deberta-v3-large`
(Moritz Laurer) is MIT.
