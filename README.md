# ollaya

**Run open decision models locally, the way Ollama runs LLMs.**

A decision model reads a *state* (a message, an email, a ticket, any JSON) plus typed questions
(`choice`, `score`, `noul`) and returns calibrated probabilities in a single forward pass, in
milliseconds. It never generates text. Ollaya pulls these models by name, serves them from a
local daemon, and speaks TypeSafe's `/v1/systemone` wire format, so existing Jev clients work by
changing one environment variable.

```sh
curl -fsSL https://ollaya.cobanov.dev/install.sh | sh
ollaya run laya --preset triage "I was charged twice this month and want a refund."
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
| `laya:en` | English decision model (ModernBERT-large, 421M) |
| `laya:multilingual` | 100+ languages (mmBERT-base, 322M) |
| `laya:typed-decisions` | Fine-tuned on the typed-decisions workflows |

Browse them at [ollaya.cobanov.dev/search](https://ollaya.cobanov.dev/search). Tags ending in
`-fp32` or `-fp16` pin the precision.

## Install

- **Linux** (x86_64 or arm64, glibc ≥ 2.38, e.g. Ubuntu 24.04+):
  `curl -fsSL https://ollaya.cobanov.dev/install.sh | sh`. When an NVIDIA GPU is present
  (driver R580+), the installer adds the CUDA runtime.
- **macOS** (Apple silicon): the same command.
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

Apache-2.0. Each model keeps its own license: the Laya family is Apache-2.0, by Convai
Innovations.
