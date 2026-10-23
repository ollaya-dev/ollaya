# Ollaya

Runs open decision models locally, the way Ollama runs LLMs (see `README.md`).

- Layout: `crates/` is the Rust workspace (CLI, axum daemon, ONNX Runtime runner processes, registry client); `convert/` is Python build-time tooling (ONNX export, parity, goldens); `site/` is the static website (Hono JSX and Tailwind, Cloudflare static assets).
- Tests: `cargo test --workspace`. Runtime parity: `cargo run --release -p ollaya-runner --example parity -- <model-dir> <goldens.jsonl>`. Export parity and goldens: "Development" in `README.md`. Site: `cd site && npm run build`.
- Before committing Rust changes: `cargo fmt --all`, then `cargo clippy --workspace --all-targets --locked -- -D warnings`. CI runs `cargo fmt --all --check` first and stops there, so unformatted code fails main before clippy and the tests run.
- Invariant: parity tests must pass. Never loosen a tolerance to make a change pass.
- Invariant: Python is build-time only. Nothing at runtime depends on `convert/`.
- Invariant: never re-host model weights. They come unmodified from the author's Hugging Face repo, so never commit, upload or bundle them.
- Project-specific overrides for the skills in `.claude/skills/`: see `.claude/skills/PROJECT_NOTES.md`.

## Test tiers

Every tier keeps the same tolerances; a tier chooses only the cases and the devices (COB-279).

- **K1, every change (2–5 min, local):** `cargo test --workspace`, and the parity example of each family the change
  touches with `--smoke`, on one GPU or, for a small model, the local CPU. `--smoke` checks only the cases listed in
  `crates/ollaya-runner/examples/smoke/cases.txt`, about 40 questions per family: every question type, refused
  requests, a long state and many options. The list is fixed; change it in review, never for one run.
- **K2, every release (15–20 min, hope and choso in parallel):** the full goldens on every GPU device we ship (Linux
  CUDA on hope's RTX 5090, Windows CUDA and Vulkan on choso's RTX 4090). On the CPU, the full set for models of 4B and
  under and `--smoke` for 9B and over. CPU parity counts only against goldens made on the same machine and build. K2
  never checks less than the release gate did before it.
- **K3, major releases, a llama.cpp or ONNX Runtime update, a new device or a new model:** full CPU parity of the
  large models on hope's Threadripper, typed-decisions, Bespoke's open benchmark and the Ollama comparison. The site,
  the README, the family docs and the reports are updated from these results (`docs/model-lifecycle.md`, stage 4).

## Working efficiently on long tasks

- Adding a model, changing a graph or the runtime, measuring or releasing: follow `docs/model-lifecycle.md`. Before measuring anything, run `python -m ollaya_convert.bench_plan` and measure what it lists, nothing more (ADR 0009). Numbers on the site come from `results/` through `{{result …}}` references, never typed in.

- A release (`release.yml`: Windows build, CUDA Docker images, desktop app) takes about 20 minutes (0.13.0: `release.yml` 16, `desktop.yml` 19, in parallel), and PR CI takes about 5–10. Don't idle-wait on them: start the next piece of work in parallel, or end the turn with a status update.
- Batch releases: ship finished work together in one version bump, not one release per feature. Each release costs about 20 minutes of pipeline time.
- Report progress every 20–30 minutes on long work, including under `/goal`: what is done, what is running, what is next. The user sees only final messages.
- Slow local steps: CPU parity on 2B+ decoders (4+ minutes), downloading multi-GB weights, sha256 checks in `package.py`. Run them in the background, and do CUDA parity first for a quick signal.
- `release.yml` runs `desktop.yml` too. The desktop app's Windows build takes its engine from the windows-amd64
  archive, and the cuda12 image from the linux-amd64 archives, as soon as the build job has packaged them
  (`scripts/wait-artifact.sh`); neither compiles it again.
