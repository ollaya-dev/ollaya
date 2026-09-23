# Ollaya

Runs open decision models locally, the way Ollama runs LLMs (see `README.md`).

- Layout: `crates/` is the Rust workspace (CLI, axum daemon, ONNX Runtime runner processes, registry client); `convert/` is Python build-time tooling (ONNX export, parity, goldens); `site/` is the static website (Hono JSX and Tailwind, Cloudflare static assets).
- Tests: `cargo test --workspace`. Runtime parity: `cargo run --release -p ollaya-runner --example parity -- <model-dir> <goldens.jsonl>`. Export parity and goldens: "Development" in `README.md`. Site: `cd site && npm run build`.
- Invariant: parity tests must pass. Never loosen a tolerance to make a change pass.
- Invariant: Python is build-time only. Nothing at runtime depends on `convert/`.
- Invariant: never re-host model weights. They come unmodified from the author's Hugging Face repo, so never commit, upload or bundle them.
- Project-specific overrides for the skills in `.claude/skills/`: see `.claude/skills/PROJECT_NOTES.md`.
