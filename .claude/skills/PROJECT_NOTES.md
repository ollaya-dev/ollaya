# Project notes for the installed skills

The skills in this directory are generic, third-party guidance (see [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md)). Where a skill conflicts with a deliberate Ollaya decision below, this file wins. Record new overrides here; don't edit the skills.

## Wire formats (api-and-interface-design)

- The native `/api/*` API mirrors Ollama and `/v1/*` mirrors TypeSafe, on purpose (Hyrum's Law). Existing clients and SDKs must work by changing only the base URL. Keep:
  - snake_case fields (`keep_alive`);
  - verb paths (`/api/pull`, `/api/show`, `DELETE /api/delete`);
  - `{ "error": "<message>" }` error bodies;
  - NDJSON streaming;
  - an unpaginated `/api/tags`.

  Don't move these toward the skill's REST nouns, camelCase, `{error: {code, message}}` or pagination defaults.
- `/v1/*` requests and responses match TypeSafe's exactly. Ollaya-only fields go on `/api/*`.
- Keys in criteria and answers keep the caller's order (`serde_json` `preserve_order`).
- `/v2/` is reserved for the static model registry.

## Parity, tests, benchmarks (test-driven-development, performance-optimization, debugging-and-error-recovery)

- Parity is the gate. It has two parts:
  - the ONNX export against the PyTorch reference, in `convert/`;
  - the Rust runtime against the goldens, with `cargo run --release -p ollaya-runner --example parity`.

  If a change breaks parity, fix the change. Don't loosen a tolerance, drop a golden case or regenerate goldens unless the user explicitly asks.
- Rust tests: `cargo test --workspace`. The skills' npm and Jest commands are illustrations only.
- Parity and benchmark runs need local weights, which are not in git. CUDA runs also need a GPU. If a run isn't possible, say so; never report it as passing.
- Benchmark with `cargo run --release -p ollaya-runner --example bench`, on the same hardware, precision and inputs as the baseline. Revert any speedup that changes a decision or breaks parity. The database, cache and Core Web Vitals material applies to `site/` only.
- Don't silence a check to get to green. That means no new `#[allow(...)]`, `#[ignore]`, `# noqa` or `@ts-ignore`, no TODO stubs and no skipped tests.

## Python is build-time only

- `convert/` (uv, Python 3.12) exports and verifies models. The CLI, daemon, runner, install script and Docker image must never need Python at runtime.

## Model weights

- Never re-host model weights. Manifests point at the author's Hugging Face repo, and `ollaya pull` fetches the weights from there unmodified.
- Never commit, upload or bundle `*.onnx`, `*.onnx.data` or `*.safetensors` files. That covers git, release assets, Docker images and `site/dist`.

## Website (frontend-ui-engineering, performance-optimization)

- `site/` is fully static: Hono JSX pre-rendered at build time, Tailwind v4, and one dependency-free script, `public/static/app.js`.
  - There is no server code, no Worker script and no client framework. Don't add any of them.
  - Pages must work without JavaScript.
  - Ignore the skill's advice about React state, hooks, React Query and optimistic updates.
- Verify with `cd site && npm run build`, which runs the typecheck and the dist checks. Don't hard-code host names; use `SITE_ORIGIN` or `{{SITE_ORIGIN}}`. Never write under `dist/v2/`.
- Everything in `site/docs/` and `site/content/` is published on the website. Keep ADRs, specs and plans out of those folders. By default they go in `docs/decisions/` and `tasks/` at the repo root.

## Observability (observability-and-instrumentation)

- Ollaya is a local daemon with runner processes, not a hosted service.
  - Use structured `tracing` events.
  - Carry a request ID and the entry point (CLI or HTTP) from the daemon to the runner.
  - Don't add metrics backends, OpenTelemetry exporters, alerting or any outbound telemetry unless the user asks.
  - Treat request `state` as user data and don't log it.

## Git, CI, releases (git-workflow-and-versioning, incremental-implementation, ci-cd-and-automation, shipping-and-launch)

- Commit only when the user asks. Where a skill says "commit", stop at a verified increment and propose a commit message instead.
- These actions need an explicit request in the current conversation:
  - pushing commits or tags;
  - opening or merging PRs;
  - publishing a release;
  - `npm run deploy` or `wrangler deploy`.
- Never discard uncommitted work (`git reset --hard`, `git checkout -- .`, `git clean`, `git stash drop`) without the user's explicit consent.
- A release is three things: versioned binaries built by GitHub Actions, the install script, and a Docker image. The website deploys separately with wrangler.
- There is no staging environment, feature-flag service or percentage rollout, and none should be added. A rollback means going back to the previous release.
- CI uses each part's own tooling: `cargo` at the root, `npm run build` in `site/`, and `uv run` in `convert/`.

## Tooling and dependencies

- Only the skills in this directory are installed. None of the upstream hooks, MCP servers (Chrome DevTools), agent personas or slash commands (`/spec`, `/plan`, `/build`, `/test`, `/review`, `/ship`) are installed. The following skills are also missing, so ignore references to them:
  - browser-testing-with-devtools
  - code-simplification (use the built-in `/simplify` instead)
  - constraint-driven-development
  - context-engineering
  - deprecation-and-migration
  - doubt-driven-development
  - idea-refine
  - interview-me
  - source-driven-development
  - using-agent-skills
- Ask before installing a tool that isn't already a project dependency. That includes `npx <pkg>`, `npm i -g`, `pip install`, `cargo install`, `brew install`, and git hooks such as husky. `site/package.json` gates dependency install scripts with `allowScripts`, and an ad-hoc `npx` bypasses that gate.
- `ort` and `tokenizers` are pinned to exact versions on purpose:
  - `ort` release candidates break APIs between releases;
  - tokenization must stay byte-identical.

  Bump them only in a dedicated change that re-runs parity.
