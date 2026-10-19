# 0007: Ollaya Bench, a deployment benchmark for decision models

- Status: accepted, 2026-10-14; first edition 2026.10 (Ollaya 0.13.0, RTX 5090 desktop)
- Applies to: `results/` (runs and editions), `convert/ollaya_convert/bench_*.py`, the bench target in
  `site/` (`src/bench/`, `scripts/build-bench.mjs`, `wrangler.bench.jsonc`) and bench.ollaya.dev.

## Context

Within a month of Jev, at least four public rankings of decision models appeared: the Decision Index
(community, most of its score from private tests), JevBench (latency from rented GPUs with an assumed
×2 + 0.15 s adjustment), S1MB and DecideBench. All of them rank accuracy. None measures a model on the
hardware a user would run it on, and none checks that the runtime it measures computes what the
model's authors' code computes.

Our own numbers were scattered: `/results` held a public-benchmark run from 2026-09-30 on 19 models,
latency runs from three dates on two machines, and typed-decisions accuracies typed by hand into the
catalog from each family's PyTorch reference.

## Decision

Ollaya Bench measures every model in the library the way a user runs it, and publishes everything:

1. **One released runtime.** Each edition runs a published Ollaya release from the install script,
   every model at its shipped tag and default precision, with its authors' temperatures. Every result
   records the model's manifest digest.
2. **Accuracy once, speed everywhere.** Parity makes a model's decisions device-independent within its
   checked tolerances, so the accuracy suites run once on the reference GPU; speed and memory run on
   every device of the edition.
3. **The score.** The mean, over 14 tasks with equal weight (Bespoke Labs' 13 public subsets and
   typed-decisions), of chance-corrected accuracy `100 × (acc − chance) / (1 − chance)`. Rejected
   requests are wrong. 95 % bootstrap intervals (fixed seed); models whose intervals overlap share a
   rank. Models trained on a task's train split stay on the board with a flag and are left out of the
   recommendations.
4. **Editions.** `results/editions/<id>.json` freezes the runs, models and scores of one session.
   The site shows the newest edition; old ones stay as published.
5. **Tools need only the standard library** (`bench_typed.py`, `bench_latency.py --memory`,
   `bench_public.py` with Bespoke's runner), so anyone can rerun a number. `bench_edition.py` turns the
   raw outputs into result files; `convert/tests/test_bench_scoring.py` covers the scoring.
6. **Hosting.** A second static target of `site/`, sharing its stylesheet, catalog and components,
   served by its own assets-only Worker (`ollaya-bench`) on bench.ollaya.dev at no cost. ollaya.dev's
   navigation links to it; `/results` stays as the archive of earlier runs, with the comparison with
   Ollama and the Windows and Vulkan measurements.

## Consequences

- typed-decisions numbers now come from the runtime over HTTP. They differ slightly from the catalog's
  reference numbers where Ollaya rejects a long input (`STATE_TRUNCATED`) that the reference truncated.
- An edition takes about 7.5 hours on the RTX 5090 desktop (pulls, GPU speed, accuracy, CPU speed);
  `convert/ollaya_convert/bench_run.py` runs it. Between editions, releases update the current one with what
  they changed ([ADR-0009](0009-measurement-per-release.md)).
- Not yet covered: other devices (more machines, community runs), hosted models such as Jev, a
  held-out slice per edition, and the per-question outputs as a public dataset.
