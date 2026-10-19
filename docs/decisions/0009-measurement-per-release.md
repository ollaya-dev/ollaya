# ADR-0009: What a release measures, and one source for every number

## Status

Accepted

## Date

2026-10-19

## Context

Ollaya Bench 2026.10 ([ADR-0007](0007-ollaya-bench.md)) took 7.5 hours on the bench machine. Nothing decided what
the next release had to measure, so each release ended up either re-running far too much or forgetting a model.
Some examples from the weeks after the edition:

- A 0.14.0 plan that re-ran every suite on every changed model came to three hours. The graphs had changed, but
  they passed parity with the same numbers, so their accuracy could not have moved.
- strands-decider, pplx-decider and nimble:9b-v3 shipped in the library and never appeared on bench.ollaya.dev.
- The scripts that ran the edition lived only on the bench machine (`~/ollaya-bench/*.sh`), as did the order of
  the steps. Waits between machines were chained by hand, and one of them waited on itself for 45 minutes.
- Numbers were typed into the site in about a hundred places: catalog stats, library pages, the home page, the
  FAQ and the README. They drifted apart. For example, the README gave Ollama's Nimble ECE as 0.122, from a
  September run, while the FAQ and the home page gave 0.115, from Ollama 0.40.6. The 0.40.6 numbers existed only
  in a Markdown note, not in `results/`.

## Decision

1. **A rolling edition.**
   - A release adds what it changed to the current edition. Its run files join the edition's runs, where a later
     row replaces an earlier row of the same model, and `updates` names the release.
   - Every row keeps the version that measured it.
   - A new edition, with all suites for every model, runs on a schedule (about once a quarter) or after a
     runtime-wide change, never for one release.
2. **The change decides the measurement.** `bench_plan` compares each tag's manifest with the one the edition
   measured, layer by layer:

   | Change | Measure |
   |---|---|
   | New model, or new weights or tokenizer | Parity (CUDA and CPU), typed-decisions, public, latency on GPU and CPU, integration record |
   | Graph or engine change that passed the fp32 gate | Parity on CUDA; GPU latency (CPU latency on request); accuracy carried |
   | New precision tag (`-fp16`), or a change to one | Its own gate, typed-decisions, GPU latency |
   | Runtime-wide change (ONNX Runtime, the scheduler) | Parity of every family; a latency sweep, or a new edition when decisions moved |
   | Docs, site, catalog text | Nothing |

   A graph change with no passing parity record since the edition's last update stops the plan.
3. **Precision variants sit under their base tag.**
   - An `-fp16` tag is shown on its base tag's page, with its speed, memory and agreement.
   - It is not ranked: its decisions are within its gate of the base tag's, and a second row would count the same
     model twice.
4. **`results/` is the only source of numbers.**
   - Pages refer to measurements (`{{result <tag> <path> <digits>}}`), and the build fills them in from `results/`.
   - The build fails on a reference that has no result.
   - A figure someone else published (a model card, a vendor's benchmark) may be quoted, attributed in the same
     sentence. It is never presented as our measurement.
   - The README's numbers are generated blocks, and the build fails when they are stale.
   - Parity runs are result files too (`parity_run`), so the results page and the bench read them like any other run.
5. **Tools in the repository, runs that resume.**
   - `parity_run`, `bench_plan`, `bench_run` and `bench_edition --update` replace the bench machine's scripts.
   - They use only Python's standard library (except the parity examples, which need the Rust build), so a
     contributor runs the same commands.
   - A bench run resumes where it stopped.
   - Every step and its budget are in [docs/model-lifecycle.md](../model-lifecycle.md).

## Alternatives considered

- **A new edition for every release.** Every release would pay the 7.5 hours, or editions would carry most rows
  unchanged under a new number. The rolling edition is the same thing without the copy.
- **Measure everything, always.** It is the safest option, and it is the reason releases took hours. Parity already
  proves that a graph's outputs did not move, so re-measuring its accuracy adds time and no information.
- **Edition-only bench.** The bench stays frozen between editions, and new models stay off it for months.

## Consequences

- A release that only changes graphs measures in under an hour. A release with a new model takes about two hours.
- bench.ollaya.dev rows can come from different releases. The model page says which release measured each row,
  and the methodology page says how a model enters.
- Scores stay comparable within an edition because the machine and the protocol don't change. A runtime-wide
  change that moves decisions starts a new edition.
- Hand-typed numbers have to move to references, and the build check keeps them from coming back.
