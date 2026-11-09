# ADR-0010: Compute a request's state once, and fork it into each question

## Status

Accepted for the ONNX graphs of `kev` and `decider`. The other families, MLX and llama.cpp are follow-ups (see
Consequences).

## Date

2026-11-10

## Context

The Qwen3.5 decoder families answer one question per row, and every row starts with the request's state. A
five-question request computes that state five times, plus the padding of each row to a multiple of 64. `decider`
also gives each level of an isolated score question its own row, and every `nimble` row carries the whole request
(only the requested field at its end differs).

Flide ([OmniJev/Flide](https://github.com/OmniJev/Flide), Apache-2.0) serves the same kind of models from one
computation of the state, forked into one branch per question, and calls it exact state forking: each branch
continues from the Gated DeltaNet state, the conv window and the attention keys at the end of the shared prefix,
so it computes the same function as the question asked alone. Flide is Python, PyTorch and Triton on CUDA, so
Ollaya takes the idea and none of the code.

## Decision

1. **A fork form of the graph.** `Qwen35Trunk.fork` (`convert/ollaya_convert/families/llm_common/qwen35.py`) takes
   `prefix_ids [J, P]`, `branch_ids [B, S]` and `branch_job [B]`. Row `b` is `prefix_ids[branch_job[b]] ++
   branch_ids[b]`; read positions count from the start of the row, so the heads read what they read before. Every
   prefix runs once, and each branch starts from its prefix's DeltaNet state and conv window and attends to its
   prefix's keys.
2. **Chunk-aligned prefixes.** P and S are multiples of 64, the gated delta rule's chunk. The prefix's chunks are then
   the row's chunks, and so are the branch's. The function is the same, not an approximation: in eager fp32 the
   hidden states are bit-identical to the row form's on kev-0.8b and decider-0.8b. The goldens stay as they are, and
   the gate is unchanged.
3. **The split lives in the runner.** `crates/ollaya-runner/src/fork.rs` cuts a request's token rows, with no family
   code:
   - rows that share 64 tokens or more fork at their longest common prefix, rounded down to 64 and short of every
     row's end (one job);
   - otherwise each row is its own job, cut one chunk before the end of the longest row, so the run costs what the row
     form costs;
   - a run carries the prefix once and as many branches as the token budget leaves room for, at least half of it, so
     a prefix over the budget still runs once per run.

   `qwen35.fork_rows` is the same split in Python, for the exports and `ort_rows.run_rows`.
4. **Both forms run.** A family's runner takes the fork path when its graph has a `prefix_ids` input and the row path
   otherwise, so installed models keep working and each family moves when its fork graph passes its gate.
   `decision.json` names the form (`contract.form`).

## Results

kev-0.8b and decider-0.8b in fp32 on ONNX Runtime, CPU, against goldens made from the authors' code (the edge cases
and 40 typed-decisions requests):

| Model | Questions | Decisions the same | Max logit difference |
|---|---:|---:|---:|
| kev:0.8b | 580 | 100% | 4.5e-5 (as the row graph) |
| decider:0.8b | 579 | 100% | 3.8e-5 |

On the 40 typed-decisions requests, kev's fork graph computes 35,968 tokens instead of 67,200. Latency of the same
requests on one development machine (an Apple M5, CPU, `bench --ids td/`), not an edition machine:

| Model | p50, row → fork | p95, row → fork | Memory, row → fork |
|---|---|---|---|
| kev:0.8b | 1,920 → 1,007 ms | 3,470 → 1,386 ms | 6.6 → 4.2 GiB |
| decider:0.8b | 4,543 → 1,878 ms | 7,637 → 2,707 ms | 9.4 → 6.1 GiB |

The gain grows with the state: short one-line states save little, long ones approach one state per request instead
of one per question.

## Consequences

- A family moves to the fork form with its export and its runner, and ships after its gate on CUDA and the CPU, as
  any graph change does (`docs/model-lifecycle.md`). kev's and decider's larger tags, the `-fp16` tags and CUDA are
  not gated yet.
- Still in the row form: `nimble` and `jeeves` (converted on a branch, waiting for a 9B gate), `decision`,
  `decision2`, `d3`, `strands`, `arbiter` and `decider-vision`. Attention-only trunks (Qwen3, Gemma) need only the
  keys and values.
- The MLX engine computes from the weights, not the graph, so it needs its own fork (`mlx/qwen35.rs`) and its own
  parity run on Metal before it can take this path.
- llama.cpp can fork a recurrent cache (`llama_memory_seq_cp` copies the state on write), but that changes the
  model's batch split, which moves its numbers away from the goldens of the authors' client. That needs its own
  decision.
- The contract can hold several requests in one run (`branch_job`), which batching across requests and a cache of
  agents' states would build on. Neither is part of this decision.
