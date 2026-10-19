Decima by [A. M. Madani](https://huggingface.co/amyrmahdy) is a family of multilingual decision models built on a late-interaction scorer. The state and each option are encoded on their own, and every option reads the state to get its score, so the order of the options never changes the answer. Score questions go through an ordinal head. It never generates text.

> Needs Ollaya 0.11.0 or newer, the first release that runs the `decima-late-interaction-v1` layout. `decima:agent` (Decima-agent 2.2) needs 0.13.0 or newer, the first release that runs its `decima-late-interaction-v2` layout; an older version downloads the weights and then refuses to load them.

## Models

| Tag | Model | Typed-decisions accuracy | Five questions, RTX 5090 / CPU |
|---|---|---|---|
| `decima:latest`, `decima:base` | Decima-base 2.0 (mmBERT-base, 321M), fp32, 1.3 GB | {{result decima:base typed.accuracy 3}} | {{result decima:base gpu.p50_ms 0}} ms / {{result decima:base cpu.p50_ms 0}} ms |
| `decima:agent` | Decima-agent 2.2 (decima-base for coding agents), fp32, 1.3 GB | {{result decima:agent typed.accuracy 3}} | {{result decima:agent gpu.p50_ms 0}} ms / {{result decima:agent cpu.p50_ms 0}} ms |
| `decima:small` | Decima-small 1.1 (multilingual-e5-small, 122M), fp32, 490 MB | {{result decima:small typed.accuracy 3}} | {{result decima:small gpu.p50_ms 0}} ms / {{result decima:small cpu.p50_ms 0}} ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya. Calibration error (ECE) at the author's temperatures: {{result decima:base typed.ece 3}} for base, {{result decima:agent typed.ece 3}} for agent, {{result decima:small typed.ece 3}} for small. The author reports 0.427 for small, and notes that small models struggle with this benchmark's long, multi-fact business cases. Latency is the median request through the HTTP API on an RTX 5090 and on the CPU of the same machine (Threadripper 3960X), Ollaya Bench.

`decima:small` is the fastest model Ollaya runs on a CPU. `decima:agent` is trained for the small, frequent decisions inside a coding agent's loop: is there a real secret in this edit, should this command run, ask or be denied, which tool or command fits, which model tier a task needs. Version 2.2 adds knowledge-graph judgments (fact status, entity matching, relation selection) and gate fixes from field logs; the author measures 0.85 on 40 agent cases written after training (2.1: 0.83, base: 0.60).

## Usage

```shell
ollaya run decima --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `decima`, `decima:agent` or `decima:small`.

## How it works

- **Rows.** Each question becomes one state row (the question with the state) and one row per option (the question with the option), exactly as the author's `decima/systemone.py`, `model.py` and `render.py` build them.
- **Scores.** The options of every question are scored in one pass. Choice and yes/no answers are a softmax of the scores at the author's fitted temperature; score questions use the author's cumulative-link ordinal head.
- **Engine.** ONNX Runtime, fp32, on an NVIDIA GPU (CUDA) or the CPU. The graph reads the author's own checkpoint files, unmodified.
- **Parity.** Ollaya's runtime matches the author's own code (fp32) on 581 questions per model (591 for `decima:agent`, with ten questions too long for the option window), on the CPU and on CUDA: identical token rows, the same decision on every question, probabilities within 5.4e-6.

## Limits

- **State.** 512 tokens per question (2,048 for `decima:agent`), the question included. `/v1/systemone` answers a longer state with `STATE_TRUNCATED`, as the author's server does; `/api/decide` answers from its first tokens.
- **Options.** Each option row holds the question and the option in 64 tokens. When they don't fit, `decima:agent` keeps the option whole and shortens the question's copy beside it (the full question is still read with the state), as the author's 2.2 runtime does. `decima:base` and `decima:small` keep cutting the row on the right, which their training saw, except when that would make two options identical. 2 to 255 choices, 2 to 10 score levels.
- **Languages.** Multilingual. The author evaluated small in 20 languages (the weakest are Swahili and Hindi). For base, the author reports Persian about 5 points below English on the same tasks, and other languages less tested.
