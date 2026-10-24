# jeeves (`jeeves-markers-v1`)

PostHog's **Jeeves-9B** ([PostHog/jeeves](https://huggingface.co/PostHog/jeeves), Apache-2.0; code at
[github.com/PostHog/jeeves](https://github.com/PostHog/jeeves), MIT) is Qwen3.5-9B with a LoRA merged in and a
Kev-style pointer head. Upstream it writes a reasoning chain before the head reads the answer when it is
unsure; with the chain skipped it still answers in one forward pass. Ollaya runs that **no-thinking** mode,
which the authors report at 0.812 on their test split for v2 (0.851 reasoning when unsure; v1: 0.804 and
0.840). Requested in #36.

**Jeeves v2** (2026-10-16, COB-562) is SFT only (one epoch, 16,867 steps) on public data, about 100 sources
(`export.json`); v1 was SFT then CISPO. It has the same base, LoRA scheme, head shape, marker format and
tokenizer files as v1: the shards' and `head.pt`'s layouts are byte-identical, only their values changed, and the
authors' `Encoder`, request parsing and option texts did not change (GitHub `6151619c` → `8c6ab532`). The no-think
temperature is 1.522; `think_temperature` (5.4195) applies only after a reasoning chain, which Ollaya does not run.
v1 stays reachable as the Hub tag `v1` (`8622b7d1`, temperature 1.859); the Hub has no `v2` tag, so the pin is
`main`.

| Tag | Weights | Temperature |
|---|---|---|
| `jeeves:9b`, `jeeves:latest` | `PostHog/jeeves@04b9f87a` (v2): five BF16 shards (the fused model) and `head.pt` | 1.522 (`export.json`) |

Thinking mode needs a generation loop in the runner (and the authors' drafters); Ollaya does not run it.

## Rows

`convert/ollaya_convert/families/jeeves/layout.py` ports the authors' `Encoder` (`loader/dataloader.py`,
format `markers-v3-plainchains`) and request parsing (`inference/api.py`) at `6151619c`:

```text
content = <|fim_prefix|> + esc(render(state)) + "\n" + <|fim_middle|> + esc(render(instructions)) + "\n" + block
block   = (<|box_start|> + esc(option) + <|box_end|> + "\n")*
ids     = tok(chat(content)) ⧺ tok("\n") ⧺ tok("</think>\n\n" + block + <|fim_suffix|>)
```

- `chat` is Qwen's template around one user message with thinking on (it ends in `<think>\n`, and it trims
  the message); `"\n"` is the empty thought.
- `render` and the option texts are Kev's (`prep/format.py`): noul reads `no` then `yes`, a choice
  `name` or `name: description`, a score its levels. `esc` rewrites `<|name|>` so text cannot form a marker.
- The pointer head scores option j as `k(h[opt_j]) · q(h[decide]) / 16`, at the j-th `<|box_end|>` after the
  last `</think>` and the final `<|fim_suffix|>`.
- **Tokenizer.** Upstream tokenizes through transformers 5, whose `Qwen2Tokenizer` rewrites the
  pre-tokenizer of Jeeves' `tokenizer.json` (as for Kev). `jaredpalmer/kev-9b`'s `tokenizer.json` is exactly
  that tokenizer, identical in every field, so the manifest takes the tokenizer from there.

`python -m ollaya_convert.families.jeeves.check <snapshot> --tokenizer <kev-9b tokenizer.json>` compares
the port with the upstream Encoder on the shared case set: 265 requests with identical token rows, the same 16
rejections.

## The graph

Kev's contract (`input_ids`, `decide_pos`, `opt_pos` → `scores`), 13 MB, weightless: transformers' Qwen3.5 text
model with Jeeves' fused weights, recomputed by `llm_common/qwen35.py`, and the head's four tensors read by
byte offset from `head.pt`. `weights_in_memory` is `bf16`, about 18 GB. The runner uses a 4,096-token budget
per `session.run`, as for `nimble`.

## Differences from upstream

- **No thinking.** The reasoning chain and the drafters are not run.
- **Strict questions.** As upstream, a question with fields other than `type`, `instructions` and `criteria`
  is rejected, and choice criteria must be an object.

## Parity (v2, measured 2026-10-24)

Goldens: the authors' own model code (`export.load_export`, PostHog/jeeves at `8c6ab532`) in fp32 on the CPU, with
their Encoder, over the shared case set (10 typed-decisions rows and the edge cases): 107 requests, 16 rejected
upstream, 430 questions. The CPU runs the 18 smoke cases (K2 for a 9B model).

| | rows | decisions | scores max | probabilities max (p99) | per request (runner, mean) |
|---|---|---|---|---|---|
| CUDA, RTX 5090 | 430 / 430 identical, 0 rejection mismatches | 100 % | 7.8e-5 | 5.8e-6 (2.6e-6) | 422 ms on the fixtures |
| CPU, Threadripper 3960X (smoke) | 44 / 44 identical, 0 rejection mismatches | 100 % | 4.8e-5 | 3.0e-6 | 40.8 s on the fixtures |

v1 (2026-09-30, RTX 4090): the same 430 questions, scores within 1.9e-4, probabilities within 1.4e-5.

## Quality

Ollaya Bench, edition 2026.10 (0.18.0 update), RTX 5090:

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label): **0.790**
  (choice 0.760, score 0.754, noul 0.868), ECE **0.225** at the authors' temperature 1.522. The mean top
  probability is 0.565, so v2 is underconfident on this split. v2 trained on typed-decisions' train split
  (`export.json`), so the test split is in-domain for it. v1: 0.680, ECE 0.028 at 1.859.
- **Public benchmark** (13 subsets): macro accuracy 0.756, macro ECE 0.083 (pooled 0.018).
- **Score.** 54.0 without the 8 tasks v2 trained on (typed-decisions, BoolQ, MultiNLI, PAWS, civil_comments,
  HelpSteer2 and both MASSIVE subsets). v1 scored 61.1 over all 14, so the two scores are not comparable.
- **Latency** (five questions, p50 over HTTP): 388 ms on CUDA, 19.4 s on the CPU. v1: 380 ms and 20.5 s.

## Limits

- **Rows.** Up to 8,192 tokens per question, the whole state included; a longer row is rejected.
- **Memory.** About 18 GB with the weights kept BF16: a 24 GB GPU.
