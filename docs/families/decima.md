# decima (`decima-late-interaction-v1`, `-v2`)

**Decima** by A. M. Madani ([amyrmahdy on Hugging Face](https://huggingface.co/amyrmahdy),
[amyrmahdy/decima](https://github.com/amyrmahdy/decima), Apache-2.0) is a family of multilingual encoders
fine-tuned with a small **late-interaction scorer**: the state and every option are encoded on their own, and
each option reads the state's tokens through cross-attention to get one score. Options never see each other, so
their order cannot change the answer. Score questions go through a cumulative-link ordinal head. Requested in
#54.

| Tag | Weights (fp32, `pytorch/` of the repository) | Encoder | Temperature |
|---|---|---|---|
| `decima:base`, `decima:latest` | `amyrmahdy/decima-base@2468005d` (Hub tag `v2.0`): `encoder/model.safetensors` (1.23 GB) and `head.safetensors` (57 MB) | `jhu-clsp/mmBERT-base` (MIT), 321M parameters | 0.7269 |
| `decima:base-clean` | `amyrmahdy/decima-base-clean@35eb218b` (Hub tag `v1.0`; `main` changes only the README): the same two files, base retrained without non-commercial data | the same | 0.6930 |
| `decima:agent` | `amyrmahdy/decima-agent@652d90ac` (Hub tag `v2.3`): the same two files, fine-tuned from decima-base for coding-agent decisions and knowledge-graph judgments, states up to 8,192 tokens | the same | 0.6125 |
| `decima:guard` | `amyrmahdy/decima-guard@98d3f734` (Hub tag `v1.0-preview`): the same two files, a guard for LLM apps fine-tuned from decima-base-clean | the same | 0.8051 |
| `decima:guard-small` | `amyrmahdy/decima-guard-small@ac110248` (Hub tag `v1.0-preview`): `encoder/model.safetensors` (562 MB) and `head.safetensors`, guard's checks and training data | `jhu-clsp/mmBERT-small` (MIT), 140M parameters | 0.7546 |
| `decima:small` | `amyrmahdy/decima-small@2e7f4d07` (Hub tag `v1.1.1`): `encoder/model.safetensors` (471 MB) and `head.safetensors` (19 MB) | `intfloat/multilingual-e5-small` (MIT), 122M parameters | 0.9356 |

The temperatures are the author's, from each checkpoint's `decima.json`. The code is pinned to the GitHub tag
`v2.3.0` (`1b07634e`, tag object `6e3d2b92`), the release of decima-agent 2.3 and decima-guard. Small, base and
agent 2.1 were exported and checked against `v1.1.1` (`2df60942`), whose `decima/` was unchanged through `v2.0.0`;
`v2.1.0` added `decima/render.py` (unchanged at `v2.2.0`), which changes their option rows only when the old cut
makes two of a question's options identical (below). Against `v2.2.0` (`efaefe2a`), `v2.3.0` changes one line of
`render.py` (an option's full text is kept when its token count fits, without the character shortcut) and adds an
opt-in `"ordinal"` choice to `systemone.py`, which Ollaya does not take yet. Decima-agent 2.2 is the first
checkpoint whose `decima.json` has `"choice_fit": true`, so it runs on its own layout id,
`decima-late-interaction-v2`: an Ollaya without the fitted rendering refuses it instead of cutting its options.
Agent 2.3, guard and guard-small have it too. `decima:agent` pointed at 2.1 (`86a07aab`, Hub tag `v2.1`,
temperature 0.6082) until 2.2, and at 2.2 (`5c14a460`, Hub tag `v2.2`, temperature 0.7193, states up to 2,048
tokens) until 2.3.

**Decima-base-clean** is decima-base retrained "without any non-commercial data and without any data from the
BTZSC benchmark": the author's `release/LICENSING.md` names ANLI and XNLI (CC BY-NC 4.0), AG News and SST-5 among
base 2.0's training sources, and decima-agent inherits them. Use base-clean where that matters. It is not a
replacement for base: the author reports it lower on every set (typed decisions 0.477 against 0.484; BTZSC, Kev's protocol, Laya's MASSIVE and XNLI sets and the author's own bench 8 to 21 points lower), so
`decima:latest` stays on base. Its `decima.json` has `"choice_fit": true`, so it runs on
`decima-late-interaction-v2`, with base's 512-token state window. Its `tokenizer_config.json` drops the padding and
truncation keys the other pins carry; Ollaya reads only `tokenizer.json` (the same file as base's), and
`decima.check` finds the rows identical to upstream's.

The author also publishes int8 ONNX exports; Ollaya's graphs read the fp32 checkpoints instead, so that the weights
come unmodified from the files the author's own code loads.

## Rows

`crates/ollaya-decision/src/decima.rs` ports `decima/systemone.py` (`to_question`), the text and tokenization
steps of `decima/model.py` and `decima/render.py` (`choice_texts`); `convert/ollaya_convert/families/decima/layout.py`
is its Python twin. Every question becomes one state row and one row per option, each encoded on its own:

```text
state row   [cls] tok(state_prefix + normalize(text + "\n" + state))[..max_state_tokens - 2]  [sep]
option row  [cls] tok(option text)[..62]                                                   [sep]
full        option_prefix + normalize(text + " " + option)
fitted      option_prefix + head + " … " + normalize(option)
```

- **Option text** (`render.py`). `head` is the longest run of leading words of `normalize(text)` (Python's
  `str.split()`, joined by one space) with which the fitted text fits 64 tokens, found by upstream's binary
  search; with no word, the text is `option_prefix + normalize(option)`, cut if it is still too long.
  - **v2** (`option_fit: true` in `decision.json`, the author's `choice_fit`; decima-agent 2.2 and 2.3, guard
    and guard-small): an option keeps its full text when it fits 64 tokens, and any other option is fitted. Only
    the token count decides (`v2.3.0`; `v2.2.0` also kept any text of at most 62 characters, assuming a token is
    at least one character). The full question still goes on the state row.
  - **v1** (small, base, agent 2.1): every option keeps its full text, cut on the right as before, unless two of
    the question's cut rows come out identical (a long question pushes every option out of the window, and
    `yes` and `no` tie at 0.5). Then that question's options are rendered as in v2.
- **Per model** (`decision.json`): small uses e5's prefixes `query: ` and `passage: `, `[cls]` 0 and `[sep]` 2;
  base, agent and the guards use no prefix, `<bos>` 2 and `<eos>` 1. `max_state_tokens` is 512 for small and
  base, 8,192 for agent 2.3 (2,048 for 2.2) and 2,048 for guard and guard-small.
- **Text.** The question's instructions: a string as is, anything else as `json.dumps(ensure_ascii=False)`,
  the question id when they are absent or null. A noul question whose criteria give a `true` or `false`
  description appends `\nTrue if: ...\nFalse if: ...`, with U+2014 for the side without one.
- **Options.** A choice's `label: description` (the label alone when the description is empty), each score
  level's text, and `yes` and `no` for noul.
- **State.** A string as is; an object or array as `json.dumps(ensure_ascii=False)`. Numbers, booleans and
  null are rejected, as upstream's server rejects them.
- **normalize.** Unicode NFC, then Python's `str.strip()`.
- **Length.** A state row keeps its first `max_state_tokens` tokens and is flagged: `/v1/systemone` answers 422
  `STATE_TRUNCATED`, as upstream's server does, and `/api/decide` answers from the cut state (as upstream's
  `model.py` does) with `state_truncated: true`. An option row is cut to 64 tokens silently, as upstream cuts
  it.

`python -m ollaya_convert.families.decima.check --model <slug>` compares the port with upstream on the shared
request set: the edge cases, the 400 typed-decisions rows, 9 inputs Decima's mapping treats specially and 10
questions too long for the option window (`fit_cases`: long noul instructions with true/false criteria, long
choice and score questions, a question with no spaces, odd whitespace and NFD text, an option longer than the
window, a few characters that are many tokens). For small, base, agent 2.1 and agent 2.2, against upstream
`v2.2.0`: 479 requests with identical rows (12 of them with a cut state for small, 7 for base, none for agent),
18 rejected by both, 3 refused first by Ollaya's shared question rules, 0 differences.

Outside `fit_cases`, only one question of the set (an extra case with a 444-character option) has an option row
over 64 tokens, and its options stay apart, so the older models' rows there are the same as with `v1.1.1`. In
`fit_cases` the right-hand cut collapses 8 of the 10 questions for small and 7 for base and agent; with the
fallback, 8 of them get new rows for small and 6 for base and agent 2.1. Agent 2.2 fits 8 of them and the
long-option case, whether or not they collapse. Up to `v2.2.0` one collapse stayed: 40 Egyptian hieroglyphs and
`yes`/`no` are 45 characters but over 64 mmBERT tokens, so the character shortcut kept the full text and both rows
were cut to the same tokens (noul 0.5), with and without `choice_fit`. `v2.3.0` checks the token count only, so
for every model these options are fitted like any other; no other question of the set has a full option text of
62 characters or fewer over 64 tokens. The port follows `v2.3.0`.

## The graph

`python -m ollaya_convert.families.decima.export --model <slug>` writes one weightless graph that answers every
question of a request in one run. Its weight tensors (259 for small, 196 for base and agent) reference the two
checkpoint files by byte offset; only small's encoder pooler is left out.

```text
inputs   state_ids, state_mask     int64 [questions, state_len]   one state row per question, right-padded
         option_ids, option_mask   int64 [options, option_len]    every option row, questions in order
         option_state              int64 [options]                the question each option belongs to
outputs  scores                    float32 [options]              the raw score, before the temperature
         ordinal_g, ordinal_gap    float32 [options]              the ordinal head's two projections
```

The runner groups consecutive questions into one run as long as the padded tokens stay within the shared token
budget. Against upstream's PyTorch modules, on 295 questions, the scores are within 9.7e-6 (small), 1.7e-5
(base), 1.3e-5 (agent 2.1) and 1.5e-5 (agent 2.2), and the projections within 1.1e-5 (5.2e-5 for agent 2.2's
`ordinal_gap`). The `-v2` layout uses the same graph; only the option rows differ.

## Answers

- **Choice and noul.** A softmax of the scores over the temperature. Upstream's noul order is `[yes, no]`; the
  runner turns it into the `[false, true]` order answers use.
- **Score.** The ordinal head, on the scores over the same temperature, so the score slot of
  `calibration.json` is 1:

  ```text
  s = scores / T                    expected = sum_k softmax(s)_k * k - (K - 1) / 2
  g = mean_k ord_g(z_k) + expected  (ord_g is linear, so the graph returns it per option)
  gaps = softplus(ord_gap(z)) + 1e-3,  theta_j = sum_{i<=j} gaps_i - sum(gaps) / 2   (j < K - 1)
  p_k = sigmoid(g - theta_{k-1}) - sigmoid(g - theta_k)   (1 for k = 0, 0 for k = K - 1)
  log p = log_softmax(log(max(p, 1e-7)))
  ```

## Differences from upstream

- **Option cache.** Upstream's `Decima` caches the encodings of up to 256 option sets; Ollaya encodes every
  row on every request, in the same run as the state.
- **int8.** Not used (above).
- **Request options.** Upstream's server (2.2.0) also takes `"long": "chunk"` (decide an over-long state in
  overlapping windows), `"aggregate"` and `"min_confidence"` (an `abstain` flag), all off by default and outside
  TypeSafe's schema. Ollaya's `/v1/systemone` does not take them: a long state is still `STATE_TRUNCATED`.
- **Batching.** Upstream's `system_one` now runs a request's questions through `decide_many`, which the author
  states gives the numbers of one question at a time; Ollaya batches them in one graph run, as before.

## Parity

Goldens: `python -m ollaya_convert.families.decima.goldens out/<slug>` runs the author's `model.py` and
`systemone.py` in fp32 on the CPU. `parity_decima` checks the rejections and every row id for id, then the graph's
outputs, the decisions and the `/v1/systemone` answers against upstream's `system_one`. On CPU the ONNX Runtime is
pyke's 1.28.0, linked statically; on CUDA it is Microsoft's 1.28.2 from the CUDA 13 pack.

**Decima 2.3 (measured 2026-10-20).** Upstream `v2.3.0` (`1b07634e`): 154 requests per model, the 150 below and four
long agent states (3,041, 5,986, 8,076 and 8,266 tokens), 18 rejected upstream. Cut states: 7 for small and base,
1 for agent (the 8,266-token one), 4 for guard and guard-small.

| Model | Device | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| small 1.1 | x86-64 CPU | 599 | 2,986 identical | 599/599 | 1.2e-5 | 2.2e-6 |
| small 1.1 | CUDA, RTX 4090 | 599 | 2,986 identical | 599/599 | 1.1e-5 | 1.8e-6 |
| base 2.0 | x86-64 CPU | 599 | 2,986 identical | 599/599 | 2.0e-5 | 3.8e-6 |
| base 2.0 | CUDA, RTX 4090 | 599 | 2,986 identical | 599/599 | 1.0e-5 | 3.5e-6 |
| agent 2.3 | x86-64 CPU | 599 | 2,986 identical | 599/599 | 1.9e-5 | 5.4e-6 |
| agent 2.3 | CUDA, RTX 4090 | 599 | 2,986 identical | 599/599 | 3.2e-5 | 8.2e-6 |
| guard 1.0 preview | x86-64 CPU | 599 | 2,986 identical | 599/599 | 1.8e-5 | 5.8e-6 |
| guard 1.0 preview | CUDA, RTX 4090 | 599 | 2,986 identical | 599/599 | 2.0e-5 | 7.3e-6 |
| guard-small 1.0 preview | x86-64 CPU | 599 | 2,986 identical | 599/599 | 1.8e-5 | 7.5e-6 |
| guard-small 1.0 preview | CUDA, RTX 4090 | 599 | 2,986 identical | 599/599 | 9.1e-6 | 4.8e-6 |

The ordinal projections are within 9.9e-5, and every `/v1/systemone` answer matches upstream's to 4 decimals.

**Decima-base-clean 1.0 (measured 2026-10-24, hope).** Upstream `v2.3.0`, the same 154 requests, 18 rejected upstream,
7 with a cut state:

| Model | Device | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| base-clean 1.0 | x86-64 CPU (Threadripper 3960X) | 599 | 2,986 identical | 599/599 | 3.2e-5 | 3.7e-6 |
| base-clean 1.0 | CUDA, RTX 5090 (Microsoft's ONNX Runtime 1.28.2) | 599 | 2,986 identical | 599/599 | 1.8e-5 | 7.0e-6 |

The ordinal projections are within 4.5e-5, and every `/v1/systemone` answer matches upstream's to 4 decimals.
`decima.check --model decima-base-clean`: 483 requests identical to upstream (11 with a cut state), 18 rejected by
both, 3 by Ollaya's shared rules, no mismatch.

**Decima-agent 2.2 (measured 2026-10-13).** Upstream `v2.2.0`: 150 requests, the 140 below and the 10
`fit_cases`, 18 rejected upstream.

| Model | Device | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| agent 2.2 | x86-64 CPU | 591 | 2,958 identical | 591/591 | 2.2e-5 | 5.4e-6 |
| agent 2.2 | CUDA, RTX 4090 | 591 | 2,958 identical | 591/591 | 2.2e-5 | 4.2e-6 |
| agent 2.2 | CUDA, RTX 5090 | 591 | 2,958 identical | 591/591 | 1.3e-5 | 2.4e-6 |

The ordinal projections are within 1.4e-4 (CPU), 5.7e-5 (RTX 4090) and 2.9e-5 (RTX 5090).

**The older models on this runtime (2026-10-13).** On their 0.11.0 goldens (`v1.1.1`, below) the rows and numbers
are those of 0.11.0: small 1.2e-5 / 1.5e-6, base 9.2e-6 / 2.7e-6, agent 2.1 1.0e-5 / 2.8e-6 (scores / probabilities)
on the RTX 5090, 1.1e-5 / 1.3e-6, 9.1e-6 / 3.5e-6, 1.6e-5 / 5.5e-6 on the RTX 4090, and 1.1e-5 / 3.1e-6,
2.1e-5 / 3.7e-6, 2.0e-5 / 4.2e-6 on a second x86-64 CPU. Goldens from
`v2.2.0` for the same three models (150 requests each, so the `fit_cases` collapses and their fallback rows are
upstream's) have the same rows as the `v1.1.1` goldens on the 140 shared requests, and pass with 2,958 identical
rows, every decision upstream's, scores within 1.8e-5 and probabilities within 4.7e-6, on CPU and the RTX 5090;
on the RTX 4090, small 9.7e-6 / 1.6e-6, base 1.0e-5 / 3.6e-6 and agent 2.1 2.7e-5 / 7.7e-6.

**Decima 2.0 and 2.1 (measured 2026-10-06).** Upstream `v1.1.1`: 140 requests per model, 18 of them rejected
upstream, 3 with a cut state for small and base (none for agent, whose states hold 2,048 tokens).

| Model | Device | Questions | Rows | Decisions | Scores max | Probabilities max |
|---|---|---|---|---|---|---|
| small | x86-64 CPU | 581 | 2,922 identical | 581/581 | 1.1e-5 | 2.3e-6 |
| small | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 1.1e-5 | 1.3e-6 |
| small | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 1.2e-5 | 1.5e-6 |
| base | x86-64 CPU | 581 | 2,922 identical | 581/581 | 2.0e-5 | 3.0e-6 |
| base | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 9.1e-6 | 3.5e-6 |
| base | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 9.2e-6 | 2.7e-6 |
| agent 2.1 | x86-64 CPU | 581 | 2,922 identical | 581/581 | 2.0e-5 | 4.3e-6 |
| agent 2.1 | CUDA, RTX 4090 | 581 | 2,922 identical | 581/581 | 1.6e-5 | 5.5e-6 |
| agent 2.1 | CUDA, RTX 5090 | 581 | 2,922 identical | 581/581 | 1.0e-5 | 2.8e-6 |

The ordinal projections are within 6.8e-5 everywhere. On every device the `/v1/systemone` answers name the same
choice as upstream's on every question, and their numbers agree to the fourth decimal (1.0e-4, the rounding).

## Quality

Typed-decisions: all 400 test states, 2,000 questions, argmax against the majority label, through Ollaya's
runtime (`logits` example and `quality_runtime.py`; small on the CPU, base and agent on CUDA). ECE at the
author's temperature. No typed-decisions question has an option row over 64 tokens, so the rendering change does
not touch these numbers.

| Model | Accuracy | Choice | Score | Noul | ECE |
|---|---|---|---|---|---|
| base | **0.495** (989 of 2,000) | 0.432 | 0.496 | 0.555 | 0.073 |
| agent 2.1 | 0.486 (971) | 0.470 | 0.380 | 0.642 | 0.163 |
| agent 2.2 | 0.461 (921) | 0.450 | 0.414 | 0.533 | 0.193 |
| small | 0.432 (863) | 0.412 | 0.356 | 0.552 | 0.110 |

The author reports 0.427 (95% interval 0.399 to 0.455) for small 1.1 and notes that every small model they
tested was below this benchmark's majority-class baseline of 0.461; base and agent are above it. The author
gives no typed-decisions figure for base or agent. Agent is trained for the decisions inside a coding agent's
loop (secret and command gates, tool, command and model choice); 2.2 adds knowledge-graph judgments and gate
fixes from field logs. The author measures it on 40 agent cases written after training: 0.85 for 2.2, 0.83 for
2.1, 0.60 for base (their earlier 130-case set had near-copies in training data). Agent 2.2 is 0.025 below 2.1 on
typed-decisions (Ollaya's measurement, 2026-10-13; 2.1 reproduced at 0.486 the same day), at this benchmark's
majority baseline, so `decima:latest` stays on base.

## Speed (measured 2026-10-06)

The triage preset (five questions) through the HTTP API, one request at a time (`bench_latency.py`, the
protocol of the 2026-10-01 sweep), on the RTX 4090 machine:

| Model | RTX 4090 p50 | p90 | CPU (i9-13900K) p50 | p90 | Load (CUDA / CPU) |
|---|---|---|---|---|---|
| small | 7.3 ms | 8.5 ms | 146 ms | 148 ms | 0.84 s / 0.58 s |
| base | 15.1 ms | 15.4 ms | 438 ms | 452 ms | 1.43 s / 1.37 s |
| agent 2.1 | 14.8 ms | 15.9 ms | 439 ms | 442 ms | 1.27 s / 1.29 s |

Agent 2.2 runs the same graph on the same file sizes and was not timed again. On this CPU no other model Ollaya
ships is as fast as small: the next are `laya:multilingual` at 308 ms and
`gliclass` at 563 ms (2026-10-01). The author measured one question with four options on one CPU core with
their int8 exports and cached option encodings (14 ms for small, 47 ms for base on an NVIDIA GB10's ARM cores);
Ollaya runs fp32 and encodes the options on every request.

## Limits

- **State.** 512 tokens per question for small and base, 8,192 for agent and 2,048 for guard and guard-small,
  the question's text included.
  Longer: `/v1/systemone` answers `STATE_TRUNCATED`, `/api/decide` answers from the first tokens.
- **Options.** Each option row holds the question's text and the option in 64 tokens. Agent and the guards keep
  the option whole and shorten the question beside it; small and base keep the right-hand cut their training saw, unless it
  makes two options identical (above). An option alone over 64 tokens is still cut. 2 to 255 choices, 2 to 10
  score levels, 1 to 256 questions.
- **Languages.** Multilingual. For small the author evaluated 20 languages (the weakest are Swahili and Hindi);
  for base, Persian costs about 5 points against English on the same tasks, and other languages are less tested.
