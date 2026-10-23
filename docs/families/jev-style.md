# jev-style (`jev-style-v1`)

**chaoliangUNSW/Jev-Style-2B-Decision-v3** is a Qwen3.5-2B fine-tune (Apache-2.0), a non-generative
typed-decision model. Unlike `jevk5` / `llm-logits`, it is **not** a last-token label-logit readout:
it uses a plain-text "render v2" input (no chat template, no system prompt, no answer letters) and a
**verdict readout** — `logit(" yes") - logit(" no")` at a per-option ` ->` slot — read under
**non-causal block attention**. The author ships its own GGUF scorer (`jev-score-v2`); this family
reproduces that scorer on Ollaya's llama.cpp runner. The author publishes GGUF builds in
**chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF**. Ollaya ships the 2B Q4_K_M file as `jev-style:2b`.

| | |
|---|---|
| Repo / commit | `chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF@283b9eb7903aeb184b16d49ab8f38b256f4afcb3` |
| File | `Jev-Style-2B-Decision-v3-Q4_K_M.gguf` 1.27 GB, sha256 `a0140af1710ccff8c00e7e911b1ec0697467a023fcdb3ffde6408dd474d5e124` (`manifest.json`) |
| Source model | `chaoliangUNSW/Jev-Style-2B-Decision-v3` (Qwen3.5-2B fine-tune, Apache-2.0) |
| Reference | the model's own runtime `jev_style_decision_gguf.py` (shared core) and `jev_score_v2.cpp`, in the GGUF repo |
| Readout tokens | ` yes` = 9542, ` no` = 874, ` ->` = 1411 (`readout_config.json` `slot_tokens`; checked against the GGUF at load) |
| Temperature | one global `0.8278650620942867` (`readout_config.json` `temperatures.global`), applied by Ollaya's calibration like every family |
| Architecture | `qwen35` (Gated-DeltaNet + attention); block-causal in the 6 full-attention layers |

## Engine: non-causal block attention on llama.cpp

Where `jevk5`/`llm-logits` read a single causal context, jev-style needs the non-causal, block-wise
decode its scorer uses:

- **One non-causal context.** The context is created once with
  `attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL`, `n_ubatch = 2048`, `n_seq_max = 2` and
  `n_outputs_max` slots (`decision.json` `llama`), and is never toggled: a non-causal ubatch sees
  all earlier cache cells of its sequence plus itself, with no causal mask inside the block — the
  block-causal layout of training. The Gated-DeltaNet layers stay recurrent.
- **One decode per renderer block** (never merged, never re-split), each `<= n_ubatch`.
- **State computed once.** The state decodes into sequence 0 once; each question does
  `seq_cp(0 -> 1)`, decodes its question-side block(s) into sequence 1 with logits requested **only**
  at the ` ->` slots, reads `logit(" yes") - logit(" no")` at each, then `seq_rm(1)`. One question per
  decode; the state never re-decodes within a request.

See [llm-logits.md](llm-logits.md) for the shared llama.cpp runtime and
[decisions/0003-llama-cpp-runtime.md](../decisions/0003-llama-cpp-runtime.md).

## Prompt (render v2, port of `jev_style_decision_gguf.py`)

```text
state prefix   State:\n<state>\n\n
short form     Question [<type>]: <question>\nOptions:\n
(q+opts+slots   - <option 1> ->\n ... - <option K> ->\n
 <= 2048 tok)
overflow form  Question [<type>]: <question>\nOptions:\n
(otherwise)    Option 1: <option 1>\n ... Option K: <option K>\n            (the catalogue)
               Judge each numbered option in the complete catalogue above:\n
               Option 1 ->\n ... Option K ->\n                               (the rubric)
```

- `<type>` is `choice`, `score` or `noul`. The state is `json.dumps`ed (`ensure_ascii=False`) unless
  it is already a string. Option bodies (`render_options`): choice `"{name}"` or `"{name}: {desc}"`;
  score `"level {i}: {desc}"`; noul `"false: …"` / `"true: …"` (default
  `"no, the statement does not hold"` / `"yes, the statement holds"`).
- Everything is tokenised with **no BOS/EOS and special tokens split** (never control tokens), so
  `<|im_end|>` in a state is text.
- The complete input (state + question + options + readout) must fit **25,600** tokens; nothing is
  truncated — a longer input is rejected. There is **no option cap** (the short form uses one block;
  more than one block's worth of options switches to the catalogue + rubric form).

## Score and probabilities

Option `k`'s score is `logit(" yes") - logit(" no")` at its ` ->` slot, in float32. Probabilities are
`softmax(scores / T)` in canonical option order (choice: criteria keys; score: levels; noul:
`false, true`), with the one global temperature — applied by Ollaya's calibration, as for every
family, so `calibration.json` carries `T` once per type.

## Reference implementation

`crates/ollaya-decision/src/jev_style.rs` ports the render v2 layout and verdict slots (unit-tested
against the reference render); `crates/ollaya-runner/src/llama/mod.rs` `run_jev_style` ports the
block-decode and readout from `jev_score_v2.cpp`.
