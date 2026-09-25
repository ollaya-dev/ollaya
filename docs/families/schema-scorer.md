# schema-scorer (`schema-pairs-v1`)

`mobarmg/jev-schema-scorer-deberta-v3-large` is a DeBERTa-v3-large cross-encoder with one scalar
logit. It was trained on the TypeSafe question schema itself: every candidate answer of every
question is a `(state, question-schema + candidate)` pair, and a softmax over a question's
candidates is the answer. The request is consumed as is, with **no Ollaya mapping layer** apart
from list-valued choice criteria.

| | |
|---|---|
| Upstream | [mobarmg/jev-schema-scorer-deberta-v3-large](https://huggingface.co/mobarmg/jev-schema-scorer-deberta-v3-large) @ `ee092c35cc4ba0bd81be06a351b8788a47668d8c`; code: `schema_scorer.py` in that repo (MIT) |
| License | MIT (base DeBERTa-v3-large, MIT) |
| Parameters | 435M, fp32 (1.74 GB) |
| Contract | P (`pairs`), one row per candidate, one class |
| Context | 512 tokens per pair (trained at 384) |
| Language | English |
| Reference | `convert/ollaya_convert/families/schema_scorer/ref.py`, which imports the repo's `schema_scorer.py` |

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (5.2 MB, graph only) | derived, hosted by Ollaya (`convert/out/jev-schema-scorer-deberta-v3-large-wl/`) |
| weights | upstream `model.safetensors` (F32) @ `ee092c35…`. 394/394 tensors are external. |
| `tokenizer.json` | upstream @ `ee092c35…`, as is. **It bakes in truncation (`max_length` 384, `OnlyFirst`) and padding. The runtime must disable both**: upstream tokenizes with `truncation=False`. |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |

## Request → rows (`compile_question`)

**State text:** a string is used verbatim, anything else is `json.dumps(state, ensure_ascii=False)`.
That is the same as `ollaya_decision::serialize_state`.

**Candidates, in order:**

- **choice:**
  - `criteria` must be an object. Ollaya maps a label list to `{label: null}`, with the first
    position winning; upstream rejects lists.
  - Candidates are `(key, str(value))`. Python `str`: a string is verbatim, `null` → `None`,
    `false` → `False`, `0` → `0`, and objects and arrays use Python repr (rules in the "State text"
    section of [von.md](von.md)).
- **score:** `(str(i), str(level))` for each level.
- **noul:**
  - `("false", str(criteria.get("false", "No. The proposition is false for this state.")))`
  - `("true", str(criteria.get("true", "Yes. The proposition is true for this state.")))`
  - The default applies only when the key is **absent**. `""` stays `""`.
  - Keys are exact: `"True"` is not `"true"`.
- **Fewer than two candidates** (for example a one-option choice) is rejected by upstream. Ollaya
  answers 400.

**Schema text (sequence B):** `json.dumps(obj, ensure_ascii=False)` with default separators
(`", "`, `": "`), where `obj` is, in this key order:

```
{"candidate": {"id": <candidate id>, "description": <candidate description>},
 "type": <type>, "instructions": <instructions as given: a JSON value, not stringified>,
 "criteria": <the question's criteria as given, after the list→object mapping>}   # omitted if null/absent
```

`ollaya_decision::pyjson::dumps(value, false)` renders it. The input must be the **raw request
JSON**, not the parsed `Question`.

**Tokenization:**

- Pair encoding of `(state_text, schema_text)` with `add_special_tokens=True` gives
  `[CLS] A [SEP] B [SEP]`.
- **No truncation**: a pair longer than 512 tokens is rejected with 400, as upstream does.
- All rows go in one padded batch with `[PAD]` = 0.

**Practical limit.** Each row contains the state, the whole question schema (every option's
description) and the candidate. So the effective budget for the state shrinks with the number and
length of options. There is no truncation: an over-long pair is a 400, as upstream.

## ONNX contract (P)

| Tensor | Type | Shape |
|---|---|---|
| `input_ids` | int64 | [r, s] |
| `attention_mask` | int64 | [r, s] |
| → `scores` | float32 | [r, 1]; one logit per candidate |

Dynamic axes: r 1..4096, s 8..512. Opset 20.

## Option logits, calibration, answers

- A question's option logits are `scores[rows, 0]` in candidate order. For noul that order is
  already `[false, true]`.
- The server softmax at T = 1 equals upstream `decode_answer`: choice argmax, `score = Σ i·p_i`,
  `noul = p[1]`.
- There is no calibration upstream. The model card warns that probabilities are **very peaked**
  (it was trained on one-hot targets), so treat them as rankings.

## Measured parity

Command: `uv run python -m ollaya_convert.families.schema_scorer.parity out/jev-schema-scorer-deberta-v3-large-wl --td-limit 100`.
Log: `convert/out/logs/parity-jev-schema-scorer-deberta-v3-large-wl.log`.

- The run covers 861 questions (2,978 rows): all edge cases and 100 typed-decisions rows.
- 22 questions are rejected, exactly where upstream raises:
  - 14 one-option choices ("at least two candidates")
  - 8 pairs over 512 tokens: the six questions on the 1.8k-token `edge/long_state`, plus the 40-
    and 77-option choices. Every row repeats the full criteria JSON, so a choice with a few dozen
    described options overflows 512 on its own.
- The reference is upstream `LocalSystemOne.score_pairs`: fp32 on GPU, TF32 off, its own
  32-pair padded batches. ONNX ran as one padded batch per request on the CPU EP.

| Check | Result |
|---|---|
| Tokenizer: Rust core (`tokenizer.json`, truncation and padding off) vs reference | 2978/2978 rows identical |
| Argmax, ONNX vs fp32 reference | 100.00% |
| Max probability diff | choice 6.9e-6, score 4.6e-6, noul 1.1e-5 |
| Max logit diff | 7.4e-5 |
| Reference vs upstream `system_one` answers | 100% argmax, max diff 1.5e-7 |

- The graph is the weightless one (`-wl`, 394/394 tensors from the upstream safetensors).
- Goldens: `convert/out/goldens-jev-schema-scorer-deberta-v3-large.jsonl`. They have 97 cases,
  each with rows (`candidate`, `text_a`, `text_b`, `ids`, fp32 `score`), probabilities and
  `rejected`.

## Quality and speed

- **typed-decisions test** (400 states, 2000 questions; argmax vs majority gold): 0.444 overall.
  Choice 0.362, score 0.388, noul 0.602.
  - 147 of the 2000 questions (7%) are rejected because state + schema + candidate exceed 512
    tokens. They are counted as wrong here. On the 1853 that fit, it is about 0.48.
  - For comparison: NLI DeBERTa 0.548, GLiClass-instruct-large 0.477, von 0.447, `laya:en` 0.361.
- **Model card:** trained on 17k synthetic questions across 54 domains. The v2 eval reports choice
  accuracy 0.841, in-distribution and synthetic. The author calls out near-chance spots: next
  action on order records, summary faithfulness error type, and student-answer grading.
- **Probabilities are very peaked.** It was trained on one-hot grouped softmax, so treat the
  output as a ranking, not as calibrated confidence.
- **Cost.** One row per candidate, and every row repeats the whole schema. A 5-question request is
  18 rows × 260 tokens. On the shared 4090 and CPU (indicative only):
  - PyTorch GPU: 129 ms fp32, 60 ms bf16
  - ONNX Runtime CPU (8 threads): 7.7 s

  That makes it the most expensive of the converted models per request.
- English only. Low adoption (about 400 downloads at conversion time).

## Reproduce

```sh
cd convert
uv run python -m ollaya_convert.families.schema_scorer.export --out out/jev-schema-scorer-deberta-v3-large
uv run python -m ollaya_convert.families.weightless_ext out/jev-schema-scorer-deberta-v3-large --prefix model. \
    --out out/jev-schema-scorer-deberta-v3-large-wl \
    --checkpoint ~/.cache/huggingface/hub/models--mobarmg--jev-schema-scorer-deberta-v3-large/snapshots/ee092c35cc4ba0bd81be06a351b8788a47668d8c/model.safetensors
uv run python -m ollaya_convert.families.schema_scorer.parity out/jev-schema-scorer-deberta-v3-large-wl --td-limit 100
uv run python -m ollaya_convert.families.schema_scorer.goldens
```

## Attribution

Schema-conditioned candidate scorer © mobarmg, MIT. DeBERTa-v3 © Microsoft, MIT.
