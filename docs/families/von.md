# von (`von-option-marker-v1`)

Von 1.1 (`wfzyx/von`) is a ModernBERT-large encoder with an option-marker scoring head. Every
option gets a `[MASK]` token in one sequence and is scored from that token's hidden state. It is
**not** Laya-compatible:

- The sequence order is different (instructions, state, `[SEP]`, then options).
- The state is rendered as Python `str()`, not JSON.
- There is no type embedding.
- The option text is the description alone.
- A zero-shot noul debias needs a second row.
- The temperature depends on the input.

The existing Laya export does not apply. This document is everything the Rust port needs.

| | |
|---|---|
| Upstream | [huggingface.co/wfzyx/von](https://huggingface.co/wfzyx/von) @ `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`; code: `von-sdk==1.1.1` (PyPI), identical to [github.com/wfzyx/von](https://github.com/wfzyx/von) @ `657f42f` for this checkpoint |
| License | Apache-2.0 (Victor Hugo Panisa). Base model: ModernBERT-large, Apache-2.0 (Answer.AI / LightOn) |
| Size | 395.3M parameters (ModernBERT-large encoder 394.8M + scorer 0.53M), fp32, 1.58 GB |
| Contract | M (`markers`), one row per sequence. The graph ignores `qtype`. |
| Context | `max_len` 8192 tokens |
| Language | English |
| Reference | `convert/ollaya_convert/families/von/ref.py` (calls `von-sdk` for state rendering, sequence text and the network) |

## Files (nothing re-hosted, with one caveat)

| Layer | Source |
|---|---|
| `model.onnx` (2.9 MB, graph only) | derived, hosted by Ollaya (`convert/out/von-wl/model.onnx`) |
| weights | **`wfzyx/von` @ `d8bb5e07…` / `option_marker.pt`** (1,581,316,050 B, sha256 `52202f17…e0f2`), referenced in place by byte offset |
| `tokenizer.json` | `wfzyx/von` @ `d8bb5e07…` / `tokenizer.json`, used as is (see [Tokenization](#tokenization)) |
| `decision.json` | derived, hosted by Ollaya |
| `calibration.json` | derived, hosted by Ollaya. It transcribes upstream `marker_calibration.json`. |

**Caveat: the weights are a PyTorch pickle, not safetensors.**

- Von's trained weights (encoder and scorer) exist only in `option_marker.pt`. The repo's
  `model.safetensors` is a different, older backbone: all 170 encoder tensors differ from the
  trained ones by up to 5e-3. Referencing it would be a different model.
- `option_marker.pt` is a `torch.save` zip archive. Each tensor storage is an uncompressed
  (`ZIP_STORED`) member, so the raw little-endian fp32 bytes sit at fixed file offsets, just as in
  safetensors. The `-wl` graph points its external data straight at those offsets.
- `convert/ollaya_convert/families/weightless_ext.py` maps them with a restricted unpickler that
  reads metadata only. The result:
  - 178 of 178 checkpoint tensors are external (114 of them through a Transpose).
  - Only 24 tiny constants stay inline (shape scalars, two 32-entry RoPE frequency tables).
  - No pickle is executed at runtime; ONNX Runtime only reads byte ranges.
- **This goes beyond the current "only safetensors can be referenced" rule, so it is flagged
  here.** The alternatives:
  1. Accept offset references into torch-zip checkpoints.
  2. Ask the author to publish `option_marker.safetensors`.
  3. Leave von out.

  Re-saving the weights ourselves would mean re-hosting them.

## Request → rows

### State text (`von.backends.option_marker_backend._format_state`)

- **String:** used verbatim.
- **Object:** one line per top-level key, joined with `"\n"`. Each line is `f"{key}: {value}"`,
  where `value` is Python `str(value)`:
  - A top-level string value is verbatim, with no quotes.
  - A top-level number, bool or null is `str()`: `True`, `False`, `None`, `19.99`, `1`.
  - A nested object or array is Python `repr`: `{'k': 'v', 'n': 1, 'b': True, 'x': None}`,
    `['a', 'b']`.
- **Anything else** (array, number, bool, null at the top level): Python `str(state)`, i.e.
  `repr` for arrays.

Python `repr` rules the port must reproduce (CPython 3.12):

- **Object:** `{` + `", "`-joined `repr(k) + ": " + repr(v)` + `}`, in key order.
- **Array:** `[` + `", "`-joined `repr(x)` + `]`.
- **Numbers and constants:**
  - An integer prints as decimal digits.
  - A float prints as `float.__repr__`, the shortest round-trip form: `32.5`, `1e+16`, `1e-05`,
    `0.0001`. This is the same as the float formatting `pyjson.rs` already does for `json.dumps`.
  - `true` / `false` / `null` print as `True` / `False` / `None`.
- **Strings.** CPython `unicode_repr`:
  - Quote choice: `'…'`, unless the string contains `'` and no `"`, in which case `"…"`.
  - Always escaped: `\` → `\\`, and the chosen quote → `\'` or `\"`.
  - `\t`, `\n`, `\r` are written as those escapes.
  - Other characters with `Py_UNICODE_ISPRINTABLE` false are escaped:
    - ≤ 0xff → `\xhh`
    - ≤ 0xffff → `\uhhhh`
    - otherwise → `\Uhhhhhhhh`

    The non-printable set is general categories Cc, Cf, Cs, Co, Cn, Zl, Zp and Zs, except U+0020.
    Use Unicode 15.0 tables to match Python 3.12. For example, U+00A0 becomes `\xa0` and a ZWJ
    inside an emoji sequence becomes `\u200d`.
  - Printable non-ASCII characters are kept as they are.

The goldens carry `state_text` for every case. For example, `edge/conversation` renders as
`[{'role': 'user', 'content': 'My order never arrived.'}, …, {'role': 'user', 'content': "It's 88213. …"}]`.

### Question → instructions and option descriptions

Start from the shared parser's `Question`. Its instructions are already strings; non-string
instructions become `json.dumps` there. Criterion values are raw JSON.

- **choice.** Labels are the criteria keys, in request order. A list of labels means
  `{label: null}`, deduplicated, first position wins. The description of each option is:
  - the label with Python `strip()` applied, if the value is falsy (`null`, `""`, `0`, `false`,
    `[]`, `{}`);
  - otherwise `text(value).strip()`, where `text(s)` is `s` for a string and Python `str()` (the
    repr rules above) for anything else.
- **score.** For each level:
  - An object: `f"{what}{ex}".strip()`.
    - `what` is `str(item.get("what", ""))`.
    - `ex` is `" Examples: " + ", ".join(examples)` when `examples` is truthy, otherwise `""`.
      `examples` is a list of strings. If it is a string, Python joins its characters.
  - Anything else: `str(item).strip()`. Strings are used as they are.
- **noul.** Descriptions are in **row order `[true, false]`**:
  - `t = text(criteria.true)` if truthy, else `"Yes, condition holds true."`
  - `f = text(criteria.false)` if truthy, else `"No, condition is false."`
  - `zero_shot = not (truthy(criteria.true) or truthy(criteria.false))`

  Keys are matched case-insensitively and the last spelling wins, as the shared parser does.

`strip()` is Python's. It removes characters with `str.isspace()`, which is Rust's
`char::is_whitespace` **plus** U+001C..U+001F.

### Sanitising

Replace every occurrence of the mask text `[MASK]` with a single space `" "`. This applies to the
state text, the instructions and every description, before packing.

This is Ollaya deviation 1:

- Upstream takes *every* `[MASK]` id in the sequence as an option marker.
- Such text therefore shifts all option scores. The shared `edge/mask_text` case shows it.
- On any input without a literal `[MASK]`, the sequence is identical to upstream.

### Row text (`OptionMarkerModel.pack_sequence`)

```python
prefix = f"{instructions} {state}".strip() if instructions else state.strip()
text   = f"{prefix} [SEP] " + " ".join(f"[MASK] {d.strip()}" for d in descriptions)
```

- `[SEP]` and `[MASK]` are literal text here. The tokenizer turns them into ids 50282 and 50284.
- The space before `[SEP]` becomes its own token `Ġ` (209).
- `[MASK]` has `lstrip`, so the space before each marker disappears.

**Zero-shot noul second row.** It is the same pack with `state = ""`, which gives
`prefix = instructions.strip()`. If the instructions are empty, the text starts with `" [SEP]"`.
This row depends only on the question, so the runtime may cache it.

Example (typed-decisions `choice`):

```
What should the observability system do with this trace? agent: {'autonomy': 'checkpointed', 'model': 'internal-agent-v1'}
constraints: [...]
... [SEP] [MASK] Let the agent proceed without interruption. [MASK] Queue this trace for a human to review. [MASK] ...
```

### Tokenization

- Call `tokenizer.encode(text, add_special_tokens=True)`. The post-processor is
  `TemplateProcessing` `[CLS] $A [SEP]`, which gives `[50281] + ids + [50282]`.
- This is identical to `encode(text, false)` with `[CLS]`/`[SEP]` added by hand.
- **Upstream `tokenizer.json` has truncation (`max_length` 512, `LongestFirst`) and `BatchLongest`
  padding baked in.** Python's `AutoTokenizer` ignores them, but `Tokenizer::from_file` in Rust
  applies them.
  - The runtime must call `with_truncation(None)` and `with_padding(None)` after loading.
  - `encode_fast` truncates too.
  - With both disabled, the Rust core reproduced 1051/1051 rows. With them left on, 9 rows
    (every row over 512 tokens) came out wrong.
- **Markers** are every position whose id is `50284` (`[MASK]`), in order. After sanitising, there
  are exactly K of them, one per description. The runtime should assert that.

### Truncation (only above `max_len` = 8192)

Upstream has no limit. Ollaya deviation 2 applies only when `len(ids) > 8192`:

1. Encode the state text alone with `add_special_tokens=false` and *character* offsets. In Rust
   that is `encode_char_offsets`; Python offsets are in Unicode code points. Let `n` be its token
   count.
2. Loop:
   - `n -= len(ids) - 8192`
   - if `n <= 0`, `kept = 0`; otherwise `kept = offsets[n-1].end`
   - rebuild the text with `state[:kept]` (code points) and re-encode
   - stop as soon as `len(ids) <= 8192`
3. If `n <= 0` and it still does not fit, reject the question: the instructions and options alone
   exceed the context.

Golden `von/over_max_len` covers this. It is an 11,900-token row cut to exactly 8192.

### Row batching

- One row per question, plus the empty-state row for zero-shot noul questions.
- All rows of a request go in one batch:
  - pad to the longest row with `[PAD]` (50283), `attention_mask` 0
  - pad `marker_pos` with 0 and `marker_mask` with false to the widest row
  - `min_markers` is 1
- `qtype` is fed as choice 0, score 1, noul 2. The graph ignores it.

## ONNX contract (M)

| Tensor | Type | Shape |
|---|---|---|
| `input_ids` | int64 | [n, s] |
| `attention_mask` | int64 | [n, s] |
| `marker_pos` | int64 | [n, k] |
| `marker_mask` | bool | [n, k] |
| `qtype` | int64 | [n] |
| → `logits` | float32 | [n, k]; masked slots are −1e4 |

- The graph is the encoder, a gather at `marker_pos`, and `OptionMarkerScorer` (LayerNorm, Linear
  1024→512, GELU, LayerNorm, Linear 512→1).
- Dynamic axes: n 1..1024, s 8..8192, k 1..255. Opset 20.
- Batched padded rows reproduce upstream's unpadded batch-1 runs (see Parity).

## Option logits (runner → server)

Per question, in Ollaya option order:

- **choice / score:** row 0's K logits, in option order.
- **noul:** row 0 gives `[l_t, l_f]`, in row order true, false.
  - With criteria: `[l_f, l_t]`.
  - Zero-shot (row 1 gives `[n_t, n_f]`): `[l_f, l_t − c]`, with `c = a·(n_t − n_f) + b`.
    `decision.json` `noul.zero_shot_prior` supplies `a = 0.7, b = 0`, upstream's constant. HEAD
    reads an optional fitted prior from `marker_calibration.json`, and this checkpoint has none.

## Calibration (server)

`calibration.json` → `temperature_map` (`kind: von-entropy-length-v1`) is upstream
`_effective_temperature`. It is computed per question on the option logits `z` from above
(K = 2 for noul):

```
p      = softmax(z)
H_norm = (−Σ p·ln max(p, 1e-12)) / ln K        (0 when K = 1)
T      = bias + entropy·H_norm + log_tokens·log10(state_tokens)/4 + n_options·K/8
T      = clamp(T, lo, hi)                      # lo 0.3, hi 12.0 — NOT Laya's [0.5, 5] clamp
probs  = softmax(z / max(T, 1e-4))
```

- `state_tokens` is `max(1, len(encode(state_text, add_special_tokens=false)))` on the state
  text **before** sanitising and truncation.
- The coefficients are `bias` 0.2056, `entropy` −3.255, `log_tokens` 20.2391, `n_options` −5.2263.
- T is monotone, so it never changes the argmax.
- If a runtime does not implement the map, `temperature` [2.2, 2.2, 2.2] is upstream's own
  fallback when the map is invalid.
- The Rust `Calibration` type must not apply `TEMP_MIN`/`TEMP_MAX` to this family.

Upstream answer rendering, for reference:

- choice: argmax label.
- noul: `P(true)`.
- score: `Σ i·p_i`.
- confidence: `p1 − p2`. Ollaya's server applies its own Jev confidence instead.

## Limits

- Context: 8192 tokens. Instructions and options must fit with at least an empty state.
- Options: no upstream limit. Each costs at least 2 tokens. Ollaya's shared limits apply
  (choice ≤ 255, score 2–10).
- Inputs von's own schema returns 422 for are accepted (Ollaya deviation 3, listed in `ref.py`):
  - non-string instructions
  - list choice criteria
  - non-string criterion values
  - noul keys with other capitalisation

## Measured parity

`uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.parity out/von-wl --td-limit 100`
(`convert/out/logs/parity-von-wl.log`).

The run covers 883 questions (1051 rows): all edge cases and 100 typed-decisions rows. The
reference is fp32 on GPU with TF32 off. ONNX ran on the CPU EP.

| Check | Result |
|---|---|
| Tokenizer: Rust core vs Python | 1051/1051 rows identical (truncation and padding off) |
| Argmax, ONNX vs fp32 reference | 100.00% |
| Max probability diff (calibrated) | choice 6.5e-6, score 1.4e-5, noul 2.3e-5 |
| Max row-logit diff | 8.7e-4 (p99 3.2e-4) |
| Reference vs `von`'s own `evaluate()` (828 questions upstream accepts) | 100% argmax, max diff 5.0e-5 (von rounds to 4 dp) |

- The weightless graph (`out/von-wl`) and the self-contained export (`out/von`) give identical
  numbers.
- Goldens: `convert/out/goldens-von.jsonl` has 98 cases: every edge case, 20 typed-decisions
  rows, and `von/over_max_len`. Each case carries row text, ids, markers, row logits, option
  logits, T, probabilities, and upstream `answers` where upstream accepts the request.

## Quality and speed

- **JevBench (author's numbers, official harness):** easy 0.938, standard 0.653, hard 0.351.
  - The calibration map was fitted on the same 231 public JevBench items, so the calibration
    numbers are in-sample.
  - The model card calls it weak on long multi-hop policy text.
- **typed-decisions test** (400 states, 2000 questions; argmax vs majority gold label; measured
  here with the reference): 0.447 overall. Choice 0.367, score 0.367, noul 0.632. For comparison:
  - `laya:en`: 0.361
  - GLiClass-instruct-large: 0.477 ([gliclass.md](gliclass.md))
  - `MoritzLaurer/deberta-v3-large-zeroshot-v2.0` with the nli-pairs templates: 0.548
    ([nli.md](nli.md))

  Gold labels come from low-agreement annotators, so treat these as relative numbers.
- **Latency.** One 5-question typed-decisions request (5 rows × 161 tokens). The RTX 4090 and CPU
  were shared with other jobs at 100% GPU utilisation, so these numbers are indicative only:
  - upstream `evaluate()`, fp32, sequential batch-1 rows: 191 ms
  - batched graph in PyTorch: 37 ms fp32, 38 ms bf16. An earlier run under heavier contention
    measured 78 and 46 ms.
  - ONNX Runtime CPU (8 threads, `-wl` graph): 0.98 s
- English only.
- Near 8k tokens, quality visibly degrades, and the temperature map saturates at T = 12.

## Von 1.2 (released during this conversion; not converted)

`wfzyx/von` `main` moved to **von 1.2.0** (commit `c2cb39f7`, 2026-09-24 00:59 UTC,
"independent_options"). This document and the export target **1.1**, pinned at `d8bb5e07`. Those
weights stay fetchable by commit, and `ref.py` loads that snapshot explicitly.

1.2 is a new layout, not a new checkpoint for this one. Its code is in github.com/wfzyx/von
`e189466`, and it is not on PyPI yet (latest `von-sdk` is 1.1.1).

**What stays the same:** the text packing, tokenization, row structure and option-logit
derivation.

**Attention.** Each option attends only to the prefix (everything before the first `[MASK]`) and to
its own span. The prefix attends only to the prefix. The trailing `[SEP]` belongs to no option.

**Positions.** Every option span's `position_ids` restart at the prefix length. The
sliding-window layers use `|pos_i − pos_j| ≤ sliding_window` on those position ids, not on the raw
index. See `build_independent_option_masks` and `build_option_invariant_position_ids`.

A `von-option-marker-v2` graph can compute both from `marker_pos`, `marker_mask` and
`attention_mask`, the same way the GLiClass graph derives label spans. It then feeds ModernBERT a
per-layer-type 4D mask dict plus `position_ids`. That needs a fresh export and parity run.

**Calibration.** The coefficients change (`bias` 2.0151, `entropy` −3.2369, `log_tokens`
11.4916, `n_options` −3.5574), and a fitted `noul_zero_shot_prior` appears (`a` −0.5, `b` 0.3).
Both fit the existing `calibration.json` and `decision.json` fields.

**Weights.** `option_marker.pt` is still a pickle (sha256 `3faf27f8…`), so the same torch-zip
offset caveat applies.

**Hazard for anyone on `von-sdk==1.1.1` today.** It downloads the unpinned `main`, so it now loads
1.2 weights and runs them with 1.1's full attention. That silently changes answers. Ollaya must
pin commits (it does).

## Reproduce

```sh
cd convert
uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.export --out out/von
uv run python -m ollaya_convert.families.weightless_ext out/von --out out/von-wl \
    --checkpoint ~/.cache/huggingface/hub/models--wfzyx--von/snapshots/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd/option_marker.pt
uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.parity out/von-wl --td-limit 100
uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.goldens --out out/goldens-von.jsonl
```

## Attribution

Von © Victor Hugo Panisa, Apache-2.0 (`@software{von2026, …}`). ModernBERT © Answer.AI and
LightOn, Apache-2.0.
