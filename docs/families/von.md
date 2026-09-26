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

The existing Laya export does not apply. This document is the spec the Rust port implements.

**Status: implemented, library entry prepared** as `von:1.1` (also `von:latest`), issue #5. The
runtime is `ollaya_decision::von` (rows), `ollaya_decision::pyrepr` (Python `str()`),
`ollaya_decision::calibration::TemperatureMap` (the input-conditioned temperature) and
`ollaya_runner::von` (the engine). Measured parity and speed are in
[Rust runtime parity](#rust-runtime-parity) and [Quality and speed](#quality-and-speed): on x86-64
CPU and CUDA the runtime is within 4.4e-4 of the goldens' logits and makes the same decision on every
question. On Apple silicon's CPU one row is 1.1e-3 off, and every decision is still the same. The goldens are the network in float64; [why](#why-the-goldens-are-fp64).

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
- **This goes beyond the earlier "only safetensors can be referenced" rule.** The alternatives were:
  1. Accept offset references into torch-zip checkpoints.
  2. Ask the author to publish `option_marker.safetensors`.
  3. Leave von out.

  Re-saving the weights ourselves would mean re-hosting them. **Decision (owner, issue #5):
  option 1.** The manifest's weights layer is the author's file, pinned and verified:
  `https://huggingface.co/wfzyx/von/resolve/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd/option_marker.pt`,
  sha256 `52202f176080d1efe38ca62c45264894fc7e444d332f4af6bac86f4e3146e0f2` (the HF LFS oid),
  1,581,316,050 bytes. `package.py` renames the graph's external-data location from
  `option_marker.pt` to the blob name `sha256-52202f17…`, so the runner opens it straight from
  the blob store, as for every other family.

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
      `examples` is a list of strings. If it is a string, Python joins its characters; a dict
      joins its keys. Where Python raises (a list with non-strings, a number, `true`), the Rust
      port renders each non-string with `str()` instead (part of deviation 3).
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
- All rows of a request are batched together:
  - pad to the longest row with `[PAD]` (50283), `attention_mask` 0, and to at least 8 tokens
  - pad `marker_pos` with 0 and `marker_mask` with false to the widest row
  - `min_markers` is 1
- The Rust runner sorts the rows by length and splits them into batches of at most 8192 padded
  tokens and 1024 rows (see Limits, memory). In a spot check on the CPU provider (one request,
  6 rows), batched and one-row-at-a-time runs gave the same logit errors against the goldens, at
  1, 8 and the default number of threads.
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
T      = clamp(T, lo, hi)                      # lo 0.3, hi 12.0, NOT Laya's [0.5, 5] clamp
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
  - non-string score `examples` (upstream's `", ".join` raises; the port uses `str()`)
- A question whose instructions and options alone exceed 8192 tokens is rejected with a 400 that
  names it.
- Memory. The graph materialises full attention scores ([rows, 16 heads, seq, seq] in fp32), so
  one 8192-token row needs about 25 GB of RAM on the CPU provider (measured: 24.8 GB peak RSS for
  the parity run, dominated by `von/over_max_len`). The runner therefore runs at most 8192 padded
  tokens per `session.run` (one full-length row alone).

## Measured parity

`uv run --with von-sdk==1.1.1 python -m ollaya_convert.families.von.parity out/von-wl --td-limit 100`

The run covers 883 questions (1051 rows): all edge cases and 100 typed-decisions rows. The
reference is upstream's network run in float64, one unpadded row at a time, as for the goldens
(`ref.Exact`; see [Why the goldens are fp64](#why-the-goldens-are-fp64)). ONNX ran on the CPU EP,
ONNX Runtime 1.30, one padded batch per request.

| Check | Result |
|---|---|
| Tokenizer: Rust core vs Python | 1051/1051 rows identical (truncation and padding off) |
| Argmax, ONNX vs the fp64 reference | 100.00% |
| Max probability diff (calibrated) | choice 3.8e-5 (the 77-option edge case; p99 1.4e-6), score 8.9e-6, noul 6.1e-6 |
| Max row-logit diff | 3.5e-4 (p99 per type 4.2e-5 to 2.3e-4) |
| Reference vs `von`'s own `evaluate()` (828 questions upstream accepts) | 100% argmax, max diff 7.7e-5 (von runs fp32 and rounds to 4 dp) |

- The weightless graph (`out/von-wl`) and the self-contained export (`out/von`) give identical
  numbers.
- Goldens: `convert/out/goldens-von.jsonl` has 98 cases: every edge case, 20 typed-decisions
  rows, and `von/over_max_len`. Each case carries row text, ids, markers, row logits, option
  logits, T, probabilities, and upstream `answers` where upstream accepts the request.

## Rust runtime parity

`crates/ollaya-runner/examples/parity_von.rs` against `goldens-von.jsonl` (98 cases, 485
questions, 653 rows). It checks the state text and its token count, every row's text, ids and
marker positions, then the row and option logits (tolerance 1e-3), the decisions, the calibration
on the reference's own logits (1e-6) and, where upstream accepts the request, von's `evaluate()`.

```sh
cargo run --release -p ollaya-runner --example parity_von -- convert/out/von-wl convert/out/goldens-von.jsonl cpu --latency
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_von -- convert/out/von-wl convert/out/goldens-von.jsonl cuda --latency
```

Results on choso-wsl (24 cores; RTX 4090, CUDA 13), `ort` 2.0.0-rc.13 with ONNX Runtime 1.28:

| | CPU | CUDA |
|---|---|---|
| state text and token count, row texts, ids and markers identical | 98 cases, 653 / 653 rows | same |
| max \|Δ row logit\| | 4.4e-4 (p99 2.5e-4) | 3.9e-4 (p99 1.4e-4) |
| decisions agree | **100 %** | **100 %** |
| max \|Δ probability\| (input-conditioned T) | 3.4e-5 (p99 7.9e-6) | 4.7e-5 (p99 7.6e-6) |
| calibration on the reference logits (as f32): T, probabilities | 3.6e-7, 7.8e-8 | same |
| vs von `evaluate()` (368 questions, 4-decimal rounding) | 0 differ, max 8.0e-5 | 0 differ, max 7.4e-5 |

The worst rows are `edge/many_options_77` `intent` (1444 tokens) on the CPU and
`td/agent_trace_observability_000000` `urgency` (151 tokens) on CUDA.

### Why the goldens are fp64

The first goldens were upstream's own forward in fp32 on the GPU, TF32 off. Against them the CPU
runtime was 1.2e-3 off on one row, `preset/router/conversation` `is_sensitive` (102 tokens):
`[1.200809, -0.177873]` against a golden `[1.199640, -0.177809]`, with identical ids, markers and
masks. To place that difference, the 651 golden rows below 8192 tokens were run through every path
available, one unpadded row at a time unless noted, and compared with the **exact** value, the same
network in float64:

| Path (fp32 unless noted) | vs the fp32 goldens: max (rows > 1e-3) | vs exact: max (rows > 1e-3) |
|---|---|---|
| fp32 goldens: PyTorch CUDA, SDPA memory-efficient kernel | 0 | **1.05e-3 (1)** |
| PyTorch CUDA, SDPA math kernel | 8.7e-4 (0) | 7.2e-4 (0) |
| PyTorch CPU | 2.5e-3 (1) | 2.6e-3 (1) |
| ONNX Runtime 1.30 (Python), all optimizations | 8.7e-4 (0) | 3.5e-4 (0) |
| ONNX Runtime 1.30 (Python), optimizations off | 1.8e-3 (2) | 1.6e-3 (1) |
| Rust runtime, CPU (ORT 1.28, batched) | 1.2e-3 (1) | **4.4e-4 (0)** |
| Rust runtime, CUDA (batched) | 9.6e-4 (0) | **3.9e-4 (0)** |
| the exact value | **1.05e-3 (1)** | 0 |

- **The fp32 goldens were 1.05e-3 from the exact value on that row.** They reproduce bit for bit
  with PyTorch's CUDA SDPA memory-efficient attention kernel, which PyTorch picks for fp32
  attention on the GPU (TF32 off does not change that). On the same GPU the math kernel was
  1.9e-4 from exact on this row, and the runtime 1.2e-4. The exact value itself failed the 1e-3
  gate against that golden, so no accurate implementation could pass it.
- **Von's logits are sensitive to rounding.** On `td/agent_trace_observability_000000` `urgency`
  PyTorch fp32 on the CPU is 2.6e-3 from exact. Per-layer hidden states (fp32 vs exact) show the
  relative error at the option markers growing from 6e-7 after layer 7 to 3.7e-4 after layer 28,
  against 2e-6 on an ordinary row (`td/agent_trace_observability_000013`). The scorer head adds
  less than 1e-6 on exact inputs: the logit error is the first-order image of the encoder's final
  hidden-state error (the fp64 gradient predicts it to 1e-6). The head turns a unit of hidden state
  into up to about one logit unit and the final states have a norm near 37, so a relative error of
  1e-5 in the encoder already means about 4e-4 in a logit. So von's logit differences run larger
  than those of other ModernBERT graphs (`nli:modernbert-large`: 1.5e-4), and an fp32 reference
  is not a fixed point to test against.
- **So the goldens are the network in float64** (owner's decision, issue #5; the 1e-3 gate is
  unchanged). `ref.Exact` runs upstream's own `OptionMarkerModel` (same code, same weights)
  converted to float64. transformers' ModernBERT builds the rotary tables and applies them to q and
  k in fp32 whatever the model's dtype, so `ref.fp64_rotary` keeps that step in float64 too; the
  inverse frequencies stay the model's fp32 buffer, as in the ONNX graph. With it, fp64 on the GPU
  and on the CPU agree to every printed digit. Rows up to 4096 tokens run on the GPU; the two
  8192-token rows of `von/over_max_len` run on the CPU (fp64 attention at 8192 tokens would need
  about 25 GB of GPU memory; on the CPU the whole run peaks at 6 GB).
- **Old vs new goldens.** Every id, state, question, state text and token count, row text, ids and
  markers (so every attention mask) and every upstream `answers` entry is identical; only the
  reference numbers moved. Row logits moved by at most 1.05e-3 (the row above; p99 1.5e-4,
  median 8.6e-6), option logits by at most 1.05e-3, temperatures by at most 3.5e-4 and
  probabilities by at most 4.5e-5. No decision changed. The fp32 goldens are kept next to the new
  ones as `convert/out/goldens-von.fp32-sdpa.jsonl`, and `--precision fp32` still produces them.


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
- **Latency of the Rust runtime** (choso-wsl, RTX 4090 and 24 cores):
  - `parity_von --latency` on CUDA, in process: 5-question golden requests (7.1 rows, 559 tokens
    on average) take 23.4 ms at the median (p95 28.5 ms).
  - Through the daemon (`/api/decide`, model loaded), the five `triage` questions about a short
    message: 23.3 ms at the median on CUDA, 0.76 s on the CPU.
  - Longer states, same five questions (median of 3), CUDA / CPU: 500 words (2,949 input tokens)
    0.17 s / 5.6 s; 1,000 words (5,629) 0.41 s / 13.3 s; 2,000 words (10,984) 0.78 s / 24.7 s;
    4,000 words (21,699) 2.6 s on CUDA; 7,000 words (37,774) 7.4 s on CUDA, with 18.7 GB of the
    card in use.
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

Both default to the float64 reference (`--precision fp64`). The goldens need a CUDA GPU for the
rows up to 4096 tokens and run the two 8192-token rows on the CPU (`--long-device`); on choso-wsl
(RTX 4090, 24 cores) the whole file takes 2.5 minutes.

## Attribution

Von © Victor Hugo Panisa, Apache-2.0 (`@software{von2026, …}`). ModernBERT © Answer.AI and
LightOn, Apache-2.0.
