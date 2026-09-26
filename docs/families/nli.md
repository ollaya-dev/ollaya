# nli (`nli-pairs-v1`)

Zero-shot NLI cross-encoders from Moritz Laurer's zeroshot-v2.0 series. Every
(premise, hypothesis) pair is one encoder row, and the model says whether the premise entails the
hypothesis. Typed questions become hypotheses through per-model templates, which are stored in
`decision.json` → `templates`. The call and scoring are exactly `transformers`'
`pipeline("zero-shot-classification")`, which is the documented usage for these models.

| | `deberta-v3-large-zeroshot-v2.0` | `ModernBERT-large-zeroshot-v2.0` |
|---|---|---|
| Upstream | [MoritzLaurer/deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) @ `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2` | [MoritzLaurer/ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) @ `a51e07b524299e309dd2b88d48b0cfa2bd9ec598` |
| License | **MIT** (base DeBERTa-v3-large, MIT) | **Apache-2.0** (base ModernBERT-large, Apache-2.0) |
| Parameters | 435.1M; checkpoint fp16 (870 MB) | 395.8M; checkpoint bf16 (792 MB) |
| Classes (`config.json` id2label) | 0 = `entailment`, 1 = `not_entailment` | 0 = `entailment`, 1 = `not_entailment` |
| `[CLS]` / `[SEP]` / `[PAD]` | 1 / 2 / 0 | 50281 / 50282 / 50283 |
| Context per pair | 512 (`tokenizer.model_max_length`) | 512 (`tokenizer.model_max_length`; the architecture allows 8k, but it was not trained on long inputs) |

- Contract: P (`pairs`), one row per hypothesis.
- Language: English. For other languages the series has `bge-m3-zeroshot-v2.0`.
- Reference: `convert/ollaya_convert/families/nli/ref.py`, which runs the real pipeline on the same
  fp32 module.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (5.2 MB DeBERTa, 3.5 MB ModernBERT; graph only) | derived, hosted by Ollaya (`convert/out/{deberta,modernbert}-large-zeroshot-v2.0-wl/`) |
| weights | upstream `model.safetensors` @ the commit above, referenced by offset. All tensors are external: DeBERTa 394/394 (F16 → Cast FLOAT), ModernBERT 174/174 (BF16 → Cast FLOAT). The BF16 path needs `families/weightless_ext.py`, because `weightless.py` keeps BF16 inline. |
| `tokenizer.json` | upstream `tokenizer.json` @ the commit above, as is. **DeBERTa-v3 needs no conversion**: the repo already ships a fast `tokenizer.json` (Unigram / Metaspace, sentencepiece `Precompiled` normalizer), and the Rust `tokenizers` crate loads it. Neither file bakes in truncation or padding. |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |

**DeBERTa tokenizer: the file is the reference, not transformers 5.**

- Under transformers 5, `AutoTokenizer` rebuilds DeBERTa-v2/v3's normalizer in Python
  (`Replace(\s{2,}|[\n\r\t] → " ")`, NFC, right strip) instead of the file's `Strip` +
  `Precompiled` charsmap. Characters the charsmap would fold then tokenize differently: for example
  `，` becomes `[UNK]` instead of `,`. In typed-decisions the affected characters are `‑`
  (U+2011), U+202F, `’`, `–`, `—` and `…`, which touch 70 of the 400 states.
- The checkpoint was trained under transformers 4.37.2, which loaded the file verbatim.
- So the reference, and the pipeline check built on it, tokenizes with the file
  (`families/repo_tokenizer.py`). The Rust runtime uses the same file, so no derived tokenizer is
  hosted.
- ModernBERT is unaffected: its file and transformers 5 agree.

## Request → rows

**Premise.** The state as Ollaya renders it everywhere: a string is used verbatim, anything else is
`json.dumps(ensure_ascii=False)` (`ollaya_decision::serialize_state`).

**Hypotheses.** Built from the shared parser's `Question`:

- Instructions are strings; anything else becomes `json.dumps`.
- A list of choice labels means `{label: null}`.
- noul keys are lower-cased.

Placeholders `{name}` are substituted in **one left-to-right pass** (`re.sub(r"\{(\w+)\}")`), so
`{…}` inside a value is never expanded again.

| type | rows (in option order) | fields |
|---|---|---|
| choice | one per option | `instructions`; `label`; `description` = `""` if the value is null or `""`, else `render_criterion(value)`; `option` = `label` or `label: description`. Uses template `choice.with_description` if the description is non-empty, else `choice.without_description`. |
| score | one per level | `instructions`; `level` = index; `description` = `render_criterion(level)` |
| noul, **both** `false` and `true` non-empty ("pair") | two: `[false, true]` | template `noul_pair` with `description` = `render_criterion(false)` / `render_criterion(true)` |
| noul otherwise ("single") | one | template `noul_single` with `instructions` |

Templates:

| key | DeBERTa-v3-large | ModernBERT-large |
|---|---|---|
| `choice.with_description` | `The answer to "{instructions}" is: {option}` | `{description}` |
| `choice.without_description` | `The answer to "{instructions}" is: {option}` | `This text is about {label}.` |
| `score` | `The answer to "{instructions}" is: {description}` | `{description}` |
| `noul_pair` | `{description}` | `{description}` |
| `noul_single` | `{instructions}` | `{instructions}` |

- ModernBERT's `without_description` is Laurer's canonical `hypothesis_template`, "This text is about {}".
- TypeSafe noul instructions are statements ("This trace requires human review."), which is exactly
  what an NLI hypothesis is.
- A noul phrased as a question still runs, but less well.

**Tokenization** (per pair, as `ZeroShotClassificationPipeline._parse_and_tokenize`):

- Encode the pair `(premise, hypothesis)` with `add_special_tokens=True`. The post-processor gives
  `[CLS] premise [SEP] hypothesis [SEP]`.
- Truncation: `only_first`, from the right, to 512. Padding off.
- In Rust: `encode((premise, hypothesis), true)` with
  `TruncationParams { max_length: 512, strategy: OnlyFirst, direction: Right, stride: 0 }`.
- Equivalently: `[CLS] + P[..512-3-len(H)] + [SEP] + H + [SEP]`, where `P` and `H` are
  `encode(·, false)`.
- If `len(H) + 3 > 512`, reject the question with 400. Upstream falls back to no truncation in
  that case.
- `token_type_ids` are not needed. DeBERTa-v3 here has `type_vocab_size` 0 and ModernBERT has
  none.

**Batching.** All rows of all questions go in one batch, padded with `[PAD]` and
`attention_mask` 0. A request costs Σ options rows (a noul costs 1 or 2).

## ONNX contract (P)

| Tensor | Type | Shape |
|---|---|---|
| `input_ids` | int64 | [r, s] |
| `attention_mask` | int64 | [r, s] |
| → `scores` | float32 | [r, 2] raw logits `[entailment, not_entailment]` |

Dynamic axes: r 1..4096, s 8..512. Opset 20.

## Option logits (runner → server)

With `E = scores[:, 0]` (entailment) and `N = scores[:, 1]` (not_entailment):

- **choice / score / pair noul:** `E` of the question's rows, in row order. For noul that order
  is `[false, true]`. The server softmax (T = 1) is the pipeline's `multi_label=False`.
- **single noul:** `[N₀, E₀]`. The softmax is the pipeline's single-candidate / `multi_label=True`
  result: `P(true) = softmax([N, E])[1]`.

There is no calibration upstream, so `calibration.json` is `[1, 1, 1]`. **Zero-shot NLI scores are
not calibrated.** Softmax across entailment logits spreads mass by how "entailable" each
hypothesis reads, not by probability of correctness.

## Template choice

Measured on typed-decisions test (400 states, 2000 questions; argmax vs majority gold). The
chosen variant is in bold.

| variant | DeBERTa | ModernBERT |
|---|---|---|
| choice: `{instructions} {option}` | 0.555 | 0.425 |
| choice: `The answer to "{instructions}" is: {option}` | **0.580** | 0.393 |
| choice: `{description}` / `This text is about {label}.` | 0.508 | **0.460** |
| score: `{instructions} {description}` | 0.410 | 0.475 |
| score: `The answer to "{instructions}" is: {description}` | **0.443** | 0.456 |
| score: `{description}` | 0.421 | **0.479** |
| noul with both descriptions: single `{instructions}` | 0.542 | 0.515 |
| noul with both descriptions: pair `[false, true]` descriptions | **0.640** | **0.605** |
| noul without descriptions: single `{instructions}` | **0.705** | **0.650** |

The templates are selected in-sample on this set, so they are slightly optimistic. The two models
disagree on choice and score, which is why templates are per model. This sweep ran with
transformers-5 tokenization. The final numbers under [Quality](#quality-and-speed) use the file
tokenizer and differ by up to 0.003.

## Limits

- 512 tokens per pair, and the premise is cut first.
- Rows scale with the option count: a 77-option choice is 77 rows. Latency grows about linearly with
  Σ options.
- Ollaya's shared limits apply (choice ≤ 255, score 2–10).

## Measured parity

Command: `uv run python -m ollaya_convert.families.nli.parity {deberta-v3-large|modernbert-large} out/<slug>-wl --td-limit 100`.
Logs: `convert/out/logs/parity-{deberta,modernbert}-large-zeroshot-v2.0-wl.log`.

- The runs cover 883 questions (2,953 pair rows): all edge cases and 100 typed-decisions rows.
- The reference is fp32 on GPU with TF32 off, one unpadded pair at a time, as the pipeline runs
  them. ONNX ran as one padded batch per request on the CPU EP.

| Check | DeBERTa-v3-large | ModernBERT-large |
|---|---|---|
| Tokenizer: Rust core (`tokenizer.json`, `OnlyFirst`@512) vs reference | 2953/2953 | 2953/2953 |
| Manual rule `[CLS] P[..budget] [SEP] H [SEP]` vs reference | 2953/2953 | 2953/2953 |
| Argmax, ONNX vs fp32 reference | 100.00% | 100.00% |
| Max probability diff | 5.4e-6 | 1.0e-5 |
| Max entailment-logit diff | 4.1e-5 | 1.5e-4 |
| Reference vs `pipeline("zero-shot-classification")` (882 questions) | 100% argmax | 100% argmax |

- Probabilities match the pipeline to ≤ 1e-7, **except one-option choices** (14 edge-case
  questions).
  - For a lone candidate the pipeline switches to multi-label: `P = softmax([not_entail, entail])`.
  - Ollaya answers any one-option choice with 1.0, as every layout does.
  - The logs predate excluding these from the comparison, which is why they show a max diff of
    about 1.0. `ref.upstream_probabilities` now skips them.
- Both graphs are weightless: they reference the upstream fp16 / bf16 safetensors through Cast
  nodes.
- Goldens: `convert/out/goldens-{deberta,modernbert}-large-zeroshot-v2.0.jsonl`. They have 97
  cases, each with the premise, rows (`hypothesis`, `ids`, fp32 `scores`), option logits,
  probabilities and `rejected`.

## Metal parity (MLX)

The MLX engine ([decision record](../decisions/0001-mlx-engine.md)) runs `modernbert-large` on
the Apple GPU from the upstream `model.safetensors`, with the `sequence-classification` arch layer.
It is built with the `mlx` feature, which is not in releases yet. `deberta-v3-large` stays on ONNX
Runtime until the DeBERTa backbone (phase 3).

```sh
cargo run --release -p ollaya-runner --features mlx --example parity_nli -- <model-dir> convert/out/goldens-modernbert-large-zeroshot-v2.0.jsonl metal
```

Apple M4 Pro, macOS 27, against the same goldens and the same gate as the CPU (row scores and
option logits within 1e-3, same decisions):

| | MLX on Metal | ONNX Runtime CPU, same Mac |
|---|---|---|
| hypotheses and token ids identical | 483 questions, 1,513 rows | same |
| max \|Δ row score\|, \|Δ option logit\| | 7.2e-4, 7.0e-4 | 4.1e-4, 4.1e-4 |
| decisions agree | **100 %** | **100 %** |
| max \|Δ probability\| | 5.9e-5 (p99 1.8e-5) | 5.2e-5 (p99 1.4e-5) |
| through the daemon, p50 / p95 per request | 140 / 429 ms | 312 / 941 ms |

The daemon timings are the golden requests (4.7 questions and about 15 rows each) sent twice, with other
jobs running on the machine.

## Quality and speed

- **typed-decisions test** (400 states, 2000 questions; argmax vs majority gold; measured here
  with the chosen templates):

  | model | overall | choice | score | noul |
  |---|---|---|---|---|
  | deberta-v3-large-zeroshot-v2.0 | **0.548** | 0.580 | 0.440 | 0.662 |
  | ModernBERT-large-zeroshot-v2.0 | 0.515 | 0.460 | 0.479 | 0.620 |

  For comparison: von 0.447, GLiClass-instruct-large 0.477, `laya:en` 0.361. The templates were
  picked on this set, so the numbers are optimistic. The gold labels have low agreement.
- **Zero-shot NLI is uncalibrated.** There is no temperature upstream. Probabilities come from
  softmaxing entailment logits across independently scored hypotheses. They rank well but say
  little about confidence. A calibration layer fitted per model and question type would be the
  obvious next step, and it only needs a new `calibration.json`.
- **Model cards:**
  - DeBERTa: mean F1-macro 0.673 over 28 classification sets (0.676 for the `-c` variant).
  - ModernBERT: accuracy 0.85 on a mix that includes its own training datasets, so that number
    is not zero-shot. The author says it is "slightly worse than DeBERTa-v3 on average".
- **Cost scales with options.** A 5-question typed-decisions request is 18 rows; a 77-option choice
  is 77 rows. The GPU and CPU were shared with other jobs, so these numbers are indicative only:

  | model | rows | PyTorch GPU fp32 | PyTorch GPU bf16 | ONNX Runtime CPU (8 threads, `-wl`) |
  |---|---|---|---|---|
  | DeBERTa-v3-large | 18 × 161 | 82 ms | 62 ms | 3.2 s |
  | ModernBERT-large | 18 × 117 | 52 ms | 19 ms | 3.8 s |
- English only. The context is 512 tokens per pair, and the premise is cut first.

## Reproduce

```sh
cd convert   # M = deberta-v3-large | modernbert-large, SLUG = <M>-zeroshot-v2.0
uv run python -m ollaya_convert.families.nli.export $M --out out/$SLUG
uv run python -m ollaya_convert.families.weightless_ext out/$SLUG --prefix model. --out out/$SLUG-wl \
    --checkpoint ~/.cache/huggingface/hub/models--MoritzLaurer--<repo>/snapshots/<commit>/model.safetensors
uv run python -m ollaya_convert.families.nli.parity $M out/$SLUG-wl --td-limit 100
uv run python -m ollaya_convert.families.nli.goldens $M --out out/goldens-$SLUG.jsonl
```

## Attribution

- zeroshot-v2.0 © Moritz Laurer. DeBERTa variant MIT, ModernBERT variant Apache-2.0.
  Laurer et al., *Building Efficient Universal Classifiers with Natural Language Inference*,
  arXiv:2312.17543.
- DeBERTa-v3 © Microsoft, MIT. ModernBERT © Answer.AI and LightOn, Apache-2.0.
- **Data caveat (from the model card):** models without `-c` in the name, including
  `deberta-v3-large-zeroshot-v2.0`, were trained on a broader mix that includes non-commercially
  licensed NLI data (ANLI, WANLI, LingNLI, …). The weights are MIT. For strict commercial
  requirements, the card recommends `deberta-v3-large-zeroshot-v2.0-c`, which would convert with
  the same layout. ModernBERT-large-zeroshot-v2.0 uses "the same dataset mix".
