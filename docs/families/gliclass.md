# gliclass (`gliclass-uni-v1`)

GLiClass uni-encoder zero-shot classifiers by Knowledgator. The labels and the text share one
sequence. Each label is pooled from its own span and scored against the `[CLS]` embedding by an
MLP. GLiClass has no typed-question API, so the mapping from `choice` / `score` / `noul` onto its
`labels` / `prompt` / `text` call is Ollaya's. It was chosen on typed-decisions (see
[Template choice](#template-choice)). Everything from the text string onward is upstream's own
code path.

| | `gliclass-instruct-large-v1.0` | `gliclass-instruct-edge-v1.0` |
|---|---|---|
| Upstream | [knowledgator/gliclass-instruct-large-v1.0](https://huggingface.co/knowledgator/gliclass-instruct-large-v1.0) @ `825e5478c1bf4bffbf297690517097ccbdb2e006` | [knowledgator/gliclass-instruct-edge-v1.0](https://huggingface.co/knowledgator/gliclass-instruct-edge-v1.0) @ `727be8a417f6a7718e591b025e07054c146d8139` |
| Encoder | DeBERTa-v3-large (MIT) | Ettin-encoder-32M, ModernBERT architecture (MIT) |
| Parameters | 438.7M, fp32 (1.75 GB) | 32.7M, fp32 (131 MB) |
| License | Apache-2.0 (Knowledgator) | Apache-2.0 (Knowledgator) |
| `<<LABEL>>` / `<<SEP>>` / `<<EXAMPLE>>` ids | 128001 / 128002 / 128003 | 50368 / 50369 / 50370 |
| `[CLS]` / `[SEP]` / `[PAD]` | 1 / 2 / 0 | 50281 / 50282 / 50283 |

- Upstream code: `gliclass==0.1.20` (PyPI), [github.com/Knowledgator/GLiClass](https://github.com/Knowledgator/GLiClass) @ `40baa67`.
- Contract: M (`markers`), one row per question.
- Context: `max_len` 1024 tokens, the `ZeroShotClassificationPipeline` default.
- Reference: `convert/ollaya_convert/families/gliclass/ref.py`.
- `knowledgator/gliclass-edge-v1.0` does not exist. The edge model is `gliclass-instruct-edge-v1.0`.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (4.8 MB large, 1.2 MB edge; graph only) | derived, hosted by Ollaya (`convert/out/gliclass-instruct-{large,edge}-wl/`) |
| weights | upstream `model.safetensors` @ the commit above, F32, referenced by offset. Large: 405/406 tensors external. Edge: 77/78. The one unused tensor is `model.logit_scale`, which this config does not use. |
| `tokenizer.json` | upstream `tokenizer.json` @ the commit above, as is. **It has truncation `max_length` 4096 and `BatchLongest` padding baked in.** The runtime must set truncation to 1024 / right (see below) and turn padding off. |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |

**Tokenizer: the file is the reference, not transformers 5.**

- Under transformers 5, `AutoTokenizer` for DeBERTa-v2/v3 (large) does not load `tokenizer.json`
  verbatim. It rebuilds the normalizer in Python: `Replace(\s{2,}|[\n\r\t] → " ")`, NFC, right
  strip. The file instead uses `Strip` plus the sentencepiece `Precompiled` charsmap.
- As a result, characters the sentencepiece charsmap would fold tokenize differently. For example,
  a full-width `，` becomes `[UNK]` instead of `,`. In typed-decisions the affected characters are
  `‑` (U+2011), U+202F, `’`, `–`, `—` and `…`.
- That hit 31 of 882 case-set rows (the zh state) and **70 of the 400 typed-decisions states**.
- The checkpoint was trained under transformers 4.57.3, which loaded the file as is.
- So the reference tokenizes with the file (`families/repo_tokenizer.py`), and the Rust runtime
  uses that same file. No derived tokenizer is needed.
- The edge model (ModernBERT tokenizer) is unaffected: its file and transformers 5 agree.

## Request → row

One row per question.

1. **State text.** Ollaya's usual rendering (`ollaya_decision::serialize_state`): a string is used
   verbatim, anything else is `json.dumps(ensure_ascii=False)`.
2. **Labels and prompt**, from the shared parser's `Question`:

   | type | labels (in order) | prompt | option logits from label logits `z` |
   |---|---|---|---|
   | choice | `render_options()`: `label`, or `label: render_criterion(desc)` | instructions | `z` (K) |
   | score | `render_options()`: `level {i}: render_criterion(level)` | instructions | `z` (K) |
   | noul, **both** `false` and `true` descriptions non-empty ("pair") | `[render_criterion(false), render_criterion(true)]` | instructions | `[z0, z1]` = [false, true] |
   | noul otherwise ("single") | `[instructions]` | `""` | `[0, z0]` |

   `render_options` / `render_criterion` are the Laya functions already ported in
   `question.rs`. For noul, only the two descriptions are used here, not Laya's
   `false: …` / `true: …` strings.
3. **Sanitise.** Replace every `<<LABEL>>`, `<<SEP>>` and `<<EXAMPLE>>` with `" "` in the state
   text, the prompt and every label. Upstream would read them as extra classes or segments.
4. **Text.** Upstream `UniEncoderZeroShotClassificationPipeline.prepare_input`, with
   `prompt_first=True`:
   ```
   "".join("<<LABEL>>" + label for label in labels) + "<<SEP>>" + prompt + state_text
   ```
   There is no separator between the prompt and the state text. That is upstream's literal
   concatenation, and adding a space measured the same.
5. **Tokenize.** `encode(text, add_special_tokens=True)` gives `[CLS] … [SEP]`, truncated **from the
   right to 1024 tokens**, with the final `[SEP]` kept.
   - In Rust: `with_truncation(Some(TruncationParams { max_length: 1024, strategy: LongestFirst, direction: Right, stride: 0 }))`.
   - Equivalently: `[CLS] + encode(text, false)[..1022] + [SEP]`.
6. **Markers.** Every position whose id is the `<<LABEL>>` id, in order. If there are fewer than K
   (labels cut off by truncation), reject the question with 400. Upstream would crash.

Example (typed-decisions `choice`):

```
<<LABEL>>continue: Let the agent proceed without interruption.<<LABEL>>human_review: Queue this trace for a human to review.<<LABEL>>…<<SEP>>What should the observability system do with this trace?{"agent": {"autonomy": "checkpointed", …}, …}
```

## ONNX contract (M)

| Tensor | Type | Shape |
|---|---|---|
| `input_ids` | int64 | [n, s] |
| `attention_mask` | int64 | [n, s] |
| `marker_pos` | int64 | [n, k]; the `<<LABEL>>` positions, ascending |
| `marker_mask` | bool | [n, k] |
| `qtype` | int64 | [n]; ignored |
| → `logits` | float32 | [n, k]; masked slots are −1e4 |

- Dynamic axes: n 1..1024, s 8..1024, k 1..255. Opset 20. `min_markers` is 1.
- Pad rows with the pad id and `attention_mask` 0.

The graph is `GLiClassUniEncoder.forward` for this configuration:

1. `segment = 1` from the first `<<SEP>>` onward, `0` before it (upstream `argmax` semantics). It is
   added to the word embeddings before the encoder.
2. Label k's embedding is the mean of the hidden states from marker k up to, but not including,
   marker k+1. **The last label's span runs to the end of the sequence**, so it covers
   `<<SEP>>`, the prompt, the text and the final `[SEP]`. Padding is excluded. This is upstream's
   `_extract_class_features_averaged` with `extract_text_features=False`.
3. The text embedding is the hidden state at 0, then `text_projector`. Labels go through
   `classes_projector`. Then `MLPScorer`.

Sigmoid and softmax stay out of the graph.

## Option logits, calibration, answers

The server softmaxes the option logits with T = 1. Upstream ships no calibration, so
`calibration.json` is `[1, 1, 1]`. The result is exactly upstream's:

- choice / score / pair noul: `classification_type="single-label"`, a softmax over the labels.
- single noul: `"multi-label"`, `sigmoid(z)` = softmax(`[0, z]`).

## Template choice

Measured on typed-decisions test (400 states; argmax vs majority gold) with instruct-large. The
chosen variant is in bold.

| type | variant | acc |
|---|---|---|
| choice (600) | labels = keys | 0.378 |
| | **labels = `key: description`** | **0.422** |
| | labels = description only | 0.387 |
| | keys + descriptions in the text (training-augmentation style) | 0.20–0.23 |
| score (800) | levels verbatim | 0.388 |
| | **`level i: description`** (same as Laya) | **0.386** |
| noul with both descriptions (400) | single label = instructions (sigmoid) | 0.460 |
| | Laya `false: …`/`true: …` labels | 0.475 |
| | **[false description, true description], prompt = instructions** | **0.605** |
| noul without descriptions (200) | **single label = instructions (sigmoid)** | **0.670** |
| | labels `yes`/`no`, prompt = instructions | 0.690 |

This sweep ran with transformers-5 tokenization. The final numbers under
[Quality](#quality-and-speed) use the file tokenizer; score, for example, rises from 0.386 to 0.403.

For noul without descriptions, the single-label form is GLiClass's documented NLI usage ("the
hypothesis as a label"). It is kept over yes/no, since they are within noise at n = 200.

## Limits

- 1024 tokens per question row: labels, prompt and state share them, and the state is cut first.
- DeBERTa-v3 uses relative positions, so 1024 is fine (it is upstream's default).
- Labels: no hard cap. `max_num_classes` 25 in the config is only a training-time allocation, and
  the pipeline allocates dynamically. Quality beyond about 25 labels is untested.
- Ollaya's shared limits apply (choice ≤ 255, score 2–10).

## Measured parity

Command: `uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.parity {large|edge} out/gliclass-instruct-{large|edge}-wl --td-limit 100`.
Logs: `convert/out/logs/parity-gliclass-instruct-{large,edge}-wl.log`.

- The runs cover 882 questions: all edge cases and 100 typed-decisions rows.
- 1 question is rejected by design: `edge/many_options_77`, 77 long labels that are over 1024 tokens.
- The reference is fp32 on GPU with TF32 off, one unpadded row per question. ONNX ran as one padded
  batch per request on the CPU EP.

| Check | large | edge |
|---|---|---|
| Tokenizer: Rust core (`tokenizer.json`, truncation 1024, padding off) vs reference | 882/882 | 882/882 |
| Argmax, ONNX vs fp32 reference | 100.00% | 100.00% |
| Max probability diff | 3.4e-6 | 7.5e-6 |
| Max label-logit diff | 5.2e-5 | 8.6e-5 |
| Reference vs the upstream `ZeroShotClassificationPipeline(...)` call itself | 100% argmax, max diff 1.1e-7 | 100% argmax, max diff 1.5e-7 |

- Both are the weightless graphs (`-wl`), which reference the upstream safetensors.
- Goldens: `convert/out/goldens-gliclass-instruct-{large,edge}.jsonl`. They have 97 cases (all
  edge cases plus 20 typed-decisions rows). Each carries labels, prompt, text, ids, markers, fp32
  logits, option logits and probabilities, plus `rejected`.

## Quality and speed

- **typed-decisions test** (400 states, 2000 questions; argmax vs majority gold; measured here
  through the reference with the mapping above):

  | model | overall | choice | score | noul |
  |---|---|---|---|---|
  | instruct-large | **0.477** | 0.427 | 0.403 | 0.627 |
  | instruct-edge | 0.258 | 0.187 | 0.129 | 0.500 |

  For comparison, von 0.447 and `laya:en` 0.361. The gold labels have low annotator agreement,
  so read these as relative numbers. The mapping was chosen on this same set, so it is slightly
  optimistic.
- **The edge model is not usable for typed decisions.** It is at or below chance on choice and
  score. That matches its model card: average F1 0.49 vs 0.72 for large on 14 zero-shot
  classification sets.
- **Uncalibrated.** There is no temperature upstream.
  - Single-label (sigmoid) noul scores lean low: "The user asks for a refund." scored 0.03 on a
    text that asks for one.
  - Softmax over label logits is a ranking, not a calibrated probability.
- **Latency.** One 5-question typed-decisions request (5 rows). The RTX 4090 and the 24-core CPU
  were shared with other jobs at 100% GPU utilisation, so these numbers are indicative only:

  | model | PyTorch GPU fp32 | PyTorch GPU bf16 | ONNX Runtime CPU (8 threads, `-wl` graph) |
  |---|---|---|---|
  | large (5 × 198 tokens) | 37 ms | 54 ms | 1.38 s |
  | edge (5 × 174 tokens) | 16 ms | 18 ms | 93 ms |

  On CPU the edge graph ran the whole 882-question parity set in 15 s, against 300 s for large.
- English only.

## Reproduce

```sh
cd convert   # V = large | edge, SLUG = gliclass-instruct-$V
uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.export $V --out out/$SLUG
uv run python -m ollaya_convert.families.weightless_ext out/$SLUG --out out/$SLUG-wl \
    --checkpoint ~/.cache/huggingface/hub/models--knowledgator--gliclass-instruct-$V-v1.0/snapshots/<commit>/model.safetensors
uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.parity $V out/$SLUG-wl --td-limit 100
uv run --with gliclass==0.1.20 python -m ollaya_convert.families.gliclass.goldens $V --out out/goldens-$SLUG.jsonl
```

## Attribution

GLiClass © Knowledgator Engineering, Apache-2.0. Please cite: Stepanov et al., *GLiClass:
Generalist Lightweight Model for Sequence Classification Tasks*, arXiv:2508.07662. DeBERTa-v3 ©
Microsoft, MIT. Ettin © JHU CLSP, MIT.
