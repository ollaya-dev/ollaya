# d1-omni (`d1-omni-markers-v1`)

Liquid AI's [d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) (2026-10-05) is a 587M decision model on
[LFM2.5-Encoder-350M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M): a bidirectional LFM2 trunk (10 short
convolutions, 6 GQA attention layers), a two-layer decision head with a question-type embedding, and one `<|mask|>`
marker per option, all of a question's options in one sequence. The same trunk reads images (a SigLIP2 tower) and
speech (a FastConformer) as an embedding prefix in front of the text. **Ollaya serves the text path**; images and
audio are not ported (see [Not ported](#not-ported-images-and-audio)).

It is Von's successor: the Von card was archived on 2026-10-19 and points to d1-omni. It is **not** the `d1` family
(d1-3B is a generative LFM2.5-VL decoder run from GGUF on llama.cpp with max-pooled answer tokens); d1-omni is an
encoder with a marker head, so it is its own ONNX family. Its contract is Laya's and Von's (M, `markers`), with the
`qtype` input live (the head has a type embedding).

| | |
|---|---|
| Upstream | [huggingface.co/LiquidAI/d1-omni-600M](https://huggingface.co/LiquidAI/d1-omni-600M) @ `02b55d7076f15129e59ab3f94783f32c4b088674`; code: `modeling_d1.py`, `encoder.py`, `prompt.py` in the repo (`trust_remote_code`), `transformers>=5.15` |
| License | LFM Open License v1.0 (`license: other`, `license_name: lfm1.0`), the same text as d1-3B's: commercial use by an organisation with US$10M or more in annual revenue needs a separate agreement with Liquid AI. `latest` may point at it (the threshold is not a non-commercial clause). The package carries the full license. |
| Size | 587M parameters in one fp32 `model.safetensors` (2.35 GB): trunk 354.5M, head 26.2M, vision 94.2M, audio 112.2M. The text graph references trunk and head (381M, 1.52 GB); the vision and audio tensors stay unused. |
| Contract | M (`markers`), one row per question; `qtype` is read. |
| Context | `max_len` 16,384 tokens (config `max_length`): state, instructions and options together |
| Languages | the card lists en, de, es, fr, it, nl, pl, pt, ar, hi, ja, ru, tr, vi, zh |
| Reference | `convert/ollaya_convert/families/d1_omni/ref.py` (the repo's own modules, fp32, TF32 off) |

## Intake

- **License**: known (above), allows serving and a derived graph.
- **Weights**: on Hugging Face at a pinnable commit, fetched unmodified; nothing re-hosted.
- **Reference**: the repo's own inference code (`D1OmniModel.probabilities`, `prompt.encode`); the goldens come from it.
- **Fit**: fp32 reference 2.35 GB, trivially on the 24 GB parity GPU.
- **Engine**: ONNX port under `families/d1_omni/`.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph only) | derived, hosted by Ollaya (`convert/out/d1-omni-600m/model.onnx`) |
| weights | `LiquidAI/d1-omni-600M` @ `02b55d70…` / `model.safetensors`, referenced in place by byte offset (fp32, no cast) |
| `tokenizer.json` | the same repo and revision, used as is |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya; the calibration transcribes `config.json` `temperatures` |
| license | the repo's `LICENSE` |

## Request → rows (`prompt.encode`)

One row per question, `max_len` = 16,384:

```text
<|startoftext|> <|reserved_7|> enc(state)[:room] <|reserved_8|> enc(instructions)
    ( <|reserved_9|> <|mask|> enc(" " + option)[:per] <|reserved_10|> )*  <|reserved_11|>
```

- `enc(s)` tokenizes `escape(s)` without special tokens, where `escape` rewrites `<|name|>` to `<¦name¦>` (the regex
  `<\|([A-Za-z0-9_]+)\|>`), so caller text can never form a delimiter or a marker. Ollaya reuses Kev's
  `escape_delimiters`, the same rule.
- **State**: a string as it is, anything else `json.dumps(state, ensure_ascii=False)` (`serialize_state`, as Laya).
  `null` is the empty string.
- **Instructions**: Python `str()` of the value (a string verbatim; a dict as `{'k': 'v'}`), read from the raw
  definition, not the shared parser's `json.dumps`.
- **Options** are Laya's texts (`prompt.render_options` is `laya.common.render_options`): a choice `label` or
  `label: description` (`description` as JSON when structured), a score `level i: text`, a noul
  `false: …` then `true: …`, with `no, the statement does not hold` / `yes, the statement holds` when a side has no
  description. The noul keys are read exactly: `false` else `no`, `true` else `yes` (no lower-casing).
- **Budgets**: `budget = max(96, min(24k + 32, max_len / 2))`, `per = max(2, (budget - 3k) / k)` tokens per option
  text; the instructions keep at most `max(16, budget)` tokens (the question marker included);
  `room = max(0, max_len - len(question) - 2)` tokens of state. Markers are the positions of the `<|mask|>` tokens; the
  row is cut to `max_len` and rejected when its last marker does not fit ("the options do not fit in the context").
- **Rejections**, as upstream `Question.__post_init__`: a choice with fewer than 2 options, a score outside 2..10
  levels, noul criteria that are not a dict. Ollaya accepts a choice given as a list of labels (upstream wants a dict).
- **Option logits** are the row's marker logits in option order; a noul is read as `[false, true]` (upstream returns
  `[yes, no]` and Ollaya's order is `[false, true]`, so nothing is reordered).

## ONNX contract (M)

```text
inputs   input_ids      int64 [rows, seq]       right-padded with <|pad|> (0)
         attention_mask int64 [rows, seq]       1 on real tokens
         marker_pos     int64 [rows, markers]   position of each option's <|mask|>
         marker_mask    bool  [rows, markers]
         qtype          int64 [rows]            0 choice, 1 score, 2 noul
outputs  logits         float32 [rows, markers] the scorer at each marker; masked slots -1e4
```

The graph is upstream's `encoder.Trunk` (no media prefix: prefix length 0, so the convolution's right tap is kept
everywhere and the attention mask only hides padding) and `encoder.DecisionHead`, exported with `torch.export`
(opset 20) and rewritten weightless. The head's `nn.TransformerEncoderLayer` runs its standard path (the fused
fast path is switched off for the export; same arithmetic).

## Calibration

`config.json` `temperatures`, Laya's buckets: `choice:2` 1.747, `choice:3-5` 1.400, `choice:6-10` 1.175,
`choice:11+` 1.373, `noul:2` 1.666, `score:3-5` 1.730, `score:6-10` 1.0, with `choice`, `score` and `noul` 1.0 as the
fallback (a 2-level score). `calibration.json` carries them as `temperature_by_options` and `temperature`; the
runtime's `Calibration::temperature(qtype, k)` is the same lookup as upstream's `temperature_key`. All within the
default [0.5, 5] range. Image and audio requests run uncalibrated upstream, which does not concern the text path.

## Export parity

`families/d1_omni/parity.py` on choso (RTX 4090 for the reference, ONNX Runtime CPU for the graph), 2026-10-24: the
shared edge cases plus 50 typed-decisions rows, JSON round-tripped.

| | value |
|---|---|
| requests / questions (one row each) | 127 / 606; 14 requests rejected by both (a one-option choice in the shared edge set) |
| token rows and markers identical to upstream `prompt.encode` | 606 / 606, no rejection mismatches |
| argmax agreement, ONNX vs fp32 / vs upstream `probabilities` | **100 %** / **100 %** |
| max \|Δ logit\| | 5.1e-5 (choice), 1.1e-4 (noul), 7.1e-5 (score); p99 ≤ 6.9e-5 |
| max \|Δ probability\| | 9.6e-6 (choice), 1.1e-5 (noul), 1.0e-5 (score) |
| contract (marker logits + bucket temperature + softmax) vs upstream `probabilities` | ≤ 8.7e-8 |

Report: `convert/out/d1-omni-600m/parity.json`. Goldens: `convert/out/goldens-d1-omni-600m.jsonl`
(`families/d1_omni/goldens.py`, 83 requests plus the `#valid` subsets of the 14 rejected ones, 20 typed-decisions
rows): per question the row, the markers, the raw marker logits, the temperature and the probabilities, and
upstream's `system_one` answers per request.

- **Files.** The graph is 3.4 MB (179 checkpoint tensors referenced, 22 small constants inline, 432 bytes). It
  references `LiquidAI/d1-omni-600M@02b55d70…/model.safetensors` (2,348,774,500 bytes, fp32); 905 tensors of the
  vision and audio encoders stay unused.

## Rust runtime parity

`parity_d1_omni` on the goldens above (96 requests incl. the `#valid` subsets, 456 questions; the 14 rejected requests
are rejected by the runtime too), recorded by `parity_run` on 2026-10-24 (`results/runs/2026-10-24-parity-rtx4090-{cuda,cpu}.json`,
ollaya 0.17.0, graph sha256 as `files.json`):

| Device | encodings | decisions | max \|Δ logit\| | max \|Δ probability\| (p99) | forward, 456 questions |
|---|---|---|---|---|---|
| RTX 4090, CUDA (TF32 off) | 456 / 456 identical | **100 %** | 8.6e-5 | 1.1e-5 (7.2e-6) | 1.7 s (18 ms/request) |
| CPU (choso, Ryzen), fp32 | 456 / 456 identical | **100 %** | 1.4e-4 | 9.8e-6 (7.8e-6) | 60.4 s (629 ms/request) |

The gate is the shared one (identical encodings and decisions, logits within 1e-3); the temperature the runtime
picks per question equals the reference's on every question. The graph is fp32 end to end, so CUDA and CPU sit at
the same distance from PyTorch.

## Not ported: images and audio

An image or audio clip enters as a prefix of embeddings in front of the text; prefix positions attend only to the
prefix and the convolution never reads the text from them, so the prefix is a function of the media alone. With media
the state and question text is cut to 896 tokens (images) or 15,360 (audio), noul options are written `false: no` /
`true: yes`, and no temperature is applied. Images: a SigLIP2 tower from LFM2.5-VL-450M with tiling and a projector
(`vision.py`); audio: a 17-layer FastConformer, English requests, clips cut at 30 s (`audio.py`). Both need their own
graphs and request paths (`images` exists in `/api/decide`; audio has no entry); backlog items, decided separately.

## Limits

- **Context and memory.** Rows up to 16,384 tokens are accepted, as upstream. The six attention layers materialise
  `[16, seq, seq]` fp32 scores per row, about 17 GB at the full length; the runner runs long rows alone
  (token budget 8,192 per batch). A long state is cut to the room the question leaves.
- **Text only.** `images` on `/api/decide` is rejected for this model; audio has no entry.
- **Option count.** A choice needs 2 options; a score 2 to 10 levels. The export's marker axis allows 255.

## Reproduce

```sh
cd convert
uv run python -m ollaya_convert.families.d1_omni.export --out out/d1-omni-600m
uv run python -m ollaya_convert.families.d1_omni.parity out/d1-omni-600m --td-limit 50 --report out/d1-omni-600m/parity.json
uv run python -m ollaya_convert.families.d1_omni.goldens out/d1-omni-600m --td-limit 20
cargo run --release -p ollaya-runner --example parity_d1_omni -- convert/out/d1-omni-600m convert/out/goldens-d1-omni-600m.jsonl cuda
python -m ollaya_convert.parity_run --machine rtx-4090 --device cuda d1-omni:600m
```

## Attribution

d1-omni-600M by Liquid AI, LFM Open License v1.0; base model LFM2.5-Encoder-350M (Liquid AI). Ollaya ships the
graph it derived, the transcribed configuration and the author's license, and downloads the weights from the
author's repository.
