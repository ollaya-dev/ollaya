# arbiter (`arbiter-fixed-v1`)

Arbiter is Codekins Pvt Ltd / Zyot Lab's 4B decision model. It is built from three parts:

- a LoRA adapter (r=16, α=32, dropout=0.05) on `unsloth/gemma-3-4b-it`;
- a **fixed 24-slot pointer head**: `nn.Linear(2560, 24)` read at the final-position hidden state of
  the prompt, initialized from the base LM-head verbalizer rows for the 24 answer tokens and trained
  jointly with the LoRA (EMA-averaged over the last 5 eval checkpoints);
- the standard arbiter prompt template (`State: / Question: / Options: / Answer:`), identical to training.

The model repository `hiteshluke/arbiter-4b` holds the adapter, `head.pt` and the tokenizer. The base
weights come unmodified from `unsloth/gemma-3-4b-it`. The inference code lives inline under
`convert/ollaya_convert/families/arbiter/` (no external `arbiter` python package).

| Model | Checkpoint | Base (license) | Temperature | Status in Ollaya |
|---|---|---|---|---|
| `hiteshluke/arbiter-4b` | v3.3 (`0c44271c59f89758e3cae17b032e98a9140093e9`) | `unsloth/gemma-3-4b-it` @ `bf46152c47f5dd20b896357cb51abc4c03b8ee8c` (Gemma Terms of Use), 2 shards | 1.0 | **converted, ONNX** (weights stay BF16 in memory) |

## Recommended engine: ONNX (single forward)

- **Why ONNX.** A decision is one causal row per question plus a fixed 24-slot `Linear(2560, 24)` head.
  The head is not the LM head, so llama.cpp could not produce these scores without custom code; ONNX
  runs the head directly.
- **LoRA.** It is **not merged**. Each adapted Linear runs as `x·Wᵀ + (α/r)·(x·Aᵀ)·Bᵀ`, so base and
  adapter weights both stay byte-referenced in the weightless export.
- **Numerics.** Merged vs unmerged differ by at most ≈1e-6 on the slot scores (fp32 compute, BF16 base
  weights widened per forward pass).

## Architecture: 24-slot pointer head

The head's output tensor has a fixed size of 24; a question's valid slots are picked by type:

| Slot(s) | Verbalizer | Used by |
|---|---|---|
| 0 | `T` | noul (true) |
| 1 | `F` | noul (false) |
| 2..17 | `A`..`P` | choice (option index 0..15) |
| 18..23 | `0`..`5` | score (level 0..5) |

The 24 verbalizer tokens include two independent `F`s: slot 1 is the `F` in `T/F`, slot 7 is the `F`
in `A..P`. They tokenize to the same base-model id; the slot order is what distinguishes them, so each
is resolved by tokenizing the single character at the correct index.

The 24 head rows are initialized from the base LM-head rows for these 24 token ids, then trained
jointly with the LoRA. Slots 0 (`T`) and 1 (`F`) have their gradient frozen during training so the
noul primitive matches the LM head's own calibration on the true/false tokens. The checkpoint ships
the EMA average of the last 5 eval heads.

## Files (nothing re-hosted)

| Layer | Source |
|---|---|
| `model.onnx` (graph) | derived, hosted by Ollaya |
| base weights, BF16 | `unsloth/gemma-3-4b-it`'s `model-0000i-of-00002.safetensors` shards at the pinned revision (≈8.6 GB total). The vision tower tensors are unused. |
| LoRA adapter, F32 | the arbiter repo's `adapter_model.safetensors` |
| 24-slot head, F32 | same repo, `head.pt`. A `torch.save` zip whose tensors are stored uncompressed, so the graph references them **by byte offset inside the zip**. Nothing is re-packed. |
| `tokenizer.json` | same repo, used as-is (the Gemma 3 tokenizer) |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |
| license | Apache-2.0 (adapter + head). The base has its own Gemma Terms of Use. |

- **Mapping.** Every base tensor of the language model, every adapter tensor and the two head tensors
  map to graph initializers; the base's are a `Cast` from BF16, the rest F32.
- **Weights in memory.** `decision.json` `weights_in_memory` is `bf16` (kept as stored, widened per
  forward pass).
- **Hashes.** sha256 values are in each export's `files.json` (`convert/out/arbiter-4b`).

## Request → rows

Source: `convert/ollaya_convert/families/arbiter/layout.py` (`ArbiterLayout.encode`).

### Validation

Failures are HTTP 422.

- `questions` needs at least one entry.
- `type` must be one of `noul`, `choice`, `score`.
- **noul.** `criteria` is an object or null (unused: the two options are always `Yes / True` and `No / False`).
- **choice.** `criteria` is an object of **1..16** entries.
- **score.** `criteria` is a list of exactly **6** entries (one description per level, 0..5).
- **instructions.** Optional; any JSON (rendered to text).

### Text rendering (`render`)

- `None` renders as `""`.
- A str, int, float or bool renders as Python `str(v)`: `True`, `1.0`, `1e-05`.
- A list renders as `"\n".join(f"{pad}- {render(x, indent+1).lstrip()}")`.
- A dict renders as `"\n".join(f"{pad}{k}:\n{render(x, indent+1)}" if dict/list else f"{pad}{k}: {render(x)}")`.
- `pad = "  " * indent`.

### Prompt format

Each row is the following text, tokenized with the base tokenizer (`add_special_tokens=False`):

```
State: {render(state)}

Question: {render(instructions)}

Options:
{options block}

Answer:
```

Options block by primitive:

| Type | Options block |
|---|---|
| noul | `T. Yes / True\nF. No / False` |
| choice | `A. {opt0}\nB. {opt1}\n...` (one letter per option, A..P) |
| score | `0\n1\n2\n3\n4\n5` |

The row's last position is where the pointer head is read. There are no delimiter tokens: the layout
is just the tokenized prompt, right-padded for batching.

### Option logits

- **noul.** `scores[row, 0]` is the true-slot score, `scores[row, 1]` is false.
- **choice.** `scores[row, 2 + option_index]` for `option_index in 0..k-1`.
- **score.** `scores[row, 18 + level]` for `level in 0..5`.
- **Calibration.** `calibration.json` ships `temperature = [1.0, 1.0, 1.0]`; no fitted temperature is
  applied in v3.3.

## ONNX contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | one row per question; **`seq` a multiple of 64**; right-pad with any id (`pad`) |
| `last_pos` | int64 | `[rows]` | index of the final valid token in each row (where the head is read) |
| `scores` | float32 | `[rows, 24]` | raw 24-slot pointer scores; the server masks to the row's valid slots by type |

- **Positions and masking.** Positions are implicit (`0..seq-1`). An attention mask is derived inside
  the graph from `last_pos` (positions ≤ `last_pos[row]` attend).
- **Precision.** fp32 compute on BF16 base weights.

## Measured accuracy

Arbiter v3.3 at ship time:

| Benchmark | Accuracy | n |
|---|---|---|
| BoolQ (dev) | 0.849 | 1,000 |
| ARC-Challenge (test) | 0.738 | 500 |
| CommonsenseQA (dev) | 0.706 | 500 |
| OpenBookQA (test) | 0.722 | 500 |

## Measured parity

`families/arbiter/parity.py`: ONNX Runtime CPU with the LoRA unmerged, against the inline fp32 reference
(`ref.load(dtype=fp32, merge=True)`). Tolerance: max |Δ slot score| < 1e-4; 100 % argmax agreement.
Numbers are filled in once a maintainer with GPU runs the export against the pinned revisions and
drops the resulting `model.onnx` sha256 into both manifests.

## Attribution

- **Arbiter** (LoRA adapter + 24-slot pointer head) by Codekins Pvt Ltd / Zyot Lab, Apache-2.0.
- **Base model**: Gemma 3 4B IT by Google DeepMind
  (<https://huggingface.co/google/gemma-3-4b-it>), under the Gemma Terms of Use.
- **Training data**: `SargeDev/jev-distill-corpus-v3`
  (<https://huggingface.co/datasets/SargeDev/jev-distill-corpus-v3>); see the dataset card for its own
  licensing.
