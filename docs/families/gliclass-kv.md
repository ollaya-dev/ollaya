# gliclass-kv (`gliclass-kv-v1`)

Knowledgator's **GLiClass v4** ([gliclass-v4-2b](https://huggingface.co/knowledgator/gliclass-v4-2b),
[gliclass-v4-9b](https://huggingface.co/knowledgator/gliclass-v4-9b), 2026-10) is GLiClass's *decoder-KV* architecture
(the SCX Router paper): a Qwen3.5 decoder reads the instruction, the text and the labels in one causal row; the label
section's hidden states go through a two-layer bidirectional DeBERTa scorer, an MLP scores every (text, label) pair, a
gated recurrent cell refines the label states for up to three steps and stops early once the probabilities settle, and
a learned calibrator rescales every label logit. It returns a score per label and generates no text. The checkpoints
also read images (the Qwen3.5 vision tower); Ollaya serves the text path.

It shares nothing with the uni-encoder `gliclass` family but the name and Ollaya's question mapping, so it is its own
family and catalog entry (`gliclass-v4:2b`, `gliclass-v4:9b`; `latest` is `2b`).

| | `gliclass-v4-2b` | `gliclass-v4-9b` |
|---|---|---|
| Upstream | `knowledgator/gliclass-v4-2b` @ `40fe023b45e8d97d1ae4f7f9342279ac1c76317d` (2026-10-19) | `knowledgator/gliclass-v4-9b` @ `14d11a92b5e0bab4a166e1fce4cc0777cac27182` (2026-10-16) |
| Decoder | Qwen/Qwen3.5-2B (hidden 2048, 24 layers) | Qwen/Qwen3.5-9B (hidden 4096, 32 layers) |
| Checkpoint | `model.safetensors`, 4.73 GB: decoder and scorer BF16, calibrator F32; the vision tower (0.5B) unused | `model.safetensors`, 17.97 GB, the same layout |
| License | Apache-2.0 (the card; no LICENSE file); base Apache-2.0 | same |
| Code | `gliclass==0.1.21` (PyPI), [github.com/Knowledgator/GLiClass](https://github.com/Knowledgator/GLiClass) @ `6a00f80` | same |
| Window | 8,192 tokens per row (the card's usage; the pipeline's own default is 1,024) | same |
| `<<LABEL>>` / `<<SEP>>` / `<<EXAMPLE>>` / pad | 248077 / 248078 / 248079 / 248044 | same |

## Intake

- **License**: Apache-2.0 on the card and in the repo metadata; the base models are Apache-2.0.
- **Weights**: on Hugging Face at pinnable commits, one `model.safetensors` each, referenced in place.
- **Reference**: the `gliclass` package's own `GLiClassModel` and `DecoderKVZeroShotClassificationPipeline`.
- **Fit**: the 2B's fp32 reference (9.5 GB) runs on the parity GPU; the 9B's (38 GB) runs on the CPU.
- **Engine**: ONNX port under `families/gliclass_kv/`.

## Request → row

GLiClass has no typed questions. The mapping onto its `labels` / `prompt` call is the uni-encoder family's
(`families.gliclass.ref.question_call`, [gliclass.md](gliclass.md#template-choice)): choice and score questions
put every option as a label (`label`, `label: description`, `level i: text`) with the instructions as the prompt and
softmax over the labels; a noul with both descriptions puts them as two labels `[false, true]`; any other noul puts
the instructions as the single label with no prompt and reads its sigmoid. The three marker texts are replaced by
`" "` in every user string.

Upstream `prepare_inputs` (`prompt_first`): the context is the prompt immediately followed by the state text (no
separator), tokenized with no special tokens and cut from the right to `max_len` minus the label section (at least
one token); the label section is `<<SEP>>` label₁ `<<LABEL>>` … labelₖ `<<LABEL>>` `<<SEP>>`, tokenized as one string:

```text
<prompt><state> <<SEP>> label₁ <<LABEL>> … labelₖ <<LABEL>> <<SEP>>
```

The label section starts after the `<<SEP>>` that follows the context; label j is read at its `<<LABEL>>`, the text
at the final `<<SEP>>`. Port: `families/gliclass_kv/layout.py`, `crates/ollaya-decision/src/gliclass_kv.rs`.

## Graph

`families/gliclass_kv/export.py`: `llm_common/qwen35.py` (the scan-based Gated-DeltaNet trunk, as for `decider`)
over the whole row, then upstream's `DecoderKVScorer` and calibrator on the label section:

1. `scorer_encoder` (DeBERTa-v2, 2 layers, relative attention) over the section with its mask;
2. text representation at the final `<<SEP>>` → `text_projector`; label representations at the `<<LABEL>>`s →
   `label_projector`; `mlp` on each `[text, label]` pair → logits;
3. `recurrent_steps − 1` (= 2) gated steps (`RecurrentReasoningCell`: reset gate, the same encoder, update gate), each
   row frozen once no label's sigmoid probability moved by `recurrent_halt_threshold` (0.01) since the previous step,
   as upstream's `torch.where` logic, in the graph;
4. the calibrator: `beta ∈ (0, 3)` and a bias from LayerNormed text and label representations, their product and
   difference, and six logit statistics; the output is `beta · z + bias`.

```text
inputs   input_ids    int64 [rows, seq]      seq a multiple of 64, right-padded (pad id is free)
         section_pos  int64 [rows, len]      positions of the label section's tokens; shorter rows repeat their final <<SEP>>
         section_mask bool  [rows, len]
         label_pos    int64 [rows, k]        each <<LABEL>>'s index within the section
         label_mask   bool  [rows, k]
         sep_pos      int64 [rows]           the final <<SEP>>'s index within the section
outputs  logits       float32 [rows, k]      calibrated label logits; masked slots -1e4
         steps        int64   [rows]         recurrent steps the halting rule ran
```

The runtime's option logits are the row's label logits, or `[0, z]` for a one-label noul; `calibration.json` is the
identity (the calibrator is in the graph). `weights_in_memory` is `bf16`.

## Export parity

EXPORT_PARITY_2B

## Rust runtime parity

RUNTIME_PARITY_2B

## Limits

- **Text only.** The vision path (images in the context through the Qwen3.5 vision tower) is not ported.
- **Context.** 8,192 tokens per row; the context (prompt, then state) is cut from the right, the labels are kept whole.
- **Labels.** Upstream's `max_num_classes` is 25 in the config but the pipeline does not enforce it; the export's label
  axis allows 255.
- **Halting.** The recurrent cell's stopping rule depends on the probabilities, so a row's cost is one to three scorer
  passes; the halting threshold is the config's.

## Reproduce

```sh
cd convert
uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.export gliclass-v4-2b --out out/gliclass-v4-2b
uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.parity gliclass-v4-2b out/gliclass-v4-2b --td-limit 50 --report out/gliclass-v4-2b/parity.json
uv run --with gliclass==0.1.21 python -m ollaya_convert.families.gliclass_kv.goldens gliclass-v4-2b out/gliclass-v4-2b --td-limit 20
cargo run --release -p ollaya-runner --example parity_gliclass_kv -- convert/out/gliclass-v4-2b convert/out/goldens-gliclass-v4-2b.jsonl cuda
python -m ollaya_convert.parity_run --machine rtx-4090 --device cuda gliclass-v4:2b
```

## Attribution

GLiClass v4 by Knowledgator, Apache-2.0; base models Qwen3.5-2B and Qwen3.5-9B by the Qwen team, Apache-2.0. Ollaya
ships the graph it derived, the transcribed configuration and the author's license, and downloads the weights from
the author's repository.
