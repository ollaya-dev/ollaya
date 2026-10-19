Arbiter by [Codekins Pvt Ltd · Zyot Lab](https://huggingface.co/hiteshluke/arbiter-4b) is Gemma 3 4B IT with a LoRA and a fixed 24-slot head, trained on Jev-style decisions. One prompt per question ends at `Answer:`, and the head reads every option there at once: `T` and `F` for yes/no, the letters `A` to `P` for a choice, the digits `0` to `5` for a score. It never generates text.

> Needs Ollaya 0.12.0 or newer, the first release that runs the `arbiter-fixed-v1` layout.

## Models

| Tag | Weights | Questions it answers | Five questions, RTX 5090 |
|---|---|---|---|
| `arbiter:latest`, `arbiter:4b` | Arbiter v3.3: Gemma 3 4B IT (BF16) with the authors' LoRA and head, about 8 GB | noul, choices of 1 to 16 options, scores of exactly 6 levels | {{result arbiter:4b gpu.p50_ms 0}} ms |

On typed-decisions (all 400 states, argmax against the majority label) the fixed head answers 1,200 of the 2,000 questions: none of the 800 score questions has 6 levels. On those 1,200 it scores 0.620 (choice 0.563, yes/no 0.677), with no fitted temperature, so the number is not comparable with the other models' full scores.

## Usage

```shell
ollaya run arbiter --questions '{"team": {"type": "choice", "instructions": "Which team should handle this?", "criteria": {"billing": "Payments and refunds", "shipping": "Deliveries", "technical": "Bugs and outages"}}, "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}}' "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

The built-in presets (`--preset triage` and the others) each have a score question with 3 or 4 levels, which the fixed head cannot answer, so Arbiter rejects them: pass your own questions. Point any TypeSafe client at `http://localhost:11435` and set the model to `arbiter`.

## How it works

- **Prompt.** The three prompt templates of the authors' training script, verbatim: the state, the question, the options as `A. ...` lines (or `T. Yes / True`, `F. No / False`, or `0` to `5`), then `Answer:`.
- **Head.** A `Linear(2560, 24)` on Gemma's last hidden state at the prompt's final token. Its rows started as the language model's rows for the 24 answer tokens and were trained with the LoRA.
- **Engine.** ONNX Runtime on an NVIDIA GPU (CUDA) or the CPU. The base weights stay BF16 in memory and the LoRA runs unmerged, so every file is the authors' own, read in place.
- **Parity.** Ollaya's runtime matches the reference (transformers' Gemma 3 with the LoRA and head, fp32) on the CPU and on CUDA: the same token rows, the same decision on all 420 test questions, probabilities within 1.0e-5.

## Limits

- **Options.** A choice of more than 16 options, or a score with other than 6 levels, is rejected (422), not rescaled.
- **Rows.** Up to 8,192 tokens per question, the whole state included; the model was trained on up to 768.
- **Memory.** About 8 GB.
- **License.** The LoRA adapter and head are Apache-2.0. The base model, Gemma 3 4B IT by Google DeepMind, is under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
