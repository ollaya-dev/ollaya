[GLiClass v4 by Knowledgator](https://huggingface.co/knowledgator/gliclass-v4-2b) is the decoder generation of GLiClass: a Qwen3.5 decoder reads the instructions, the state and every label in one row, a small bidirectional scorer compares each label with the text, refines its reading for up to three steps, and the model's own calibrator rescales every score. One forward pass answers a question with any number of options, and nothing is generated. Ollaya serves the text path; the checkpoints also read images, which Ollaya does not serve yet.

It is not the `gliclass` model (the DeBERTa uni-encoder): a different architecture, so a different entry.

## Models

| Tag | Decoder | Weights read | Context |
|---|---|---|---|
| `gliclass-v4:latest`, `gliclass-v4:2b` | Qwen3.5-2B | 2.4B (BF16, about 7 GB in memory) | 8,192 tokens |
| `gliclass-v4:9b` | Qwen3.5-9B | 9.5B (BF16, about 20 GB in memory) | 8,192 tokens |

Upstream reports a Decision Index skill of 42.7 for the 2B at a median of 26 ms per decision; Ollaya's own measurements follow when the bench has run.

## Usage

```shell
ollaya run gliclass-v4 --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `gliclass-v4`.

## How it works

- **Questions as labels.** GLiClass classifies: it takes a text, labels and an optional instruction. Ollaya maps a choice or score question to one label per option with the instructions as the prompt and a softmax over the labels, a yes/no question with both descriptions to two labels, and any other yes/no question to a single label read as a probability. This is the same mapping as for `gliclass`.
- **One row per question.** The prompt, the state and the labels form one decoder row; the label section is then read by a two-layer scorer that lets labels see each other and the text, with up to three refinement steps that stop once the scores settle.
- **Calibrated by the model.** A learned calibrator predicts a temperature and a bias for every (text, label) pair from the representations and the logits; Ollaya applies no temperature of its own.
- **Weights.** The author's `model.safetensors` from Hugging Face, pinned to a commit and verified by sha256; Ollaya hosts only the ONNX graph and reads the decoder, scorer and calibrator tensors in place, the vision tower untouched.
- **Parity.** Ollaya's Rust runtime reproduces the `gliclass` package in fp32 on 483 test questions for `gliclass-v4:2b`: the same rows, markers and reasoning steps, the same decision on every question, calibrated logits within 5.7e-4 and probabilities within 1.5e-4 on CUDA (9.4e-5 and 2.5e-5 on the CPU). PARITY_9B

## Limits

- **Text only.** Images are not served.
- **Context.** Instructions and state share 8,192 tokens with the labels; a long state is cut from the end, the labels are kept whole.
- **Markers.** The literal texts `<<LABEL>>`, `<<SEP>>` and `<<EXAMPLE>>` in a state, instruction or option are replaced by a space, because the model reads them as its own markers.
