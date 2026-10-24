[d1-omni-600M by Liquid AI](https://huggingface.co/LiquidAI/d1-omni-600M) is a 587M decision model built on LFM2.5-Encoder-350M: a bidirectional LFM2 encoder with a decision head that reads every option at its own marker, all of a question's options in one pass, with no generation. Ollaya runs its text path. The same checkpoint also reads images and speech, which Ollaya does not serve yet.

It is the successor of `von`, whose author archived Von on 2026-10-19 and points to d1-omni. It is not the `d1` family: d1-3B is a generative model run from GGUF, d1-omni an encoder with a marker head.

## Models

| Tag | Trunk | Parameters | Context | Languages |
|---|---|---|---|---|
| `d1-omni:latest`, `d1-omni:600m` | LFM2.5-Encoder-350M | 587M in the checkpoint, 381M read for text | 16,384 tokens | en, de, es, fr, it, nl, pl, pt, ar, hi, ja, ru, tr, vi, zh |

## Usage

```shell
ollaya run d1-omni --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `d1-omni`.

## How it works

- **One sequence per question.** The state, then the instructions, then every option after its own `<|mask|>` marker, between the model's reserved delimiter tokens; `<|…|>` in caller text is rewritten so a state can never forge a delimiter. The head adds a question-type embedding and scores the hidden state at each marker.
- **State as JSON.** A string state is read as it is; any other JSON value is written as compact JSON, the way the model was trained.
- **Options.** A choice option reads `label: description` (or the bare label), a score level `level i: text`, a yes/no question `false: …` and `true: …` with the caller's descriptions or the model's defaults.
- **Calibration.** One temperature per question type and option count (2, 3–5, 6–10, 11+), fitted by the author and shipped in the model's config; Ollaya applies them unchanged.
- **Weights.** The author's fp32 `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256; the graph reads the encoder and head tensors in place and leaves the vision and audio tensors untouched. Ollaya hosts only the graph.
- **Parity.** Ollaya's Rust runtime reproduces the author's PyTorch code in fp32 on 456 test questions: the same token ids and markers, the same decision on every question, logits within 1.4e-4 and probabilities within 1.1e-5 on CUDA and the CPU.

## Limits

- **Text only.** Images and audio are not served; a request with `images` is rejected for this model.
- **Context.** Instructions, state and options share 16,384 tokens. The options get a token budget first, the instructions keep at most that many tokens, and a long state is cut to the room that is left. A row near the full length needs several gigabytes of memory for its attention scores.
- **Options.** A choice needs at least two options; a score takes 2 to 10 levels.
- **License.** [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/LICENSE): commercial use by an organization with US$10 million or more in annual revenue needs a separate agreement with Liquid AI.
