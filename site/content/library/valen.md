Valen-Decision-1.5 is Valen-Team's family of decision models, released under Apache-2.0: Qwen3.5 vision-language models, fully post-trained, with a Mixer head that scores every candidate answer. The state, an image if there is one, and every question share one sequence, so a request takes one forward pass however many questions it asks. It never generates text; the probabilities are a softmax of the head's scores, with no fitted temperature.

## Models

| Tag | Base | Params |
|---|---|---|
| `valen:0.8b` | Qwen3.5-0.8B, post-trained | 0.8B |
| `valen:latest`, `valen:2b` | Qwen3.5-2B, post-trained | 2B |
| `valen:4b` | Qwen3.5-4B, post-trained | 4B |

Valen-Team reports its own benchmark results on the [model cards](https://huggingface.co/Valen-Team/Valen-Decision-1.5-2B).

## Usage

```shell
ollaya run valen --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run valen --image shelf.png --questions '{"blocked":{"type":"noul","instructions":"Is the aisle blocked?"}}' "A photo from the warehouse camera."
```

Pass a PNG with `--image`, or base64 in `images` on `/api/decide`. Point any TypeSafe client at `http://localhost:11435` and set the model to `valen` for text requests.

## How it works

- **One sequence per request.** The prompt is built exactly as Valen-Team's `compiler.py` builds it: the state and its image once, every question with its candidates, then one decision readout per question. Text and images go through the same model.
- **Block attention.** In the full-attention layers the state is read on its own and the questions read the state and each other, as the authors trained it; Ollaya passes the same mask.
- **Weights.** Valen-Team's BF16 safetensors download from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the two ONNX graphs.
- **Parity.** Ollaya's Rust runtime matches the authors' own code in fp32, on CUDA and on the CPU: identical token ids, attention blocks, positions and resized pixels, and the same decision on all {{result valen:2b parity.cuda.questions 0}} test questions.

## Limits

- **No boxes.** Valen-Decision-1.5 can also locate things in an image (`bbox` questions); Ollaya's API has no answer type for a box, so only choice, score and yes/no questions are served.
- **One PNG per request**, between 65,536 and 1,638,400 pixels after the model's resize (larger images are scaled down). Convert a JPEG to PNG first. No video.
- **Length.** Up to 16,384 tokens per request, image tokens included; nothing is truncated, so a longer request is rejected.
