decider is a family of open decision models by [Mapika](https://huggingface.co/Mapika), built on Qwen3.5 base models and released under Apache-2.0. For each question it lays out the context, the question and lettered options, then reads the model's next-token logits for the option letters at an answer slot. It never generates text; every question is one forward pass.

## Models

| Tag | Base | Params | Typed-decisions accuracy |
|---|---|---|---|
| `decider:latest`, `decider:2b` | Qwen3.5-2B (v12) | 1.9B | {{result decider:2b typed.accuracy 3}} |
| `decider:4b` | Qwen3.5-4B (v3) | 4.2B | **{{result decider:4b typed.accuracy 3}}** |
| `decider:0.8b` | Qwen3.5-0.8B | 0.75B | {{result decider:0.8b typed.accuracy 3}} |
| `decider:2b-vision` | Qwen3.5-2B vision-language (v5 text weights) | 2.2B | reads images; see below |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `nli` scores {{result nli:deberta-v3-large typed.accuracy 3}}, `gliclass` {{result gliclass:large typed.accuracy 3}} and `laya:en` {{result laya:en typed.accuracy 3}}. The labels have low annotator agreement, so compare the numbers against each other rather than reading them as absolutes. On Mapika's own numbers `4b` (v3) is ahead of `2b` (v12): JevBench public items hard 0.640 vs 0.595, a 4,500-request sample of the Decision Index 0.3 at 51.0 vs 38.2.

> `decider:4b` needs Ollaya 0.7.0 or newer, which keeps its weights BF16 in memory. Older versions widen them to fp32 (about 17 GB).

## Usage

```shell
ollaya run decider --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run decider:4b --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

## Images: decider:2b-vision

`decider:2b-vision` answers questions about an image as well as the state: a photo, a screenshot, a diagram or a game frame. Pass a PNG with `--image`, or base64 in `images` on `/api/decide`:

```shell
ollaya run decider:2b-vision --image shelf.png --questions '{"blocked":{"type":"noul","instructions":"Is the aisle blocked?"}}' "A photo from the warehouse camera."
```

- **One PNG per request**, up to about one megapixel (1024x1024). The model's preprocessing is reproduced value for value, so convert a JPEG to PNG first: Rust's JPEG decoders differ from the authors' by a few levels on some pixels.
- **Up to 10 options** per question (v5's lettered prompt, A to J). Text-only requests work too.
- **Parity.** On 404 test questions, 12 of them with images: identical resized pixels, token ids and positions, the same decision on every question, and option logits within 1.6e-4 of the Python reference, on CPU and CUDA.
- **Speed.** On an RTX 4090, five questions about a 256x240 image take about 200 ms, and about 0.9 s about a 1000x750 one. On the CPU, 3 to 16 seconds.
- `/v1/systemone` takes no images: TypeSafe's API has no field for them.

> `decider:2b-vision` needs Ollaya 0.7.5 or newer.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** five questions with a short state take {{result decider:0.8b gpu.p50_ms 0}} ms on `0.8b`, {{result decider:2b gpu.p50_ms 0}} ms on `2b` and {{result decider:4b gpu.p50_ms 0}} ms on `4b` at the median: slower than the encoders (`laya:en`: {{result laya:en gpu.p50_ms 0}} ms), and for `0.8b` and `2b` still faster than hosted Jev (236–276 ms).
- **State length:** cost grows with the number of options times the context length.
- **Memory:** on a 24 GB GPU, `2b` handles states up to about 8k tokens. Longer states spill out of GPU memory and become very slow.

## How it works

- **Weights.** They come from Mapika's own `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (8 to 11 MB).
- **Precision.** fp32 compute on the BF16 weights. `0.8b` and `2b` widen the weights when they load; `4b` keeps them BF16 (half the memory) and widens each layer just before it runs, with the same answers and about 12% more time per pass.
- **Calibration.** `4b` (v3) and `2b` (v12) have one temperature per answer type from Mapika's config: choice 2.098 and 2.465, score 3.152 and 2.833, yes/no 0.05 on both. The yes/no value is Mapika's choice for JevBench's rule that a yes/no probability between 0.2 and 0.8 counts as wrong, so yes/no answers come out near 0 or 1 and are not calibrated probabilities (the fitted values would be 2.795 and 3.185, per the model cards). `0.8b` has one temperature.
- **Parity.** Ollaya's Rust runtime matches the Python reference exactly: identical token ids and answer-slot positions, and the same decision on every test question, on CPU and CUDA.

## Limits

- **Memory.** With requests of about 1,400 tokens, `2b` took about 13 GB of GPU memory and `4b` about 14 GB. All three are slow on the CPU, where `4b` peaked at 21 GB; `4b` belongs on a GPU.
- **Score criteria.** Criteria given as an object instead of a list are rejected, as TypeSafe's API does.
- **Instructions.** Every question needs `instructions`. Mapika's newer package also accepts a yes/no question without them; Ollaya rejects it.
