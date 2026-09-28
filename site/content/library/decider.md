decider is a family of open decision models by [Mapika](https://huggingface.co/Mapika), built on Qwen3.5 base models and released under Apache-2.0. For each question it lays out the context, the question and lettered options, then reads the model's next-token logits for the option letters at an answer slot. It never generates text; every question is one forward pass.

## Models

| Tag | Base | Params | Typed-decisions accuracy |
|---|---|---|---|
| `decider:latest`, `decider:2b` | Qwen3.5-2B | 1.9B | 0.591 |
| `decider:4b` | Qwen3.5-4B (v2.1) | 4.2B | **0.680** |
| `decider:0.8b` | Qwen3.5-0.8B | 0.75B | 0.506 |
| `decider:2b-vision` | Qwen3.5-2B vision-language (v5 text weights) | 2.2B | reads images; see below |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `nli` scores 0.548, `gliclass` 0.477 and `laya:en` 0.361. The labels have low annotator agreement, so compare the numbers against each other rather than reading them as absolutes. On Mapika's own benchmarks `4b` is ahead of `2b` (in-task 0.831 vs 0.805, held-out 0.784 vs 0.755, JevBench hard 0.649 vs 0.459), and the Decision Index scores it 36.6 against 26.1.

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

- **One PNG per request**, up to about one megapixel (1024x1024). The model's own preprocessing (Qwen's resize to multiples of 32 pixels, PIL's bicubic filter, normalization) is reproduced value for value, so a JPEG has to be converted to PNG first: Rust's JPEG decoders differ from the one the model's authors use by a few levels on some pixels.
- **Up to 10 options** per question (v5's lettered prompt, A to J). Text-only requests work too.
- **Parity.** On 404 test questions, 12 of them with images: identical resized pixels, token ids and positions, the same decision on every question, and option logits within 1.6e-4 of the Python reference, on CPU and CUDA.
- **Speed.** On an RTX 4090, five questions about a 256x240 image take about 200 ms, and about 0.9 s about a 1000x750 one. On the CPU, 3 to 16 seconds.
- `/v1/systemone` takes no images: TypeSafe's API has no field for them.

> `decider:2b-vision` needs Ollaya 0.7.5 or newer.

## Speed

- **RTX 4090, end to end:** a five-question request with a short state takes about 155 ms on `0.8b`, 190 ms on `2b` and 520 ms on `4b` at the median. That is slower than the encoder models (Laya: 8–10 ms). `0.8b` and `2b` are still faster than hosted Jev (236–276 ms).
- **State length:** cost grows with the number of options times the context length.
- **Memory:** on a 24 GB GPU, `2b` handles states up to about 8k tokens. Longer states spill out of GPU memory and become very slow.

## How it works

- **Weights.** They come from Mapika's own `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (8 to 11 MB).
- **Precision.** Every model computes in fp32 on the BF16 weights. `0.8b` and `2b` widen the weights to fp32 when they load. `4b` keeps them BF16, half the memory, and widens each layer just before it runs; its answers are the same, and a forward pass takes about 12% longer.
- **Calibration.** `4b` (v2.1) has one fitted temperature per answer type: choice 1.110, yes/no 1.560, score 1.287. `2b` and `0.8b` have one each.
- **Parity.** Ollaya's Rust runtime matches the Python reference exactly: identical token ids and answer-slot positions, and the same decision on every test question, on CPU and CUDA.

## Limits

- **Memory.** On the GPU, with requests of about 1,400 tokens, `2b` took about 13 GB and `4b` about 14 GB (`4b` keeps its 8.4 GB of weights BF16). All three are slow on the CPU, where `4b` peaked at 21 GB on the same requests; `4b` belongs on a GPU.
- **Score criteria.** Criteria given as an object instead of a list are rejected, as TypeSafe's API does.
- **Instructions.** Every question needs `instructions`. Mapika's newer package also accepts a yes/no question without them; Ollaya rejects it.
