# decider:2b-vision (`decider-vision-v1`)

[Mapika/decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) (Apache-2.0) is the
vision variant of decider-2b: decider-2b **v5** language weights transplanted into the full
Qwen3.5-2B vision-language model and fine-tuned so that an image plus a lettered question gives a
probability over the options at one answer slot. It never generates. Upstream code:
`decider/vision.py` and `decider/prompt.py` at `863e2908`. Ollaya serves it as `decider:2b-vision`.

## Request -> rows

One row per question, as for `decider-slots-v1` ([decider.md](decider.md)), narrowed to v5's
prompt: at most 10 options, lettered `A`-`J`, every question one `Answer: (` slot. Upstream then
decodes each row's ids and tokenizes the text again, with the image placeholder in front:

```text
[<|vision_start|>] + [<|image_pad|>] * n + [<|vision_end|>] + encode(decode(ids))   with an image
encode(decode(ids))                                                                 without
```

`n` is the image's patch count / 4 (2x2 merge). Every question's row carries the whole image, as
upstream batches one `(image, example)` pair per question.

## Image preprocessing (`crates/ollaya-runner/src/vision.rs`)

Qwen's `Qwen2VLImageProcessorPil`, value for value:

1. **Decode** PNG to 8-bit RGB as `PIL.Image.convert("RGB")` does: alpha dropped (not composited),
   grayscale replicated, palette expanded, 16-bit reduced to the high byte.
2. **smart_resize** to multiples of 32 (patch 16 x merge 2), rounding halves to even as Python's
   `round` does, then scaled into [65,536, 16,777,216] pixels.
3. **Resize** with PIL's bicubic filter in its 8-bit fixed-point form (`libImaging/Resample.c`:
   `precompute_coeffs`, 22-bit coefficients, horizontal pass over the rows the vertical pass needs,
   then the vertical pass, each rounding to 8 bits).
4. **Rescale** `x * (1/255)` in f64, then f32; **normalize** `(x - 0.5) / 0.5` in f32.
5. **Patches** of 3 x 2 x 16 x 16 (the temporal pair repeats the image), in 2x2 merge-block order.
   With them go the position tables the vision graph reads: bilinear taps (align corners) into the
   learned 48 x 48 position table, and each patch's (row, column) for the 2-D rotary embedding.

**PNG only.** Upstream decodes with PIL (libjpeg-turbo for JPEG). The Rust JPEG decoders tried
(`zune-jpeg` 0.5, `jpeg-decoder` 0.3) differ from libjpeg-turbo 3.1 by up to 4 levels on many pixels
(4:2:0 and 4:4:4, baseline and progressive), so the preprocessing would no longer be exact. A JPEG
gets a 422 that asks for PNG.

**Size cap.** At most 4,096 patches after the resize (1,024 visual tokens, about 1 MP). The vision
tower attends over all of an image's patches at once and every row carries the image: with five
questions on an RTX 4090 a 256x240 image took 8.3 GB, 0.8 MP 10.6 GB, 1.3 MP 13.7 GB and 2 MP
19.3 GB. Upstream has no cap (up to 16.7 MP); a larger image gets a 422 that says so.

## Positions (interleaved M-RoPE)

The full-attention layers use M-RoPE with sections [11, 11, 10] and partial rotary 0.25. Text
tokens count up on all three axes. An image starting at position `p` (1, after `vision_start`)
puts merged cell (i, j) at `(p, p + i, p + j)`, and the text after it (from `vision_end`) continues
at `p + max(rows, columns)` of the merged grid (upstream `get_rope_index`). Text-only rows have
positions `0..len` on every axis, the 1-D positions of the text model.

## ONNX contract

Two weightless graphs, both reading the author's `model.safetensors` (BF16) by byte offset:

| Graph | Inputs | Output |
|---|---|---|
| `vision.onnx` (registry layer annotated `org.ollaya.graph: vision`) | `patches` [N, 1536], `pos_idx` [N, 4], `pos_w` [N, 4], `rot_ids` [N, 2] | `image_embeds` [N/4, 2048] |
| `model.onnx` | `input_ids` [R, T], `position_ids` [3, R, T], `image_embeds` [M, 2048], `image_pos` [M], `slot_pos` [R] | `letter_logits` [R, 10] |

`T` is a multiple of 64 with at least one padding position; `image_pos` holds flat indices
`r * T + t` where the image embeddings replace the token embeddings. A text-only batch passes one
zero embedding at a padding position after every row, which no slot attends to (the model is
causal). The daemon passes the vision graph to the runner as `--vision-graph`.

## API

`/api/decide` takes `images` (base64 or a base64 `data:` URL, Ollama's field), and `ollaya run`
takes `--image FILE`. One image per request. `/v1/*` stays identical to TypeSafe's API, which has
no image field. A model that reads no images rejects a request with `images` (422).

## Calibration

None upstream: `decider_config.json` has no temperature for the vision model, so the temperature
is 1.0.

## Parity (measured)

Goldens: `ollaya_convert.families.decider_vision.goldens`, upstream `VisionDecisionModel` in fp32
on synthetic PNGs (sizes that resize down, up to the pixel minimum and not at all; RGBA, grayscale
and palette PNGs; text) plus the shared text cases, 103 requests (12 with an image, 17 rejected
upstream). `cargo run --release -p ollaya-runner --example parity_decider_vision`:

| | CPU | CUDA (RTX 4090) |
|---|---|---|
| resized pixels | identical (max diff 0) | identical |
| token ids, slots, M-RoPE positions, rejections | identical | identical |
| decisions | 100 % of 404 questions | 100 % |
| option logits, max diff | 1.1e-4 | 1.6e-4 |
| probabilities, max diff | 2.7e-5 | 3.6e-5 |

Three legend-object score questions are skipped (not TypeSafe wire, so the API rejects them).
The export itself matches upstream to 1.6e-5 (with an image) and 6.7e-6 (text only) on ONNX Runtime.

## Speed (measured in the runtime, five questions)

| Image | Visual tokens | RTX 4090 | CPU |
|---|---|---|---|
| 256x240 | 64 | 200 ms | 3.1 s |
| 512x512 | 256 | 390 ms | 6.7 s |
| 640x480 | 300 | 400 ms | 6.9 s |
| 1000x750 | 713 | 0.9 s | 16 s |
