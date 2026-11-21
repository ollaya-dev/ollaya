# valen (`valen-decision-1.5-v1`)

Valen-Team's **Valen-Decision-1.5** models
([0.8B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-0.8B), [2B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-2B),
[4B](https://huggingface.co/Valen-Team/Valen-Decision-1.5-4B), Apache-2.0) are fully post-trained Qwen3.5
vision-language models with a **Mixer candidate head**. One sequence holds the state (text and an image) once and
every question after it (upstream's `shared_state` execution); the head scores each candidate from four pooled
roles. They never generate text, and the probabilities are a plain softmax of the head's logits (temperature 1,
nothing fitted). The model repositories hold the authors' inference code next to the weights: `modeling_valen.py`,
`compiler.py` (the request → sequence layout), `schema.py`, `block_attention.py`, `heads.py`, `runtime_model.py`
and `responses.py`, loaded with `trust_remote_code`. The training code is at
[github.com/Valen-Team/Valen](https://github.com/Valen-Team/Valen).

This family is Valen-Decision-1.5 only. Valen-Decision-1.0 has another attention pattern (causal) and other
limits; it would be its own layout.

| Tag | Weights | Status in Ollaya |
|---|---|---|
| `valen:0.8b` | `Valen-Team/Valen-Decision-1.5-0.8B@e079395f`: Qwen3.5-0.8B, one BF16 shard (1.7 GB) and the F32 heads | **converted, ONNX** |
| `valen:2b`, `valen:latest` | `Valen-Team/Valen-Decision-1.5-2B@1b890606`: Qwen3.5-2B, one BF16 shard (4.5 GB) | **converted, ONNX** |
| `valen:4b` | `Valen-Team/Valen-Decision-1.5-4B@9b0cacc9`: Qwen3.5-4B, two BF16 shards (9.1 GB) | **converted, ONNX** |
| `Valen-Team/Valen-Decision-1.5-9B` | Qwen3.5-9B, four shards | not converted yet: same architecture, see Limits |
| `Valen-Team/Valen-Decision-1.5-27B` | Qwen3.8-27B | not converted: it does not fit the parity GPU |

## Engine: Ollaya's own ONNX export

- **Why not GGUF.** The authors publish GGUF files (`-GGUF` repositories), but they need the authors' patched
  llama.cpp runtime (the Mixer head, the grounding head and the block mask are not in llama.cpp), so stock
  llama.cpp cannot run them and ADR 0003 does not apply.
- **Why not the authors' ONNX bundles.** The `-ONNX` repositories hold modular bundles (`embedding`, `vision`,
  `decision`, `grounding` graphs plus a `valen_onnx.json` manifest) for the authors' SDKs, which build the block
  mask and run the graphs in sequence. Their weights are a second, converted copy (fp32: 3.5 GB for 0.8B against
  the 1.7 GB BF16 checkpoint), and their contract is the SDKs'. Ollaya's export instead references the
  checkpoint's own safetensors by byte offset, so the weights stay the pinned, sha256-checked originals, and it
  follows the contract the other Qwen3.5 families use (one decoder graph per request, a vision graph).
- **Export.** `convert/ollaya_convert/families/valen/export.py`: the vision tower is decider-vision's
  `VisionGraph` (the same Qwen3.5 tower); the decoder reuses `llm_common/qwen35.py` (the scan-based Gated DeltaNet
  export) with interleaved M-RoPE and the block mask as an input; the head is the authors' `MixerHead` module
  with its weights. The export checks itself against the authors' code, in PyTorch and then on ONNX Runtime.

## Request → sequence

Source: `Compiler._messages`, `Compiler._compile_shared`, `schema.candidates`, `schema.fold_noul_criteria` and the
Qwen chat template, at the pinned revisions. Port: `families/valen/layout.py` (Python) and
`crates/ollaya-decision/src/valen.rs` (Rust).

```text
ids = tok("<|im_start|>user\n" + strip(image + state) + "<|im_end|>\n")          the shared prefix
      + tok("<|im_start|>user\n")
      + for question i: tok("Question {i}\nTask: {type}\nQuestion: " + instructions + "\nCandidates:\n")   instruction span
                        + for each candidate: tok([key + ": "] + description)                            candidate span
                                              + tok("\n")
      + tok("<|im_end|>\n<|im_start|>assistant\n")
      + for question i: tok("Question {i} Decision:")   (its last token is the decision position)
                        + tok("\n")
```

- **State.** A string is used as is. Anything else is written as `json.dumps(ensure_ascii=False, sort_keys=True)`,
  as the authors' benchmark harness feeds structured states; upstream's `predict` itself takes only a string or a
  message list. `strip` is the chat template's `|trim`.
- **Image.** `image` is `<|vision_start|>` + `<|image_pad|>` × n + `<|vision_end|>`, n the image's merged visual
  tokens, before the state text in the same user message (upstream `{"messages": [{"role": "user", "content":
  [image, text]}]}`). One image per request.
- **Candidates.** noul reads two fixed candidates, `true` then `false`, with fixed descriptions; its criteria,
  when given, must have exactly the keys `true` and `false` and are appended to the instructions as
  `"\nTrue criterion: " + str(true) + "\nFalse criterion: " + str(false)` (Python's `str`). A choice's candidates
  are `key: description` in the caller's order (1 to 255); a score's are its level descriptions alone (2 to 255).
  Every description must be a non-blank string.
- **Spans.** Each piece after the prefix is tokenized on its own. A span is the tokens whose characters overlap
  the instructions (or the description), read from the tokenizer's character offsets as upstream does.
- **Rejections.** Text that holds one of the tokenizer's special tokens (`<|im_end|>`, `<|endoftext|>`, the
  vision and audio markers) is rejected, as upstream does; other added tokens (`<think>`, `<|fim_prefix|>`) are
  ordinary text. Nothing is truncated: a sequence over 16,384 tokens (`valen_max_length`) is rejected.
- **Order.** The head scores noul as `[true, false]`; the runtime returns `[false, true]`, the order answers use.

## Attention blocks and positions

Upstream's `bbox_blocks` mask (`block_attention.py`) applies to the full-attention layers only; the Gated DeltaNet
layers stay causal. Each token gets a block: the prefix (−1), the shared glue (`<|im_start|>user\n` before the
questions and the assistant header before the readouts, 0), and one block (1) for every decision question and
its readout together. A prefix token sees the prefix only; any other token sees the prefix, the glue and its own
block. Padding is never seen; a padding query sees the prefix so that its row stays finite.

Positions are interleaved M-RoPE (sections [11, 11, 10], partial rotary 0.25), as upstream's `get_rope_index`:
text counts up on all three axes; the image's merged cell (i, j), starting at position s, is (s, s + i, s + j),
and the text after it continues at s + max(rows, columns).

## The head

The authors' `MixerHead`, F32: for every candidate four roles are pooled from the last hidden states, the mean of
the prefix, the mean of the question's instruction span, the mean of the candidate's span and the question's
decision token. A LayerNorm and a shared projection (hidden → 256) plus learned role embeddings, two Mixer blocks
(a 4 → 16 → 4 MLP across roles, a 256 → 1,024 → 256 MLP across channels) and a normalized linear readout of the
candidate role give one logit. The graph pools every candidate of every question at once with span-mask products.

## ONNX contract

| Graph | Inputs | Output |
|---|---|---|
| `vision.onnx` | `patches` [N, 1536], `pos_idx` [N, 4], `pos_w` [N, 4], `rot_ids` [N, 2] | `image_embeds` [N/4, hidden] |
| `model.onnx` | `input_ids` [1, T], `position_ids` [3, 1, T], `image_embeds` [M, hidden], `image_pos` [M], `blocks` [1, T], `instruction_spans` [Q, 2], `decision_pos` [Q], `candidate_spans` [O, 2], `candidate_question` [O] | `logits` [O] |

`T` is a multiple of 64 with at least one padding position (right padding, `blocks` −2). Without an image, one
zero embedding is written at the last padding position. The vision inputs are decider-vision's
(`crates/ollaya-runner/src/vision.rs`). Both graphs are weightless: every tensor references the author's
safetensors (BF16 backbone widened by a Cast, F32 head); the grounding head's tensors are unused.

## Image preprocessing

PNG only, Qwen's `Qwen2VLImageProcessorPil` reproduced value for value (`vision.rs`, see
[decider-vision.md](decider-vision.md)), with the model's own size limits: between 65,536 and 1,638,400 pixels
(`valen_media_kwargs`), so a larger image is resized down as upstream resizes it. Upstream's processor defaults to
the torchvision backend, whose resize differs from PIL's by one level on a few pixels; the reference and the
goldens use the PIL backend, the one the Rust port reproduces exactly.

## Parity

Goldens: `families/valen/goldens.py`, the authors' code in fp32 (TF32 off) on CUDA, through `predict`'s own path
(`Compiler.compile`, `forward_state`). Cases: the shared edge cases and 40 typed-decisions rows, Valen's own text
cases (every branch of the compiler: whitespace and blank states, structured states, noul criteria of every kind,
span edges, 60 options, a long state, a request over 16,384 tokens, special tokens in questions, blank and typed
descriptions) and synthetic PNGs (decider-vision's scenes plus one above the pixel maximum). Every golden row
(token ids, the prefix, every token's block, the M-RoPE positions, every span and decision position) is checked
against `layout.py` before it is written.

`crates/ollaya-runner/examples/parity_valen.rs` checks the rejections, the resized pixels and the merged grid,
the whole sequence, then the candidate logits (tolerance 1e-3) and the decisions:

```sh
cargo run --release -p ollaya-runner --example parity_valen -- convert/out/valen-0.8b convert/out/goldens-valen-0.8b.jsonl cpu
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_valen -- convert/out/valen-0.8b convert/out/goldens-valen-0.8b.jsonl cuda
```

Measured on one NVIDIA H200 host (CUDA) and its 32 CPU cores, goldens and both runs on the same machine and build,
13 November 2026: 176 golden records, 30 rejected upstream and by the runtime alike, 146 run (15 with an image),
642 questions, 71,575 tokens.

| | 0.8B CPU | 0.8B CUDA | 2B CPU | 2B CUDA | 4B CPU | 4B CUDA |
|---|---|---|---|---|---|---|
| rejections, resized pixels, token ids, blocks, positions, spans | identical | identical | identical | identical | identical | identical |
| decisions | 100 % of 642 | 100 % | 100 % | 100 % | 100 % | 100 % |
| candidate logits, max diff | 1.7e-4 | 1.4e-4 | 1.7e-4 | 1.9e-4 | 1.2e-4 | 1.2e-4 |
| probabilities, max diff | 3.0e-5 | 1.4e-5 | 2.8e-5 | 2.4e-5 | 2.5e-5 | 3.3e-5 |

The tolerance is 1e-3, unchanged. Only the full set was run; `--smoke` and the other devices Ollaya ships
(Windows CUDA, Vulkan) were not.

## Limits

- **bbox.** Valen-Decision-1.5 also answers `bbox` questions (where something is in an image) with its grounding
  head. Ollaya's API has no question type for a box, so the grounding head is not exported and a `bbox` question
  is rejected by the shared question parser. A request without bbox questions is laid out exactly as upstream
  lays it out.
- **Images.** One PNG per request (no JPEG, no video). Upstream reads several images, video and multi-turn
  message lists.
- **Questions.** Ollaya's shared question rules apply first: `instructions` is required, choice criteria are an
  object, score criteria a list.
- **9B.** Valen-Decision-1.5-9B has the same architecture and layout; it is not converted in this change.
- **Memory.** The 4B keeps its weights BF16 in memory (`weights_in_memory`); rows grow with the image and the
  number of questions, up to 16,384 tokens.

## Attribution

Valen-Decision-1.5 is by Valen-Team, released under Apache-2.0. The backbones, Qwen3.5-0.8B, Qwen3.5-2B and
Qwen3.5-4B, are by the Qwen team (Apache-2.0).
