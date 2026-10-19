# ADR-0008: fp16 tags for the Qwen3.5 decoders, and `Split` instead of `SplitToSequence`

## Status

Accepted

## Date

2026-10-19

## Context

Ollama 0.40.6 serves the same Kev, Nimble and Clef weights 4 to 8 times faster than Ollaya 0.13 on one RTX 5090
(`docs/measurements/2026-10-ollama-0.40.md`). It runs Q8_0 GGUFs on llama.cpp; Ollaya computes in fp32 on the
authors' BF16 weights, which ADR-0002 kept because only fp32 passes the parity gate (logits within 1e-3, every
decision the same as the author's code).

A profile of a first fp16 export of `kev:4b` on an RTX 4090 (ONNX Runtime's session profiler, five-question
requests, 206 ms of node time per request) showed where the time went:

| Node | Device | ms per request | share |
|---|---|---:|---:|
| `SplitToSequence` (32, one per layer) | CPU | 38.6 | 19% |
| `MemcpyFromHost` (the copies back) | GPU | 37.1 | 18% |
| `MatMul` (1,057) | GPU | 27.0 | 13% |
| `Scan` (24, the gated delta rule) | GPU | 17.7 | 9% |

The fp32 graphs have the same nodes.

`torch.split` and `torch.chunk` export as `SplitToSequence` followed by `SequenceAt`. ONNX Runtime has no CUDA kernel
for `SplitToSequence`, so every layer copied its activations to the host, split them there and copied them back.

## Decision

1. **`Split` instead of `SplitToSequence` (every Qwen3.5 graph).** `onnx_export.sequence_splits_to_split` rewrites a
   `SplitToSequence` whose outputs are read only by `SequenceAt` at constant positions into one `Split`; the readers
   become `Identity`. The values are the same, so the fp32 tags keep their parity numbers. The export runs it on every
   new graph; `python -m ollaya_convert.families.llm_common.onnx_export out/<model>` applies it to an exported one.
2. **fp16 tags.** `qwen35.half_trunk` exports the trunk with fp16 weights and activations. The parts that need range
   stay fp32, as the authors' code computes them: the RMSNorms, the gated delta rule and the attention softmax (HF's
   modules and `qwen35.py` already upcast them), the rotary frequencies and the doubling masks (fp16 positions are
   off by whole radians at 8,000 tokens), and every head (pointer, readout, LM-head rows). The weights stay the
   authors' BF16 files, narrowed by `Cast` as ADR-0002 widens them; `weights_in_memory` is unchanged.
3. **A gate of their own.** An fp16 tag is a new tag (`kev:4b-fp16`), never a change to the fp32 one. Its parity run
   (`parity_* --fp16`) keeps the encoding checks exact, requires at least 99.5% of decisions to match the fp32
   reference, and reports the logit and probability differences instead of bounding them.
4. **GPU tags.** An fp16 manifest carries only the fp16 graph. It is meant for NVIDIA GPUs; on a CPU it runs, slowly,
   as Laya's `-fp16` tags do.

## Results

Measured for 0.14.0 (`results/runs/2026-10-19-*`): parity on the RTX 4090; typed-decisions, the median five-question
request over HTTP and the GPU memory on the RTX 5090 of Ollaya Bench 2026.10. The fp32 typed-decisions numbers are the
edition's; the fp32 graphs' outputs did not change.

| Tag | Decisions the same as fp32 (parity, RTX 4090) | Typed-decisions, fp16 / fp32 | Five questions, RTX 5090: fp16 / fp32 | GPU memory, fp16 / fp32 |
|---|---|---|---|---|
| `kev:4b-fp16` | 478 / 480 (99.58 %) | 0.669 / 0.669 | 98 / 171 ms | 10.8 / 10.9 GB |
| `kev:9b-fp16` | 480 / 480 (100.00 %) | 0.723 / 0.722 | 125 / 248 ms | 17.6 / 17.9 GB |
| `nimble:9b-fp16` | 492 / 492 (100.00 %) | 0.665 / 0.665 | 428 / 1015 ms | 19.7 / 20.1 GB |
| `clef:flash-fp16` | 570 / 571 (99.82 %) | 0.703 / 0.703 | 150 / 268 ms | 19.4 / 19.6 GB |
| `pplx-decider:flash-fp16` | 695 / 696 (99.86 %) | 0.726 / 0.726 | 90 / 126 ms | 10.3 / 10.4 GB |
| `strands-decider:2b-fp16` | 649 / 650 (99.85 %) | 0.615 / 0.616 | 72 / 98 ms | 5.8 / 8.5 GB |
| `kev:0.8b-fp16` (not shipped) | 478 / 480 (99.58 %) | 0.461 / 0.460 | 67 / 62 ms | 3.6 / 4.5 GB |

Typed-decisions accuracy is within 0.002 of the fp32 tag's for every fp16 tag. `kev:0.8b-fp16` passed its gate but is not
shipped: once `Split` replaced `SplitToSequence`, the fp32 graph of the 0.8B model was as fast.

`Split` alone made every fp32 graph faster, with the same parity numbers (median five-question request on the RTX 5090):

| Tag | 0.13.0 | 0.14.0 |
|---|---:|---:|
| `decider:0.8b` | 189 ms | 60 ms |
| `decider:2b` | 193 ms | 91 ms |
| `decider:4b` | 393 ms | 207 ms |
| `decision:eos` | 209 ms | 69 ms |
| `kev:0.8b` | 142 ms | 62 ms |
| `kev:4b` | 319 ms | 171 ms |
| `kev:9b` | 375 ms | 248 ms |
| `clef:flash` | 406 ms | 268 ms |
| `jeeves:9b` | 641 ms | 380 ms |
| `nimble:9b` | 1688 ms | 1015 ms |

## Alternatives considered

- **Q8_0 on llama.cpp (Ollama's path).** The fastest, but it needs a GGUF built on the user's machine from the
  author's safetensors (Ollaya never re-hosts weights) and custom code for the pointer heads. Not now.
- **BF16 compute.** Same memory as fp16; ONNX Runtime's CUDA kernels for BF16 cover fewer operators, and BF16's
  8-bit mantissa moves more decisions than fp16's 11.
- **fp16 as the default graph of the existing tags** (as `laya:en` does). The fp32 tags are what matches each
  author's code; a user who wants the speed asks for it by name.
- **TensorRT RTX EP.** JIT-compiles per GPU, with its own numerics; a separate engine to maintain (ADR-0004).

## Consequences

- Every Qwen3.5 graph in the registry changes (same values, new blobs); `ollaya pull` fetches the new graph, the
  weights stay.
- The library gains the `-fp16` tags listed above; their pages give the decision agreement and the speed.
- ADR-0002's "fp16 or bf16 compute: rejected" still holds for the fp32 tags.
