# ADR-0001: Keep large decoder checkpoints BF16 in memory

## Status

Accepted

## Date

2026-09-26

## Context

The Qwen3.5 decoder families (`decider`, `kev`) ship as fp32 ONNX graphs whose weights are external references
into the author's BF16 safetensors. Each weight enters the graph through `Cast(bf16 -> f32)`. The outputs are
fp32 computations on the exact BF16 values, and the parity gate (1e-3 on logits, identical decisions) holds only
for fp32 compute: real fp16 or bf16 compute is out.

ONNX Runtime folds constant subgraphs when the session loads, so today every weight is widened to fp32 once and
kept that way. That doubles the memory of the checkpoint:

| checkpoint | BF16 weights used by the graph | fp32 after folding |
|---|---|---|
| decider-2b | 3.8 GB | 7.5 GB |
| decider-4b, kev-4b (Qwen3.5-4B) | 8.4 GB | 16.8 GB |
| kev-9b (Qwen3.5-9B) | 15.9 GB | 31.7 GB |

Two facts found while measuring:

- **ORT folds a node only if its output is at most 1 GiB**
  (`optimization.constant_folding_max_output_size_in_bytes`). The token embedding widened to fp32 is 2.0 GB for
  decider-2b and 2.5 to 4.1 GB for the larger bases. So the shipped decider-2b graph already widens its whole
  embedding at every forward pass, and its label head reads the widened embedding again through an identity
  `Transpose` and a `GatherND`.
- **The CPU `Cast` from BF16 is single-threaded** (Eigen); FP16 has a parallel MLAS path, BF16 does not.

## Decision

1. **Gather before Cast (export).** The weightless rewrite moves each widening `Cast` below the `Gather` and
   `GatherND` nodes that read rows of a BF16 tensor: `Gather(Cast(E), ids)` becomes `Cast(Gather(E, ids))`. Only
   the gathered rows are widened. The values are identical, since `Cast` is elementwise.
   (`weightless_sharded.sink_casts_below_gathers`, applied to every decoder export from now on.)
2. **`decision.json` `weights_in_memory`.** `"fp32"` (the default, and what older graphs mean) keeps today's
   folding. `"bf16"` makes the runner set the ORT folding limit to 1 MiB: norms, biases and masks still fold,
   every weight matrix stays BF16 in memory, and each forward pass widens one layer's weights just before they
   are used (ORT runs the casts interleaved with the layers, so only one layer's fp32 copies are alive).
3. **Who gets `"bf16"`.** The exporter writes `"bf16"` when the graph uses more than 6 GiB of BF16 checkpoint
   tensors: the 4B and 9B bases. Models up to 2B params stay `"fp32"`, where it fits and is faster on the GPU.
4. **Exact CUDA arena growth for decoders.** The decider and kev sessions grow ORT's CUDA arena by the size requested
   (`ArenaExtendStrategy::SameAsRequested`) instead of doubling it; see "The CUDA arena" below.

The switch is per model and the same on every device. Measured below, `"bf16"` costs 10 to 13% per request on
the GPU and on x86 CPUs (decider-2b on the Mac mini: none). In exchange a 4B loads in 2 s instead of 27 s, without
a 27 GiB host-memory spike while ORT folds, and holds 14 GiB of GPU memory instead of 21; and a 9B fits a 24 GB GPU at
all. Folding a 4B on a larger GPU would be faster, but the runner cannot see free device memory before it loads a
model, and a model that does not fit spills into system memory under Windows and WSL (measured below: over tenfold
slower) rather than degrading gracefully.

## Measurements

All runs use the runner's `bench` example on typed-decisions requests from the goldens (five questions each,
about 1,400 tokens per request): `--ids td/ --full-only`, 20 requests twice on CUDA, 8 to 20 on the CPU (choso-wsl:
RTX 4090 24 GB, i9-13900K, WSL 2; Mac mini M4 Pro 24 GB). "fp32" is the ONNX Runtime default (casts folded at load),
"bf16" is `weights_in_memory: bf16`. Host peak is the process's maximum RSS; GPU is the device's peak from `nvidia-smi`
minus what other processes held, in GiB. On WSL, what the GPU cannot hold spills into shared system memory (Windows
`GPU Adapter Memory\Shared Usage`).

**decider-2b** (1.9B), graph with the Gathers moved above the Casts:

| device | mode | load | host peak | GPU | p50 | p95 |
|---|---|---|---|---|---|---|
| RTX 4090 | fp32 | 13.6 s | 11.3 GiB | 8.6 GiB | 907 ms | 1,436 ms |
| RTX 4090 | fp32, exact arena | 10.9 s | 11.3 GiB | 9.4 GiB | 918 ms | 1,428 ms |
| RTX 4090 | bf16 | 2.4 s | 2.3 GiB | 8.6 GiB | 1,028 ms | 1,609 ms |
| RTX 4090 | bf16, exact arena | 1.9 s | 2.3 GiB | 6.8 GiB | 1,017 ms | 1,599 ms |
| Mac mini CPU | fp32 | 6.5 s | 16.0 GB footprint | | 5,552 ms | 9,017 ms |
| Mac mini CPU | bf16 | 1.8 s | 9.0 GB footprint | | 5,418 ms | 8,324 ms |
| i9-13900K CPU, 8 requests, box shared with another job | fp32 | 11.4 s | 12.4 GiB | | 9,038 ms | 22,811 ms |
| i9-13900K CPU, same | bf16 | 1.2 s | 9.8 GiB | | 10,160 ms | 24,096 ms |

The shipped decider-2b graph (without the Gather rewrite) takes 13.6 GiB of GPU memory at p50 927 ms, and 19.8 GiB in
bf16 mode: it widens and copies the 2 GB embedding at every run. The rewrite alone saves 5 GiB there.

**decider-4b** (4.2B):

| device | mode | load | host peak | GPU | p50 | p95 |
|---|---|---|---|---|---|---|
| RTX 4090 | fp32, exact arena | 26.5 s | 27.2 GiB | 20.8 GiB | 2,059 ms | 3,164 ms |
| RTX 4090 | bf16, exact arena (shipped) | 2.2 s | 2.1 GiB | 14.3 GiB | 2,325 ms | 3,528 ms |
| RTX 4090 | bf16, default arena | parity run: 24.0 GB on the device and 4.1 GB spilled, over tenfold slower; stopped | | | | |
| i9-13900K CPU | fp32 | 29.5 s | 27.0 GiB | | 22.1 s | 57.3 s |
| i9-13900K CPU | bf16 (shipped) | 1.2 s | 21.5 GiB | | 24.3 s | 63.9 s |

On the GPU the folded 4B runs these requests with 3 GiB to spare; longer states spill. On the CPU most of the peak
is activations: attention scores grow with the square of the row length, and the CPU arena keeps its high-water mark.

**kev-4b and kev-9b** (bf16, exact arena, shipped): kev-4b loads in 2.7 s, holds 11.0 GiB of GPU memory, p50 981 ms;
kev-9b loads in 4.4 s, holds 17.9 GiB, p50 1,346 ms. Folded, kev-9b's weights alone (31.7 GB) would exceed the card.
Its parity run (rows up to 2,033 tokens) peaked at 22.0 GiB of the 24.0: a 24 GB GPU is its minimum, and states
much longer than that spill.

**The CUDA arena.** ORT's default arena grows by doubling. The decoder graphs allocate large buffers of many sizes on
every run (attention scores grow with the square of the row length; with BF16 kept, one layer's widened weights),
and on decider-4b the doubling outgrew the 24 GB card. With exact growth decider-4b's parity run peaked at 17.4 GiB
with no spill and took 4.5 minutes. decider-2b's CUDA parity is unchanged with it (identical numbers).

## Alternatives considered

- **Keep folding (status quo).** Simplest and fastest where it fits. A 4B folded holds 17 GB of weights on the
  device and spikes to 27 GiB of host memory while ORT folds; a 9B (32 GB) does not fit a 24 GB GPU or a 24 GB Mac
  at all. Rejected for 4B and up; kept for 2B and smaller.
- **Disable `ConstantFolding` entirely** (`with_disabled_optimizers`). Also keeps weights BF16, but loses the
  folding of the export's shape arithmetic and small constants. The size limit keeps those. Rejected.
- **Make the casts unfoldable in the graph** (list the BF16 initializers as overridable graph inputs). Works on
  any runtime, but changes the graph contract (hundreds of optional inputs) for what one session option does.
  Rejected.
- **fp16 or bf16 compute.** Half the memory and faster, but it breaks the parity tolerance. Rejected by the parity
  rule.
- **MLX on Apple silicon.** Reads BF16 directly, about 3x the ORT CPU speed for encoders; a separate engine,
  planned after v0.6.0. Complementary, not a replacement on Linux and
  Windows.

## Consequences

- decider-4b, kev-4b and kev-9b ship with `"weights_in_memory": "bf16"`; decider-0.8b, decider-2b, kev-0.8b and
  qwen3guard keep `"fp32"` (no field) and behave exactly as before.
- Loading is 5 to 10 times faster in `"bf16"` mode, since there is nothing to fold.
- On memory pressure the OS can drop the BF16 pages (they are clean, file-backed) and read them again; a machine
  that swaps loses much more time to that than to the casts (measured on the Mac mini under another job's
  17 GB load: casts took 67% of the time instead of 16%).
- The shipped decider-0.8b/2b and qwen3guard graphs predate the Gather-before-Cast rewrite. Re-exporting them
  would cut decider-2b's GPU memory from 13.6 to 8.6 GiB and its CUDA p50 by about 2%; that needs new graph blobs
  and a parity run, and is left as a follow-up.
- Every decider and kev session on CUDA grows its arena exactly; qwen3guard and the encoders keep ORT's default.
- The runner ignores `weights_in_memory` for layouts other than `decider-slots-v1` and `kev-pointer-v1`.
