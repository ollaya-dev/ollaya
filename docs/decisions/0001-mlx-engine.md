# 0001. An MLX engine for Apple silicon GPUs

- **Status:** accepted for development. The engine is built behind the `mlx` cargo feature, which is
  off by default and in no release. Shipping it waits on the owner's decision to raise the minimum
  macOS to 14.0 ([To ship it](#to-ship-it)).
- **Date:** 2026-09-26.
- **Scope so far:** phases 1 (foundation) and 2 (ModernBERT) of the plan below.

## Context

On a Mac, Ollaya runs every model on the CPU through ONNX Runtime. Both of ONNX Runtime's Mac GPU
paths were measured and rejected: the Core ML provider fails to initialise in MLProgram form and is
slower with wrong outputs in NeuralNetwork form, and the WebGPU provider fails parity (1 of 483
`laya:en` decisions differ, probabilities up to 0.13 apart) while being slower than the CPU.

Apple's [MLX](https://github.com/ml-explore/mlx) runs arrays on the Apple GPU. It does not run ONNX
graphs, so every architecture has to be written again, but it reads the author's safetensors
directly. Ollama moved its Apple backend to MLX in 0.19. A research pass (2026-09-25, M4 Pro Mac
mini, MLX 0.32.2) compared the ways to use it from Rust and ran fp32 parity for ModernBERT,
DeBERTa-v3 and Qwen3.5 with its own model code:

| Configuration | Decisions | Max probability difference |
|---|---|---|
| ModernBERT, MLX GPU fp32, own code | 483/483 | 4.0e-5 |
| ModernBERT, MLX fp16 / bf16 | 482 / 481 | 2.6e-2 / 8.5e-2 |
| ModernBERT, mlx-embeddings fp32 (tanh GELU) | 483/483 | 8.4e-3 |
| DeBERTa-v3, MLX GPU fp32, own code | 483/483 | 7.4e-6 |
| Qwen3.5, stock mlx-lm 0.31.3 fp32 (q/k norm epsilon bug, mlx-lm#1896) | 350/350 | 1.3e-2 |
| Qwen3.5, fixed norm and a custom gated-delta kernel, fp32 | 350/350 | 5.1e-6 |

Only fp32 with our own model code passes; fp16, bf16 and the stock MLX model libraries do not.

## Decision

- **Bind [mlx-c](https://github.com/ml-explore/mlx-c), Apple's C API, from Rust** through our own
  `ollaya-mlx-sys` crate, with a small safe layer, `ollaya-mlx`. Rejected: mlx-rs (unofficial, two
  maintainers, nine months behind MLX until recently, API churn), an mlx-swift sidecar (needs
  xcodebuild and a second language), ONNX Runtime's MLX provider (needs ONNX Runtime 1.29+, results
  "not bit-identical"), and Core ML / WebGPU (measured, above).
- **Pin MLX v0.32.2 and mlx-c `ebc88f10`** (Ollama's pins), build them from sha256-checked source
  tarballs with CMake, and **link them statically**. Ollama loads `libmlxc.dylib` at run time and
  has had dlopen bugs (ollama#15479, #15642, #15648); our macOS build is arm64 only anyway.
- **fp32 only, our own code per backbone and head**, as for ONNX: each model keeps its family's
  tokenizer, layout and readout, and only the network changes.
- **Ship MLX's Metal kernels as a file**, `lib/ollaya/mlx_metal/mlx.metallib`, compiled ahead of
  time. MLX cannot embed it.
- **Order:** ModernBERT, then DeBERTa-v3, then Qwen3 and Qwen3.5.

## Design as built

### Crates

- **`ollaya-mlx-sys`** (`crates/ollaya-mlx-sys/build.rs`). Only with its `build` feature:
  - downloads MLX v0.32.2 and mlx-c `ebc88f10` and checks their sha256;
  - builds them with CMake at `CMAKE_OSX_DEPLOYMENT_TARGET=14.0` (MLX's minimum; below 26.2 it also
    leaves out the M5 "NAX" kernels, the only ones that compute fp32 as TF32), static, with
    `MLX_METAL_JIT=OFF`: kernels compiled at run time use fast math on macOS 15 and later
    (mlx#4553), the ahead-of-time metallib does not;
  - caches the install prefix under `$OLLAYA_MLX_DIR/<pins>` when set (a cold build takes about
    3.5 minutes on the M4 Pro with 8 jobs), and deletes the build tree;
  - generates the bindings with bindgen (`mlx_*` only) and links `libmlx.a`, `libmlxc.a`, the
    Metal, Foundation, QuartzCore and Accelerate frameworks, and `libclang_rt.osx.a`;
  - copies `mlx.metallib` next to the binaries (`target/<profile>/`), as `ort` does for its
    provider libraries, and warns when `MACOSX_DEPLOYMENT_TARGET` is below 14.0 (rustc's default is
    11.0, and the binary's own minimum comes from rustc, not from MLX).

  Without the feature both crates are empty, so the default build needs neither CMake nor Xcode's
  Metal toolchain, and non-Mac builds are unchanged.
- **`ollaya-mlx`**: `Array` owns one `mlx_array` and is `!Send`; the ops our models use; mlx-c's
  default error handler (print, then `exit(-1)`) replaced by one that returns MLX's message as an
  `Err`; `Worker`, the one thread per runner that owns the model's arrays and runs requests as jobs
  (Ollama's `mlxthread` does the same); a start-up self-check that loads the metallib and runs a
  kernel, with a clear error when the file is missing or broken; a refusal to start unless
  `MLX_ENABLE_TF32=0`; and MLX's buffer cache capped at 1 GiB (its default is its whole memory
  limit, and the GPU shares RAM with everything else).
- **`ollaya-runner`**, feature `mlx`:
  - `net.rs`: the laya, nli, gliclass and von engines hold a `Net`, either an ONNX Runtime session
    or an MLX network, which take the same padded batch. The ONNX path sends exactly the inputs it
    sent before.
  - `mlx/arch.rs` reads the arch layer, `mlx/weights.rs` the author's weights file (safetensors, or
    a PyTorch zip at the byte offsets the arch layer lists; nothing is unpickled at run time),
    `mlx/modernbert.rs` is the backbone and `mlx/heads.rs` the heads: laya's `DecisionModel`,
    `ModernBertForSequenceClassification`, von's `OptionMarkerScorer` and GLiClass's uni-encoder.
    Weights are upcast to fp32.
  - `mlx::LAYOUTS` lists the layouts `auto` runs on MLX: the ones whose models passed their gate
    (below).

### The arch layer

`application/vnd.ollaya.arch` is a small JSON file: the backbone's hyperparameters, the head's, and
the tensor names, derived at build time from the author's `config.json` (and, for laya,
`rl_agent_config.json`) by `convert/ollaya_convert/arch.py`, which checks every name, shape and dtype
against the weights file. It holds no weights: MLX reads the same upstream weights layer the ONNX
graph references. A model without an arch layer, or with more than one weights file, runs on ONNX
Runtime only. `package.py` adds the layer to the tags whose catalog entry has `arch`, which are the
tags that passed on Metal: `laya:en`, `laya:multilingual` and `nli:modernbert-large`. The published
registry does not have them yet.

### Devices and the runner protocol

- `metal` is a device for `OLLAYA_DEVICE`, the runner's `--device` and the parity and bench
  examples. It means MLX only: a model MLX does not run fails to load instead of falling back.
- `auto` in an `mlx` build tries MLX first when the model has an arch layer, its layout is in
  `mlx::LAYOUTS` and `mlx.metallib` is found; then CUDA, then the CPU. The runner decides; the
  daemon never initialises Metal. An MLX model answers a warm-up request while it is chosen; if
  that fails, `auto` moves on to the next device, as it does for CUDA since 0.6.1. A `-fp16` tag
  still runs fp32 on MLX.
- `mlx.metallib` is looked up in `$OLLAYA_LIBRARY_PATH/mlx_metal/`,
  `<exe>/../lib/ollaya/mlx_metal/` (tarball), `<exe>/../Resources/mlx_metal/` (app bundle), next to
  the executable, and where the build installed it.
- The runner's start-up line and `/health` gain `"engine": "onnx" | "mlx"`; `ollaya ps` shows the
  device as `metal`, and its memory counts as GPU memory.
- `ollaya_runner::prepare_process` sets `MLX_ENABLE_TF32=0` first thing in `main`, so every runner
  inherits it.

### Numerics

Each of these was measured against the goldens:

- fp32 everywhere, TF32 off, no `mx.compile`.
- GELU is exact (`erf`), as `ACT2FN["gelu"]`; mlx-embeddings' tanh form moves probabilities by
  8e-3.
- **RoPE is ours**, not `mx.fast.rope`. MLX's kernel computes `metal::fast::cos` and `sin`, whose
  error grows with the angle: on unit inputs 1.8e-4 at positions up to 512 and 2e-3 up to 8192,
  against 6e-7 for our tables. The tables are cos and sin of `p * inv_freq` in f64, rounded once to
  f32, with `inv_freq` computed in f32 exactly as transformers computes its buffer (bit for bit at
  head dim 64).
- Attention is MLX's fused SDPA with a boolean key mask. A query row that may see no key comes out
  NaN and spreads through the next layer, so the padding queries of the sliding-window layers get
  the full key mask; their outputs are never read.

## Parity on Metal

Apple M4 Pro (16-core GPU), macOS 27, the same goldens and gates as the ONNX runtime, run locally
(GitHub's macOS runners have no usable Metal GPU, runner-images#7085). "ORT CPU" is the same build's
ONNX Runtime path on the same Mac.

```sh
cargo run --release -p ollaya-runner --features mlx --example parity -- <model-dir> <goldens.jsonl> metal
# likewise parity_nli, parity_gliclass, parity_von
```

A model directory holds the family's usual files plus `arch.json` (from `arch.py`) and the upstream
weights file under its upstream name.

| Model | Goldens | Gate | Metal | ORT CPU, same Mac | On MLX |
|---|---|---|---|---|---|
| `laya:en` | 97 cases, 483 questions | ids and markers identical, same decisions, laya's answers within rounding | ids identical, 100%, max prob diff 2.5e-5, 0/483 answers off | 100%, 1.9e-4, 0/483 | yes |
| `laya:en` | goldens-all, 2,383 questions | same | 100%, max prob diff 1.4e-4 (p99 8.5e-6) | 100%, 8.0e-5 | yes |
| `laya:multilingual` | goldens-all, 2,383 questions | same | 100%, 1.0e-5 (p99 6.2e-6) | | yes |
| `nli:modernbert-large` | 97 cases, 483 questions, 1,513 rows | row scores and option logits within 1e-3, same decisions | 100%, logits 7.2e-4, prob diff 5.9e-5 | 100%, 4.1e-4, 5.2e-5 | yes |
| `gliclass-instruct-edge` (not in the library) | 97 cases, 482 questions | label logits within 1e-3, same decisions | 100%, logits 7.7e-5, prob diff 1.4e-5 | 100%, 9.1e-5, 6.9e-6 | yes (head verified) |
| `von:1.1` | 98 cases, 485 questions, 653 rows (float64 network) | row and option logits within 1e-3, same decisions | 100% decisions, prob diff 3.1e-5, **logits 2.0e-3 on one row** | **1.1e-3 on the same row** | **no** |

- **`laya:en` on goldens-all.** The maximum is one question, `preset/guard/email_dict` `jailbreak`
  (93 tokens), which is also the worst question of the ONNX path; the p99 is 8.5e-6. The FAQ's 1.1e-4 is the Python ONNX export
  against PyTorch; the Rust runtime's ONNX path is at 1.9e-4 on the 97-case set on this Mac, and at
  2.2e-4 on CUDA (`docs/distribution.md`).
- **von stays on ONNX Runtime.** The row `td/agent_trace_observability_000000` `urgency` amplifies
  rounding through the encoder (`docs/families/von.md`, "Why the goldens are fp64": relative error
  at the markers grows from 6e-7 after layer 7 to 3.7e-4 after layer 28). Every fp32 variant tried
  on Metal lands between 1.0e-3 and 4.1e-3 there (MLX's RoPE 1.0e-3, ours 2.0e-3, fp32 angles
  3.8e-3, attention with materialised scores and a precise `exp` 4.1e-3), while every other row
  stays under the gate (p99 3.3e-4). ONNX Runtime on this Mac's CPU is also over the gate on that
  row (1.1e-3; 4.4e-4 on x86 and CUDA), which `docs/families/von.md` does not yet say. The head is
  implemented and runs with `--device metal` in the parity example; von is not in
  `mlx::LAYOUTS`, and `package.py` gives it no arch layer.
- **These maxima are single ill-conditioned items**, and they move with any change in rounding
  order. The alternatives for RoPE on the same goldens:

  | RoPE | `laya:en` 97 / all | `laya:multilingual` | nli logits | gliclass-edge logits | von logits |
  |---|---|---|---|---|---|
  | `mx.fast.rope` | 7.8e-5 / 3.7e-5 | 1.1e-5 | 9.8e-4 | 1.3e-4 | 1.0e-3 |
  | f64 angles, f64 `inv_freq` | | | 7.6e-4 | 1.2e-4 | 1.9e-3 |
  | **f64 angles, f32 `inv_freq` (chosen)** | 2.5e-5 / 1.4e-4 | 1.0e-5 | 7.2e-4 | 7.7e-5 | 2.0e-3 |
  | f32 angles (transformers' fp32 arithmetic) | 2.3e-5 / 9.9e-5 | 1.2e-5 | 1.1e-3 | 1.3e-4 | 3.8e-3 |

  The chosen tables are the accurate ones; `mx.fast.rope` passes these goldens only because they
  stop at 512 to 1,024 tokens, and its error at 8,192 positions is ten times larger.

## Latency

Through the daemon (`ollaya serve` with `OLLAYA_DEVICE=cpu`, then `auto`, one model loaded at a
time): each golden case sent as is to `/api/decide` (state and all its questions, 4.7 on average;
14 of the 97 cases are edge cases the API rejects), two rounds after ten warm-up requests. Client
wall time in ms. Other jobs kept the machine at a load average of 6 to 7 during both runs, which
costs the CPU path more than the GPU, so the ratios are indicative.

| Model | ORT CPU p50 / p95 | MLX on Metal p50 / p95 | p50 speed-up | Load, CPU / Metal |
|---|---|---|---|---|
| `laya:en` (97-case goldens) | 264.6 / 416.0 | 115.7 / 189.3 | 2.3x | 2.3 s / 0.8 s |
| `laya:multilingual` (first 97 cases of goldens-all) | 114.9 / 298.5 | 41.6 / 72.1 | 2.8x | 0.8 s / 0.6 s |
| `nli:modernbert-large` (97 cases, 3.1 hypothesis rows per question) | 312.2 / 941.1 | 140.4 / 429.1 | 2.2x | 2.8 s / 0.4 s |

`gliclass-instruct-edge` is not in the library, so it was timed in process by its parity example:
97 requests in 2.91 s on the CPU and 0.97 s on Metal (30 and 10 ms per request). `ollaya ps` shows
the MLX runners as `metal`, and the daemon log `engine=mlx`.

## Sizes

- `bin/ollaya` for darwin-arm64: 41.6 MiB (the 0.6.0 release), 55.1 MiB built here with
  `ollaya-runner/coreml,ollaya-runner/mlx`. MLX adds about 14 MiB and links only system frameworks.
- `mlx.metallib`: 130.1 MiB. The `ollaya-darwin-arm64-mlx` archive is 9.4 MiB as `.tar.zst`
  (zstd -19) and 34.1 MiB as `.tgz`. The base archive with that binary is 12.2 MiB as `.tar.zst`.

## Packaging and CI (prepared, off)

- `scripts/package.sh --mlx` (darwin-arm64) builds `ollaya-darwin-arm64-mlx.tar.zst` and `.tgz`
  (`lib/ollaya/mlx_metal/mlx.metallib` with its `FILES.sha256`, and
  `share/doc/ollaya/mlx_metal/THIRD_PARTY_NOTICES`) and `ollaya-darwin-arm64-mlx.sha256`, the same
  keep-if-unchanged fingerprint as the CUDA archive. A binary built with the feature also gets
  MLX's notices in its own `THIRD_PARTY_NOTICES`: MLX, mlx-c, {fmt} and nlohmann/json (compiled in
  by MLX's build), and PocketFFT and metal-cpp from MLX's `ACKNOWLEDGMENTS.md`.
- `release.yml`: a manual run with the `mlx` input builds darwin-arm64 with the feature at
  `MACOSX_DEPLOYMENT_TARGET=14.0` and packages the MLX archive. Tag builds never do.
- `ci.yml`: an `mlx` job on `macos-latest` builds MLX (cached by the pins), runs clippy on the
  workspace and the runner's and daemon's unit tests with the feature, and builds the binary. It
  cannot run the GPU tests or Metal parity.
- No self-hosted runner. GPU tests (`cargo test -p ollaya-mlx --features build`) and the parity
  runs above happen on a Mac by hand.

## To ship it

When the owner accepts macOS 14.0 as the minimum on Apple silicon:

1. `release.yml`: give the darwin-arm64 matrix entry `features: ollaya-runner/coreml,ollaya-runner/mlx`
   and `package-args: --mlx`, set `MACOSX_DEPLOYMENT_TARGET: '14.0'` for it, add the MLX build cache
   and the Metal toolchain step to it unconditionally, and drop the `mlx` input. The "Minimum macOS
   version" step then shows `minos 14.0`.
2. `scripts/install.sh`: on macOS, download `ollaya-darwin-arm64-mlx.tar.zst` next to the base
   archive into `$PREFIX/lib/ollaya/mlx_metal`, skipping it when `ollaya-darwin-arm64-mlx.sha256`
   matches the installed `FILES.sha256` (the CUDA logic), and refuse macOS below 14.0.
3. The app (`desktop.yml`, `desktop/src-tauri/tauri.conf.json`): build the engine with
   `ollaya-runner/mlx` and `MACOSX_DEPLOYMENT_TARGET=14.0`; add `target/release/mlx.metallib` to the
   bundle's resources as `mlx_metal/mlx.metallib` (it lands in `Contents/Resources/mlx_metal/`,
   where the runner looks); raise `bundle.macOS.minimumSystemVersion` from 11.0 to 14.0. The
   metallib is data, sealed by the app's signature; notarization covers it.
4. The registry: run `package.py` for `laya` and `nli` to add the arch layers (older clients
   download and ignore them), and publish.
5. `docs/distribution.md` and the site: the MLX archive, the macOS 14 minimum, and `metal` in the
   device documentation.

## Risks

- **MLX is 0.x.** Minor releases break APIs. Upgrade the pins only through the parity runs above.
- **Fast math.** Anything compiled at run time (`MLX_METAL_JIT`, `mx.compile`, custom Metal kernels)
  uses approximate math on macOS 15 and later. Qwen3.5's gated-delta kernel (phase 4) is a custom
  kernel and needs its own parity check.
- **TF32.** Handled by `MLX_ENABLE_TF32=0` and the 14.0 deployment target (no NAX kernels); a
  future build that targets 26.2 for the M5 kernels must keep TF32 off.
- **CI has no Metal.** A change can pass CI and break Metal parity. Run the parity examples on a
  Mac before merging anything under `crates/ollaya-mlx*` or `crates/ollaya-runner/src/mlx`.
- **Build supply chain.** MLX's CMake fetches metal-cpp, {fmt} and nlohmann/json by tag or URL at
  build time; only the MLX and mlx-c tarballs are checked by sha256.
- **Memory.** Weights are upcast to fp32 in unified memory (`laya:en` about 1.7 GB); the parity
  runs peak at 3.0 to 5.3 GB of memory footprint. A runner caps MLX's buffer cache at 1 GiB. Qwen3.5 2B in fp32 needs about
  7.6 GB.
- **Sensitive rows.** von shows that an fp32 engine can differ from the float64 reference by more
  than 1e-3 on a chaotic input; the gate is kept, so von stays on ONNX Runtime.
