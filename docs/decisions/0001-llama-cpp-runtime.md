# 0001: How Ollaya runs GGUF models on llama.cpp

- Status: accepted, 2026-09-26 (issue #3)
- Applies to: `winnow-v1` and `llm-logits-v1` (`docs/families/winnow.md`, `docs/families/llm-logits.md`)

## Context

The open decision models closest to Jev are LLM fine-tunes whose authors publish only GGUF files
(Winnow-12B, Winnow-E4B). Ollaya needs to run them the way it runs its ONNX families: the author's
file, unmodified and pinned by sha256; a runner subprocess per model; the same numbers on every
request; CPU and GPU on Linux, macOS and Windows; and install packages that keep unchanged files on
upgrade (`FILES.sha256`, `docs/distribution.md`).

What a decision needs from llama.cpp is small: tokenize with the GGUF's vocabulary, evaluate a
state prefix once, evaluate each question's suffix after it, and read the logits of a handful of
label tokens at the last position, with a fixed evaluation plan (llama.cpp's numbers depend on how
a prompt is split into batches; `docs/families/llm-logits.md`, "Determinism").

## Options

| | A: `llama-server` subprocess | B: libllama inside Ollaya's runner | C: vendored or patched source |
|---|---|---|---|
| Label logits | No raw logits: top-n or a `logit_bias` + sampler trick, fp32 resolution about 1e-5 | Exact, `llama_get_logits_ith` | Exact |
| Fixed plan | Needs a request protocol (prefix, barrier, question) and `timings.cache_n` checks | By construction: `llama_memory_seq_rm` back to the prefix, one batch per suffix | Same as B |
| Contract | CLI flags, REST JSON, log lines; no stated compatibility rule | `include/llama.h`, under semver since v0.2.0 (2026-08-21), with an API/ABI check per release | Our own fork |
| Crash isolation | Process | Process (Ollaya runners are already subprocesses) | Process |
| Host libraries (Linux) | OpenSSL 3 for `llama-server`, plus libgomp | libgomp only | Our choice |
| Builds in Ollaya CI | None (prebuilt) | None (prebuilt libraries) | Every backend, CUDA included |

What other projects do (sources in the evidence list below):

- **Ollama** switched to an upstream `llama-server` subprocess in May 2026 (PR #16031, removing
  430k lines of vendored code), built from source with patches it calls short-lived, and it reads
  GPU placement by parsing llama-server's log lines. It wants the whole chat server; Ollaya does not.
- **Jan** moved from A to B in August 2026 (PR #8762): libllama and ggml in a worker process for
  crash isolation and direct control of the KV state, pinned to a semver tag.
- **Lemonade, Docker Model Runner, RamaLama, llama-swap, text-generation-webui** use A.
  text-generation-webui's `get_logits` reads a top-128 list, an approximation of what Ollaya needs.
- **GPT4All, koboldcpp, llamafile, LocalAI** use C; GPT4All has stalled on its fork.
- **Rust bindings.** `llama-cpp-2` is the one maintained binding; it compiles llama.cpp (CUDA
  included) in the consumer's build from a master-commit submodule. Ollaya calls 30 functions,
  so it owns a thin FFI instead (`crates/ollaya-runner/src/llama/ffi.rs`).

## Decision

**B, distributed from upstream's own release binaries.**

1. **Pin.** llama.cpp **v0.5.0**, whose binaries are release build **b11146** (commit `7fe450e`).
   `scripts/llama-cpp.sh` downloads the release archives and verifies each against the sha256
   GitHub publishes for the asset, pinned in the script. Releases are mutable on GitHub, so the pin
   lives in Ollaya. Only libllama, libggml, libggml-base and the ggml backends are staged: no
   `llama-server`, no tools, no symbolic links.
2. **Linux x86-64.** Everything comes from the `ubuntu-cuda-13.4-x64` archive, so the CPU and CUDA
   backends are one build (glibc 2.38, Ollaya's existing floor).
   - `lib/ollaya/llama/`: libllama, libggml, libggml-base and the 14 CPU variants, of which ggml
     loads the best one for the CPU (`GGML_BACKEND_DL`, `GGML_CPU_ALL_VARIANTS`).
   - `lib/ollaya/cuda_v13/libggml-cuda.so` in the CUDA pack, next to the `libcudart`, `libcublas`
     and `libcublasLt` that pack already ships for ONNX Runtime. Upstream's separate 440 MB cudart
     archive is not needed, and a llama.cpp bump changes one 159 MB file of the pack.
3. **macOS arm64.** The `macos-arm64` dylibs: Metal is built into libggml with its shaders embedded,
   minimum macOS 13.3. The desktop app signs them with Ollaya's Developer ID for notarization.
4. **Linux arm64 and Windows x64.** The CPU archives (`ubuntu-arm64`, `win-cpu-x64` with LLVM's
   `libomp.dll`). CUDA on native Windows (`ggml-cuda.dll`) is left for the Windows CUDA pack.
5. **In the runner.** `ollaya runner --gguf` loads the libraries with `libloading`, checks
   `llama_version()` and the default parameter structs against the layout `ffi.rs` was written for,
   and refuses any other build. It evaluates each question as `ids[..P]` (the state prefix, one cold
   pass, kept while the next question shares it) then `ids[P..]` as one batch, and reads the label
   logits with `llama_get_logits_ith`. Devices come from ggml's registry (`ollaya llama-devices`
   prints them); `auto` falls back to the CPU when the GPU cannot hold the model.
6. **Parity reference.** The goldens come from the family's Python reference prompt, evaluated on
   the **same pinned build's** `llama-server` with the same plan
   (`convert/ollaya_convert/families/llm_common/plan.py`), one goldens file per device. For winnow,
   the author's own server is a second, independent check of the prompt token ids and decisions.

## Consequences

- Ollaya compiles no C++ and no CUDA. The binaries are ggml-org's, byte for byte, with GitHub build
  attestations (`gh attestation verify <asset> --repo ggml-org/llama.cpp`).
- **Every bump** of llama.cpp: change the pins in `scripts/llama-cpp.sh`, re-check `ffi.rs` against
  the new `include/llama.h` (the load-time check refuses a mismatched build), regenerate
  `packaging/llama.cpp/THIRD_PARTY_NOTICES`, and regenerate every GGUF model's goldens and rerun
  parity on every device. Numbers move between builds.
- The Linux libraries need GCC's OpenMP runtime (`libgomp.so.1`) from the host; `install.sh`
  warns when it is missing and the Docker image installs it. The Windows libraries need the Visual
  C++ runtime (`MSVCP140.dll`, part of the Visual C++ Redistributable), which upstream does not bundle.
- One question at a time, sequentially. Forking the prefix into several sequences and decoding all
  suffixes in one batch (`llama_memory_seq_cp`) would be faster, but it is a different split, so it
  needs its own reference and goldens first.
- Nobody else ships ggml-org's prebuilt libllama this way today; Jan and Ollama build their own. If
  the prebuilt ABI ever becomes a problem, the fallback stays within B: build the same tag in CI
  with upstream's release flags.

## Evidence

- llama.cpp releases and versioning: [v0.5.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0)
  (points to [b11146](https://github.com/ggml-org/llama.cpp/releases/tag/b11146)),
  [docs/release.md](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/docs/release.md),
  [discussion #1579](https://github.com/ggml-org/ggml/discussions/1579),
  [release.yml](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/.github/workflows/release.yml)
  (build flags per platform), [PR #28186](https://github.com/ggml-org/llama.cpp/pull/28186) (Linux
  CUDA archives), [GitHub asset digests](https://github.blog/changelog/2025-06-03-releases-now-expose-digests-for-release-assets/).
- libllama API history: [issue #9289](https://github.com/ggml-org/llama.cpp/issues/9289);
  llama-server's: [issue #9291](https://github.com/ggml-org/llama.cpp/issues/9291).
- llama-server has no raw-logits output:
  [server-common.cpp](https://github.com/ggml-org/llama.cpp/blob/b11149/tools/server/server-common.cpp)
  (`get_token_probabilities`), [server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/b11149/tools/server/server-context.cpp)
  (`n_cmpl` children are discarded).
- Ollama: [PR #16031](https://github.com/ollama/ollama/pull/16031),
  [llama/server/CMakeLists.txt](https://github.com/ollama/ollama/blob/v0.34.2/llama/server/CMakeLists.txt),
  [llama/compat](https://github.com/ollama/ollama/blob/v0.34.2/llama/compat/README.md),
  [llm/llama_server.go](https://github.com/ollama/ollama/blob/v0.34.2/llm/llama_server.go).
- Jan: [PR #8762](https://github.com/janhq/jan/pull/8762). Lemonade:
  [llamacpp_server.cpp](https://github.com/lemonade-sdk/lemonade/blob/main/src/cpp/server/backends/llamacpp/llamacpp_server.cpp).
  Docker Model Runner: [native/README.md](https://github.com/docker/model-runner/blob/main/llamacpp/native/README.md).
  RamaLama: [build_llama.sh](https://github.com/containers/ramalama/blob/main/container-images/scripts/build_llama.sh).
  text-generation-webui: [llama_cpp_server.py](https://github.com/oobabooga/text-generation-webui/blob/main/modules/llama_cpp_server.py).
  LocalAI: [backend/cpp/llama-cpp](https://github.com/mudler/LocalAI/tree/master/backend/cpp/llama-cpp).
  llama-cpp-2: [build.rs](https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-sys-2/build.rs).
- Measured here (2026-09-25 and 26): archive contents, `NEEDED` libraries and glibc floors
  (`objdump -p`, `objdump -T`), the Metal build's `minos 13.3` (`otool -l`), every pinned sha256
  against GitHub's asset `digest`, and `gh attestation verify` of the pinned
  `llama-b11146-bin-macos-arm64.tar.gz` (signer `.github/workflows/release.yml@refs/heads/master`,
  source commit `7fe450e`).
