# Intel Arc 140T parity measurements

Measured on 2026-09-27 and 2026-09-28 for [PR #27](https://github.com/ollaya-dev/ollaya/pull/27).
The full Winnow-E4B comparison against the reviewer's CUDA reference **fails** the gate.
Same-backend Vulkan parity passes for both measured models.

## Setup

- Windows x64; Intel Core Ultra 9 285H; 64 GiB system RAM.
- Intel Arc 140T integrated GPU, `Vulkan0`, driver `32.0.101.8860`.
  Vulkan reports about 37 GiB accessible memory. This is shared system memory, not a separate VRAM pool.
- Ollaya revision `5ad1f9f61d4d46b15d9f937be536aab597575fa7`, after merging main `91f873c02fd4b8f95649c49e2aeb43f146a7fe7a`.
  CUDA comparison runs used the same runner executable at checkout revision
  `80ffebb898a1c327d011d2750b8f1f03c59319ea`; the intervening commit changed documentation
  and reference-fixture filenames, not Rust inference code.
- Stock `llama-b11146-bin-win-vulkan-x64.zip`: version `0.5.0-dev`, build 11146,
  commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, Clang 20.1.8.
  The native parity runs used the stock libraries, not the local diagnostic build described below.
- Python reference encoder: laya 0.3.7. The default 40 typed-decisions rows and the full edge-case set were used.

After these measurements, current main v0.7.5 (`32acb6d2ea13f616b1e11d9dfb8260b305106847`)
was merged without conflicts in `067bfbd5949ec830b3dd9ed575894c9dbd75fe3b`.
It changes neither the llama runner source nor `parity_llama`, the Winnow registry pin,
or the pinned llama.cpp distribution script. The five llama tests, four server-configuration
tests and formatting check pass, and the release parity example rebuilds. A stock-Vulkan
recheck on the four selected CUDA cases reproduces 12/15 decisions and max option-logit
error 0.3261. This recheck is not a new full 505-question result.
After the merge, `cargo test --workspace --locked`,
`cargo clippy --workspace --all-targets --locked -- -D warnings`,
`cargo build --release --locked -p ollaya`, and the site's `npm run typecheck` also pass.

| Model | Pinned author GGUF | Context | Plan | Temperature |
| --- | --- | ---: | --- | ---: |
| Winnow-E4B | `EldanRing/Winnow-E4B`, revision `734302fe5fbfeb3f21a7ece62653c9539be4aaf3`, `gguf/Winnow-E4B-Q8_0.gguf` | 8192, full sliding-window cache | prefix | 1.2574172017327816 |
| JevK5 | `alibiserikbay/JevK5-GGUF`, revision `ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a`, `jevk5-4b-v0.3-Q8_0.gguf` | 16384 | cold | 1.22 |

GGUF SHA-256:

- Winnow-E4B: `840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`.
- JevK5: `aea433883bc7ed399f2fbd539e53d2eac7caf71a946fe6650995a413979d4a30`.

## Same-backend native parity

The reference prompt encoder ran through stock llama-server on **Vulkan0**, using the fixed evaluation plan.
`parity_llama` then compared Ollaya on **Vulkan0** with those fixtures, including prompt IDs,
split points, label candidates, state token counts, truncation and rejected requests.
Logit differences below are measured after log-softmax over the options, as in the parity tool.

| Model | Cases | Rejected cases | Matching decisions | Maximum logit difference | Maximum probability difference | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Winnow-E4B | 123 | 15 | 505/505 | 1.143e-5 | 2.299e-6 | PASS |
| JevK5 | 123 | 4 | 593/593 | 9.521e-6 | 1.563e-6 | PASS |

Fixture SHA-256:

- Winnow Vulkan: `de45022c172f5f590e3ca18e461102babeae48e73868d9eb294ec42aa4a57011`.
- JevK5 Vulkan: `2db2417a9c00ef5da3e7cd92838df6924dc285dbee6a86db1576ee64ff498aa0`.

Winnow used 108 prefix steps, 397 barriers and 505 question steps. JevK5 used 579 cold steps;
single-option questions require no inference. These passes establish fidelity to the same backend's
reference. They do not establish matching outputs between Vulkan and CUDA.

## CUDA-reference gate

The maintainer supplied the [exact CUDA reference](https://gist.github.com/cobanov/c808f61976a8210235d8faf8ae91aa15)
in [PR comment 5864110141](https://github.com/ollaya-dev/ollaya/pull/27#issuecomment-5864110141).
Its SHA-256 was verified as `b0b15b5646e8cda17364b6097a9613eb231a88a4813b52c129423313447695fe`.
The supplied decision and calibration JSON files exactly match the registry files used above.
The reference comes from stock b11146 CUDA on an RTX 4090, Linux, 2026-09-26.

| Arc implementation compared with CUDA | Matching decisions | Maximum logit difference | Maximum probability difference | Gate |
| --- | ---: | ---: | ---: | --- |
| Stock Vulkan llama-server, full 123 cases | 502/505 | 0.32614444 | 0.05600171 | FAIL |
| Ollaya Vulkan runner, full 123 cases | 502/505 | 0.3261 | 0.05600 | FAIL |

Both full comparisons cover the same 505 questions and 15 rejected cases. The stock comparison
found no differences in prompt IDs, split points, candidates, wire order, state token counts or
truncation. 502 questions exceed the 1e-3 logit tolerance. The runner reported the same decision
changes and numerical range, separating the backend difference from an Ollaya wrapper error.

The changed decisions were `preset/router/mask_text` / `domain`, `edge/many_options_40` / `intent`,
and `td/customer_service_000080` / `churn_risk`.

A targeted regression set contains the complete requests for those three cases, plus
`preset/email/mask_text`, which has the largest logit difference: four cases, 15 questions.
Its SHA-256 is `1253f4099d15b9e76ce6c48df85af8abb5d0901e6359e71b047c754f2a58f711`.

| Targeted setting | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Disable F16 | 12/15 | 0.3321 | 0.01769 |
| Force MMVQ | 12/15 | 0.3279 | 0.01805 |
| Conservative settings | 12/15 | 0.3299 | 0.01782 |
| Disable cooperative matrices and cooperative matrices 2 | 13/15 | 0.1517 | 0.02258 |
| Disable both cooperative matrix modes and force MMVQ | 12/15 | 0.1414 | 0.02262 |
| Stock Vulkan server, flash attention disabled | 12/15 | 0.32484553 | 0.01751260 |
| Ollaya CPU | 15/15 | 0.1915 | 0.03166 |

Every targeted setting fails the unchanged logit tolerance. They were not expanded into further
full runs, since the selected counterexamples already disprove a complete pass.

## CPU-reference diagnostic screen

The completed stock CPU versus stock Vulkan comparison for JevK5 covers all 123 cases,
593 questions and four rejected requests: 586/593 matching decisions, maximum normalized
logit difference 0.33286801 and maximum probability difference 0.08266701.
574 questions exceed the 1e-3 logit tolerance; prompt and state metadata match in all cases.
The full CPU fixture SHA-256 is `d82aca8fb3d23cafcf554eb8aa55b49a843401e0d4b123afc5040ab06b6d5216`.

A stock CPU reference was generated independently for JevK5. A fixed screen selected ten spread
cases, 50 questions, from an initial 18-case snapshot of that reference. The identical screen was
used for all five variants below. This is a diagnostic sample, not the complete CUDA gate.

| Vulkan setting | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Default | 48/50 | 0.1899 | 0.05146 |
| `GGML_VK_DISABLE_F16=1` | 48/50 | 0.1895 | 0.05059 |
| Conservative settings below | 48/50 | 0.1892 | 0.05132 |
| `GGML_VK_FORCE_MMVQ=1` | 48/50 | 0.1807 | 0.05070 |
| `GGML_VK_DISABLE_MMVQ=1` | 48/50 | 0.1899 | 0.05146 |

The conservative run disabled F16, fusion, cooperative matrices, cooperative matrices 2,
DOT2 and integer dot products, using the corresponding `GGML_VK_DISABLE_*` variables.
Every variant failed the unchanged 1e-3 logit tolerance and exact-decision requirement.
The 18-case CPU snapshot SHA-256 was
`10552c9365c0cbd06ff58687a913ecbc3555b472af6103693914ecac6e0f8a6b`;
the ten-case input SHA-256 was `035b4c149439cf9eb477913b8fdf7cc3d2e04b1ca215fd859117241a9d2a9410`
in both screen batches. CPU is not a substitute for the requested CUDA fixture.
Runs overlapped other work, so elapsed times are not presented
as performance benchmarks.

## Operation-level investigation

A local MSVC build of the pinned source enabled `GGML_VULKAN_CHECK_RESULTS` and disabled
`GGML_BACKEND_DL`. The check mode did not compile unmodified: its globals and two check
functions were private to `ggml-vulkan-debug.cpp`, while `ggml-vulkan.cpp` referenced them.
A local declaration/linkage patch enabled the diagnostic build. It changes no inference arithmetic
and is not included in the shipped libraries or this Ollaya PR.

The existing `test-backend-ops` Q8_0/F32 MUL_MAT cases with `m=16`, `n=1|8` and `k=256|4096`
passed 12 supported tests with default Vulkan settings and with forced MMVQ. Unsupported
permuted/broadcast cases were skipped. The operation test has its own NMSE tolerance of 5e-4;
it does not test the final model's option-logit gate.

The internal CPU check reported `avg_err` around 0.0035-0.0078 for default matrix-vector cases,
and around 1e-7 for their forced-MMVQ counterparts. Its denominator is `max(abs(reference), 1)`.
Each run initializes its own random tensors, so these are operation diagnostics rather than
paired model-input measurements. A large-batch case retained about 0.00655 error with forced MMVQ.
The pinned source selects a dequantized route for Intel Windows matrix-vector operations by default;
forcing MMVQ changes that route. The full-model sample above still fails after this change.

Attempts to run the check-enabled server on a Winnow input did not complete: the HTTP connection
was reset, including a retry with fusion, async execution and graph optimization disabled and
serialized submissions enabled. Those attempts provide no complete model parity result.
The exact cause of the remaining model drift has not been established.

There are concrete precision differences in the pinned source:

- [Vulkan matrix selection](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L6097)
  forces F32 activations into an F16 input path for cooperative matrices with quantized weights.
  Disabling those matrices changes the path and reduces, but does not eliminate, the measured drift.
- [Vulkan activation quantization](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp#L114)
  stores its scale and sum as `f16vec2`; disabling F16 computation does not change this storage format.
- [CUDA's MMQ layout](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cuda/mmq.cuh#L59)
  selects four FP32 activation scales for Q8_0, and
  [the quantizer](https://github.com/ggml-org/llama.cpp/blob/7fe450e19305b828c199d602c23a8337aaa1f03b/ggml/src/ggml-cuda/quantize.cu#L550)
  writes those scales without the F16 conversion.

These differences identify possible sources of drift. They do not prove that changing scale
storage alone will make the entire model pass. No numerical backend patch has been validated.

## Reproduction

From `convert/`, use the pinned author GGUF and the stock server from build b11146:

```powershell
uv run python -m ollaya_convert.families.llm_common.export_llama winnow `
  --server $SERVER --gguf $WINNOW --slug arc-vulkan-native `
  --repo EldanRing/Winnow-E4B --revision 734302fe5fbfeb3f21a7ece62653c9539be4aaf3 `
  --file gguf/Winnow-E4B-Q8_0.gguf --temperature 1.2574172017327816 `
  --upstream-commit 77d14580c6732ca2f3745750c1dc1fd446d8bcee `
  --n-ctx 8192 --device Vulkan0 --port 8098

uv run python -m ollaya_convert.families.llm_common.export_llama jevk5 `
  --server $SERVER --gguf $JEVK5 --slug arc-vulkan-native `
  --repo alibiserikbay/JevK5-GGUF --revision ec67b0bfce5119a8b11a2cdb430bb43e3fa3e82a `
  --file jevk5-4b-v0.3-Q8_0.gguf --temperature 1.22 `
  --n-ctx 16384 --device Vulkan0 --port 8097
```

Place the matching GGUF as `model.gguf` beside each generated `decision.json` and `calibration.json`.
From the repository root, set `OLLAYA_LIBRARY_PATH` to the stock install's `lib/ollaya` directory:

```powershell
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/winnow-arc-vulkan-native convert/out/winnow-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
cargo run --release -p ollaya-runner --example parity_llama -- `
  convert/out/jevk5-arc-vulkan-native convert/out/jevk5-arc-vulkan-native/goldens-vulkan.jsonl Vulkan0
```

To reproduce the CPU screen, generate JevK5 goldens with the same arguments and
`--device cpu --slug arc-cpu`. Select these records from the resulting `goldens-cpu.jsonl`
and run `parity_llama` with that ten-case fixture on `Vulkan0`, changing only the settings above:

```text
preset/triage/tr_billing
preset/triage/tr_outage
preset/triage/en_injection
preset/triage/ar_complaint
preset/triage/de_cancel
preset/triage/conversation
preset/triage/mask_text
preset/email/tr_billing
preset/email/en_pricing
preset/email/hi_refund
```

The operation diagnostic command was:

```powershell
test-backend-ops.exe test -b Vulkan0 -o MUL_MAT `
  -p 'type_a=q8_0,type_b=f32,m=16,n=(1|8),k=(256|4096),'
```

The export/replay tools now name Vulkan fixtures `goldens-vulkan.jsonl` and record `Vulkan0`
in their metadata. Before this correction they used the filename `goldens-cpu.jsonl` even though
the server and metadata selected Vulkan. The measured Vulkan fixtures above were renamed after
export without modifying their contents.

## Private FP32 activation-scale experiment (2026-09-28)

A local experiment used the same b11146 source commit, MSVC 19.51 and shaderc 2026.3.
A control build without arithmetic changes reproduced the stock results on the four
selected CUDA regression requests (15 questions). Both builds had
`GGML_VULKAN_CHECK_RESULTS=OFF`, `GGML_BACKEND_DL=ON` and `GGML_NATIVE=OFF`.

The experiment changed only Q8_0 MMQ activation scales: four FP32 scales replace
four FP16 scale/sum pairs in the same 144-byte block. The producer, shader storage,
shared-memory cache and register cache preserve FP32. A separate quantization pipeline
identifies the new representation for buffer reuse. MMVQ and formats requiring the
activation sum retain their original representation. Cooperative matrices were disabled
to exercise MMQ instead of the F16 activation path.

| Build and settings | Decisions vs CUDA, selected 15 | Max option-logit difference | Max probability difference |
| --- | ---: | ---: | ---: |
| Local control, no cooperative matrices | 13/15 | 0.1517 | 0.02258 |
| FP32 scales, no cooperative matrices | 14/15 | 0.1337 | 0.01088 |
| Local control, no cooperative matrices, forced MMVQ | 12/15 | 0.1414 | 0.02262 |
| FP32 scales, no cooperative matrices, forced MMVQ | 14/15 | 0.1428 | 0.01103 |

Both experimental variants still change `td/customer_service_000080/churn_risk`.
These are selected counterexamples, not a new full 505-question result. They already
disprove acceptance of the experimental build under the unchanged CUDA gate, so the
experiment was not expanded to all 505 questions.

The existing operation tests passed 24/24 supported Q8_0 and Q4_1 cases in each build:

```powershell
$env:GGML_VK_DISABLE_COOPMAT = '1'
$env:GGML_VK_DISABLE_COOPMAT2 = '1'
test-backend-ops.exe test -b Vulkan0 -o MUL_MAT `
  -p 'type_a=(q8_0|q4_1),type_b=f32,m=16,n=(1|8),k=(256|4096),'
```

The same ten-request, 50-question JevK5 CPU diagnostic fixture was also replayed on
the local control and experimental Vulkan builds, with cooperative matrices disabled:

| JevK5 build | Decisions vs CPU, selected 50 | Max option-logit difference | Max probability difference |
| --- | ---: | ---: | ---: |
| Local control | 47/50 | 0.2072 | 0.02863 |
| FP32 scales | 48/50 | 0.1716 | 0.03094 |

This second-model comparison remains a CPU diagnostic; it cannot establish CUDA parity.
The [measurement data](intel-arc-140t-fp32-results.json) records the settings,
fixture hashes, runner hash, Vulkan library hashes and result summaries for both models.

The operation check uses the upstream test tolerance, not Ollaya's final-logit gate.
The FP32 experiment improves selected decision agreement, but does not uniformly reduce
maximum logit error and does not isolate the remaining error to a specific operator.
It establishes that changing these scales alone is insufficient. It does not prove
that cross-backend parity is impossible. The private patch and binaries are not included
in this PR or the distribution; these results do not validate a numerical correction.

## Further quantization and isolated-operation diagnostics

Two additional private variants used CUDA's scalar calculation order
`d_inv = 127 / amax`, `d = 1 / d_inv`, with FP32 scales. With cooperative matrices
disabled, the initial rounding implementation gives 13/15 selected CUDA decisions
and max logit error 0.1488; forcing MMVQ gives 13/15 and 0.1505. Explicitly rounding
halfway values away from zero, without a potentially rounded `abs(x) + 0.5` intermediate,
gives 14/15 and max error 0.2428. The latter passes the same 24 supported operation
tests. None passes the selected counterexamples, so no new full-gate run was claimed.

A separate program using llama.cpp's public evaluation callback captured intermediate
tensors for the 433-token `customer_service_000080/churn_risk` question, with its
original 345-token prefix. This instrumentation changes graph scheduling/fusion;
the captured runs are diagnostics, not substitutes for an uninstrumented CUDA gate.

For the first attention multiplication `Qcur-0`, the local CPU and Vulkan weights are
byte-identical. Prefix inputs differ by at most 6.103515625e-5, while outputs differ
by at most 0.1219133. A standalone replay then used **identical captured CPU inputs
and weights** on both backends. CPU replay reproduces its captured result exactly.
The maximum raw-tensor error against that CPU replay is 0.234026 for control Vulkan,
0.232921 for the original FP32-scale experiment, and 0.246365 for the reciprocal
and explicit-rounding variant.

These raw intermediate-tensor errors are different metrics from normalized option
logits. They isolate a CPU/Vulkan multiplication difference even with identical
inputs; they do not establish CUDA's output for that input, the sole cause of the
final gate failure, or a validated fix. The remaining investigation needs to separate
activation quantization from accumulation and compare this isolated operation with CUDA.
The [diagnostic data](intel-arc-140t-localization-results.json) records the hashes and results.

## Activation quantization versus accumulation

The local CPU's actual activation quantizer matches the host ties-to-even calculation
for all 883,200 captured values. Ties-to-even and ties-away quantization differ for
524 values in that input. Dequantizing the Q8_0 weights to F32 and disabling F16 and
cooperative matrices reduces CPU/Vulkan raw-output difference to 0.0005493164,
compared with 0.234026 for the quantized operation.

A read-only private diagnostic captured the GPU's quantized activation buffer after
the isolated operation. It preserves the result bytes exactly. The FP32 reciprocal
and explicit-rounding variant has 192 quantized values differing by one from the
host source-formula calculation. All mismatches lie within 7.62939453125e-6 of a
halfway value; scale differences reach 2.384185791015625e-7.
Using the GPU's actual quantized bytes in a Float64 mathematical dot reference
leaves only 0.0001070001 maximum accumulation error. This separates a quantization
contribution from the much smaller accumulation error for this measured operation.

A private refinement uses an FMA residual to correct each division. On this one
captured input, it eliminates all 192 quantized-value differences and makes the
FP32 scales identical to the host calculation. Its accumulation error remains
0.0001072667 and the operation tests pass 24/24 supported cases. This is a measured
local improvement, not a claim of correctly rounded division for every possible input.

The refined build still fails the selected CUDA questions:

| Settings on the private refined build | Decisions, selected 15 | Max option-logit difference |
| --- | ---: | ---: |
| No cooperative matrices | 13/15 | 0.1731 |
| No cooperative matrices or fusion | 14/15 | 0.1202 |
| No cooperative matrices or fusion, forced MMVQ | 14/15 | 0.1269 |

Matching the host calculation does not establish matching the stock CUDA implementation:
the pinned CUDA build configuration uses `-use_fast_math`, and an actual same-input
CUDA measurement is still needed. The [measurement data](intel-arc-140t-quantization-results.json)
records this distinction and the private-build hashes.

A [standalone same-input reproduction](https://gist.github.com/MauricioPerera/722b4878ada7d8836e52d64760ce601a)
contains 64 captured input rows, hashes, measured CPU/Vulkan outputs, and a small
ggml program that can load the stock CUDA backend without source changes. It extracts
the public pinned model's Q8_0 weight tensor and verifies its hash, rather than embedding
model weights. The Windows CPU replay exactly reproduces the corresponding full-capture
rows; the published data bytes were downloaded and verified against their hashes.
This packet requests an isolated CUDA diagnostic, not a replacement acceptance fixture.
The [maintainer request](https://github.com/ollaya-dev/ollaya/pull/27#issuecomment-5867333315)
asks for the actual CUDA output and build identity. No CUDA execution of this isolated
packet has been obtained yet; host-formula agreement cannot resolve that missing evidence.

## Gate status

The requested CUDA-to-Arc comparison is complete for the stock Vulkan build and **fails**.
No tested setting meets the gate and no numerical backend correction has been validated.
The private FP32 activation-scale experiment also fails on known CUDA counterexamples.
Further kernel work would need to isolate the remaining differences and complete a new
validation cycle; Ollaya currently distributes the pinned upstream binaries.
Same-backend parity passes and latency measurements must not be used to claim that Vulkan
is ready for the repository's cross-backend requirements.
