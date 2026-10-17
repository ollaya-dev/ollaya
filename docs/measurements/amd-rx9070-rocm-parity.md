# AMD Radeon RX 9070 ROCm parity measurements

Measured on 2026-09-28 for [PR #33](https://github.com/ollaya-dev/ollaya/pull/33).
Same-backend ROCm parity passes for the measured model (Winnow-E4B), which is Ollaya's gate for a device.

## Setup

- Linux x86-64; AMD Ryzen 9 9950X 16-Core Processor; 64 GiB system RAM.
- AMD Radeon RX 9070 discrete GPU (`gfx1201`), 16 GiB dedicated VRAM, ROCm 7.1 / amdgpu kernel driver.
- Ollaya revision `8ea8a82`, matching PR #33.
- Stock `llama-b11146-bin-ubuntu-rocm-10.0-x64.tar.gz`: version `0.5.0-dev`, build 11146,
  commit `7fe450e19305b828c199d602c23a8337aaa1f03b`.
- Python reference encoder: laya 0.3.7. The default 40 typed-decisions rows and the full edge-case set were used.

| Model | Pinned author GGUF | Context | Plan | Temperature |
| --- | --- | ---: | --- | ---: |
| Winnow-E4B | `EldanRing/Winnow-E4B`, revision `734302fe5fbfeb3f21a7ece62653c9539be4aaf3`, `gguf/Winnow-E4B-Q8_0.gguf` | 8192, full sliding-window cache | prefix | 1.2574172017327816 |

GGUF SHA-256:

- Winnow-E4B: `840e3f50e5a9c218727f44e121d1b37cc9e2c3b318c8eb422ba6ef2e27b618a2`.

## Same-backend native parity

The reference prompt encoder ran through stock llama-server on **ROCm0**, using the fixed evaluation plan.
`parity_llama` then compared Ollaya on **ROCm0** with those fixtures, including prompt IDs,
split points, label candidates, state token counts, truncation and rejected requests.
Logit differences below are measured after log-softmax over the options, as in the parity tool.

| Model | Cases | Rejected cases | Matching decisions | Maximum logit difference | Maximum probability difference | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Winnow-E4B | 123 | 15 | 505/505 | 1.246e-5 | 3.022e-6 | PASS |

Fixture SHA-256:

- Winnow ROCm: `6778a91abc508309a7bc704b26c98369dbedbeb7ff759c74188c27f35eb7e8b3` ([Gist 76f669b9b5121f1b26b13e550aeef378](https://gist.github.com/solarpush/76f669b9b5121f1b26b13e550aeef378)).

Winnow used 108 prefix steps, 397 barriers and 505 question steps.
The pass establishes fidelity to the same backend's reference. It does not establish matching outputs between ROCm and CUDA.

## Against CUDA (information, not a gate)

The maintainer supplied the [exact CUDA reference](https://gist.github.com/cobanov/c808f61976a8210235d8faf8ae91aa15)
in [PR comment 5864110141](https://github.com/ollaya-dev/ollaya/pull/27#issuecomment-5864110141).
Its SHA-256 was verified as `b0b15b5646e8cda17364b6097a9613eb231a88a4813b52c129423313447695fe`.
The reference comes from stock b11146 CUDA on an RTX 4090, Linux, 2026-09-26.

| Implementation compared with CUDA | Matching decisions | Maximum logit difference | Maximum probability difference |
| --- | ---: | ---: | ---: |
| Stock ROCm llama-server, full 123 cases | 500/505 | 0.3298 | 0.0318 |
| Ollaya ROCm runner, full 123 cases | 500/505 | 0.3298 | 0.0318 |

Both full comparisons cover the same 505 questions and 15 rejected cases. The runner reported
the same decision changes and numerical range as stock llama-server, so the difference is the
backend's rounding, not Ollaya's. Parity is gated per device (docs/decisions/0003-llama-cpp-runtime.md,
point 7): differences between backends are measured and documented, not gated. The x86-64 CPU
differs from CUDA by a similar amount (0.3535) on the same model.

## Reproduction

From `convert/`, use the pinned author GGUF and the stock server from build b11146:

```bash
uv run python -m ollaya_convert.families.llm_common.export_llama winnow \
  --server $SERVER --gguf $WINNOW --slug rx9070-rocm-native \
  --repo EldanRing/Winnow-E4B --revision 734302fe5fbfeb3f21a7ece62653c9539be4aaf3 \
  --file gguf/Winnow-E4B-Q8_0.gguf --temperature 1.2574172017327816 \
  --upstream-commit 77d14580c6732ca2f3745750c1dc1fd446d8bcee \
  --n-ctx 8192 --device ROCm0 --port 8098
```

Place the matching GGUF as `model.gguf` beside each generated `decision.json` and `calibration.json`.
From the repository root, set `OLLAYA_LIBRARY_PATH` to the stock install's `lib/ollaya` directory:

```bash
OLLAYA_LIBRARY_PATH=lib/ollaya cargo run --release -p ollaya-runner --example parity_llama -- \
  convert/out/winnow-rx9070-rocm-native convert/out/winnow-rx9070-rocm-native/goldens-rocm.jsonl rocm:0
```
