# ollaya

Run open **decision models** locally, the way Ollama runs LLMs.

A decision model reads a *state* (text, an email, a ticket, JSON) plus typed questions
(`choice`, `score`, `noul`) and returns calibrated probabilities in one forward pass. It
never generates text. Ollaya pulls these models by name, serves them from a local daemon, and
speaks the TypeSafe `/v1/systemone` wire format, so existing clients work by changing their
base URL.

> **Status: pre-release (Phase 0).** The inference core is verified against the reference
> implementation. The CLI, daemon and registry are not built yet.

## Layout

| Path | What |
|---|---|
| `crates/ollaya-decision` | Engine-agnostic decision logic: question schema, sequence layouts, calibration, answers |
| `crates/ollaya-runner` | Inference engines. Today: ONNX Runtime (CPU, CUDA) |
| `convert/` | Build-time only (Python): export checkpoints to ONNX, parity checks, golden fixtures |
| `site/` | The website and static registry host (see `site/README.md`) |

## Phase 0 results (Laya, RTX 4090)

- **Parity with the reference.** Across 2,383 questions per checkpoint, the ONNX export gives
  the same decision as Laya's PyTorch fp32 reference 100% of the time, with a maximum
  probability difference of 1.1e-4.
- **Rust runtime.** It reproduces Laya's tokenization byte for byte and returns the same
  answers.
- **Latency.** Median for a 5-question request:

  | Model | PyTorch bf16 | Ollaya fp16 |
  |---|---|---|
  | `laya:en` | 22.4 ms | 16.3 ms |
  | `laya:multilingual` | 17.7 ms | 9.1 ms |

## Development

```sh
# export + verify a checkpoint (needs the Laya weights, see convert/ollaya_convert/laya_ref.py)
cd convert
uv run python -m ollaya_convert.export en --out out/laya-en
uv run python -m ollaya_convert.parity en out/laya-en/model.onnx
uv run python -m ollaya_convert.goldens en --out out/goldens

# check the Rust runtime against the goldens
cargo run --release -p ollaya-runner --example parity -- convert/out/laya-en convert/out/goldens/laya-en.jsonl
```

CUDA builds use `--features ollaya-runner/cuda` and need CUDA 13 and cuDNN 9 libraries on
the library path.

## License

Apache-2.0. Laya is by Convai Innovations, also under Apache-2.0.
