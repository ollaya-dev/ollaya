Cygnet by [blockbrain-ai](https://github.com/blockbrain-ai/cygnet-recipe) is not a fine-tune: it is Google's Gemma 4 12B IT, unchanged, plus a system message, the options as letters, one answer slot and one calibration temperature. Ollaya reads the probability of each option letter as the next token. It never generates text.

> Needs Ollaya 0.8.0 or newer, the first release that runs the `cygnet-v1` prompt.

## Models

| Tag | Weights | Typed-decisions accuracy | Five questions, RTX 5090 |
|---|---|---|---|
| `cygnet:latest`, `cygnet:12b` | Gemma 4 12B IT, Q8_0 GGUF, 12.7 GB | {{result cygnet:12b typed.accuracy 3}} | {{result cygnet:12b gpu.p50_ms 0}} ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya, with a calibration error (ECE) of {{result cygnet:12b typed.ece 3}} at Cygnet's temperature (0.289 without it). The authors report 203 of 231 on JevBench's public items with BF16 weights on vLLM; Ollaya runs ggml-org's Q8_0 conversion of the same Gemma revision on llama.cpp, which is close but not bit for bit the same model.

## Usage

```shell
ollaya run cygnet --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `cygnet`.

## How it works

- **Prompt.** Ollaya builds Cygnet's prompt exactly as its decision server does (`shim/decision_server.py` and `shim/cygnet_shim.py`): identical on all 1,364 test prompts. A structured state is shown as indented JSON, the rendering Cygnet was measured with.
- **Options.** A yes/no question reads `false` as `A` and `true` as `B`, the order Cygnet was measured in. Up to 20 options per question, read in one pass.
- **Calibration.** Temperature 3.4 over the letters, Cygnet's own.
- **Engine.** llama.cpp v0.5.0, ggml-org's own build, on an NVIDIA GPU (CUDA), an Apple silicon GPU (Metal) or the CPU. A 16 GB GPU holds it.
- **Parity.** Ollaya's runner matches stock llama.cpp (`llama-server` of the same build, on the same file) on CUDA: the same decision on all 502 test questions, probabilities within 4.1e-7.

## Limits

- **Options.** Up to 20 per question. Cygnet's server reads wider questions in groups; Ollaya answers them with `TOO_MANY_OPTIONS`.
- **Context.** 16,384 tokens per question; a longer prompt is rejected, not cut.
