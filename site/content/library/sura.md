Sura is Ollaya's own family of **decoder decision models**: Qwen3.5 distilled from `pplx-decider:flash` in its joint format, on the same 340,000 permissively licensed questions as coba's zero-shot tags. It reads a *state* (a message, a ticket, a JSON object, an agent's next tool call) and typed questions, and returns calibrated probabilities for every question in one forward pass. It never generates text.

## Models

| Tag | Base | Params | Trained |
|---|---|---|---|
| `sura:latest`, `sura:2b` | Qwen3.5-2B-Base | 2B | zero-shot |
| `sura:0.8b` | Qwen3.5-0.8B-Base | 0.8B | zero-shot |

**Zero-shot**: sura trained without any of the datasets of the benchmarks Ollaya reports, so its scores below are held-out measurements.

## Usage

```shell
ollaya run sura --ask "Is the customer asking for a refund?" "I was charged twice for order A-104. Refund the duplicate today."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `sura`.

## Speed

- **RTX 5090, over HTTP (Ollaya Bench):** at the median for five triage questions (one forward pass), {{result sura:2b gpu.p50_ms 0}} ms for `sura:2b` and {{result sura:0.8b gpu.p50_ms 0}} ms for `sura:0.8b`; {{result sura:2b public.median_ms 0}} ms and {{result sura:0.8b public.median_ms 0}} ms per question on Bespoke Labs' public benchmark.
- **CPU (Threadripper 3960X):** {{result sura:2b cpu.p50_ms 0}} ms (`2b`) and {{result sura:0.8b cpu.p50_ms 0}} ms (`0.8b`) for the same five questions. For a CPU, coba is the faster family.

## Accuracy

Measured by Ollaya on an RTX 5090, on the 400 typed-decisions states and Bespoke Labs' public benchmark (3,880 human-labeled questions):

| Tag | Typed decisions | Public, macro accuracy | Public, pooled ECE |
|---|---|---|---|
| `sura:2b` | {{result sura:2b typed.accuracy 3}} | {{result sura:2b public.macro_accuracy 3}} | {{result sura:2b public.pooled_ece 3}} |
| `sura:0.8b` | {{result sura:0.8b typed.accuracy 3}} | {{result sura:0.8b public.macro_accuracy 3}} | {{result sura:0.8b public.pooled_ece 3}} |

On Ollaya Bench, `sura:2b` is level with `kev:0.8b` and `sura:0.8b` with `coba:1b`. Both are below `decider:0.8b` and `decider:2b`, decoders on the same bases trained on more data, and far below their teacher, `pplx-decider:flash` ({{result pplx-decider:flash typed.accuracy 3}} typed, {{result pplx-decider:flash public.macro_accuracy 3}} public), twice the larger sura's size.

## How it works

- **The teacher's format.** A request is one sequence: the state, then every question with its options, each followed by a marker where a readout over 255 answer codes picks the answer. The full-attention layers see the whole sequence; the Gated DeltaNet layers stay causal. Sura shares pplx-decider's tokenizer and prompt, so it learned the teacher's whole distribution over the options, not only its answers.
- **Calibrated.** The readout logits are divided by a temperature fitted on held-out states (0.92 for `2b`, 0.93 for `0.8b`).
- **Weights on Hugging Face.** [ollaya-dev/sura-2b](https://huggingface.co/ollaya-dev/sura-2b) and [ollaya-dev/sura-0.8b](https://huggingface.co/ollaya-dev/sura-0.8b) hold the BF16 weights, the readout and upstream's `pplx_decider` code, so the authors' loader reads the same checkpoint Ollaya runs.
