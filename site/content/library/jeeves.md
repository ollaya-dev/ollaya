Jeeves is a decision model by [PostHog](https://huggingface.co/PostHog): Qwen3.5-9B with a LoRA merged in and a pointer head that scores every option at its own marker, trained on public data only and released under Apache-2.0. Ollaya runs Jeeves v2 (October 2026). Upstream, Jeeves can write a reasoning chain before it answers; Ollaya runs it without the chain, in one forward pass per question. It never generates text.

> Needs Ollaya 0.8.0 or newer, the first release that runs the `jeeves-markers-v1` layout.

## Models

| Tag | Base | Params | Typed-decisions accuracy | Five questions, RTX 5090 |
|---|---|---|---|---|
| `jeeves:latest`, `jeeves:9b` | Qwen3.5-9B, LoRA merged | 9B | {{result jeeves:9b typed.accuracy 3}} | {{result jeeves:9b gpu.p50_ms 0}} ms |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states, measured by Ollaya, with a calibration error (ECE) of {{result jeeves:9b typed.ece 3}}. The authors report 0.812 on their own test split without thinking (0.851 when it reasons on the questions it is unsure of). Jeeves v2 trains on typed-decisions' train split, so this number is in-distribution and not comparable with the zero-shot models.

## Usage

```shell
ollaya run jeeves --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `jeeves`.

## How it works

- **Rows.** Ollaya builds each question exactly as the authors' encoder does: the state, the question and the options between marker tokens, an empty thought, then the options again and a decide marker. The pointer head scores each option at its marker against the decide marker.
- **No thinking.** Jeeves' reasoning mode generates a chain first, which Ollaya's one-pass runtime does not do. Without it the authors report 0.812 against 0.851 on their test split for v2, at a fraction of the latency.
- **Calibrated.** One fitted temperature, 1.522, from the model's `export.json` (the authors fit a second one for answers after a reasoning chain, which Ollaya does not write).
- **Weights.** PostHog's five BF16 shards and `head.pt` download from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (11 MB).
- **Parity.** Ollaya's Rust runtime matches the authors' own code in fp32 on CUDA: identical token rows and the same decision on all {{result jeeves:9b parity.cuda.questions 0}} test questions, probabilities within {{result jeeves:9b parity.cuda.max_prob_diff e1}}. On the CPU the {{result jeeves:9b parity.cpu.questions 0}} smoke questions agree too.

## Limits

- **Questions.** As upstream, a question takes only `type`, `instructions` and `criteria`, and choice criteria must be an object of options.
- **Rows.** Each question's row, the state included, can be up to 8,192 tokens.
- **Memory.** About 18 GB with the weights kept BF16: a 24 GB GPU.
