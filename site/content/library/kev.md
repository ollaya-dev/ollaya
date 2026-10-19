Kev is a family of decision models by [Jared Palmer](https://huggingface.co/jaredpalmer): a LoRA adapter on a Qwen3.5 base model plus a small pointer head, released under Apache-2.0. For each question it lays out the state, the question and one span per option, then scores every option at the end of its own span against the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Checkpoint | Params | Typed-decisions accuracy | Out-of-domain (upstream) |
|---|---|---|---|---|---|
| `kev:0.8b` | Qwen3.5-0.8B | round 15 | 0.76B | {{result kev:0.8b typed.accuracy 3}} | 0.697 |
| `kev:latest`, `kev:4b` | Qwen3.5-4B | round 10 | 4.2B | {{result kev:4b typed.accuracy 3}} | 0.838 |
| `kev:9b` | Qwen3.5-9B | 2026-09-21 | 7.9B | **{{result kev:9b typed.accuracy 3}}** | 0.852 |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `decider:2b` scores {{result decider:2b typed.accuracy 3}}, `nli` {{result nli:deberta-v3-large typed.accuracy 3}} and `laya:en` {{result laya:en typed.accuracy 3}}. The labels have low annotator agreement, so compare the numbers against each other rather than reading them as absolutes. The last column is upstream's locked out-of-domain test (transfer-v4); Jev scores 0.857 on its development items.

## Usage

```shell
ollaya run kev --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run kev:9b --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `kev` (the same as `kev:4b`), `kev:0.8b` or `kev:9b`. Before 2026-09-27, `kev` meant `kev:0.8b`.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** the triage preset (five questions) on a short message takes {{result kev:0.8b gpu.p50_ms 0}} ms on `0.8b`, {{result kev:4b gpu.p50_ms 0}} ms on `4b` and {{result kev:9b gpu.p50_ms 0}} ms on `9b` at the median.
- **State length:** every question reads the whole state again, so the cost grows with the number of questions times the state length.
- **CPU:** `0.8b` takes {{result kev:0.8b cpu.p50_ms 0}} ms for five questions with a short state on the Threadripper 3960X (24 cores). `4b` and `9b` belong on a GPU.

## How it works

- **One row per question.** The state comes first, then the question, then one span per option, then a decide token. The pointer head compares each option's closing token with the decide token.
- **Calibrated.** The scores are divided by the temperature each checkpoint ships, fitted upstream on development data: 2.35 (`0.8b`), 2.41 (`4b`), 2.30 (`9b`).
- **Weights.** From Hugging Face, pinned to a commit and verified by sha256: the Qwen3.5 base from Qwen, the adapter and `head.pt` from Jared Palmer. The adapter is applied at run time, not merged; Ollaya hosts only the ONNX graph (10 to 14 MB).
- **Precision.** The fp32 tags compute in fp32 on the BF16 base weights. `0.8b` widens them when it loads; `4b` and `9b` keep them BF16, half the memory, and widen each layer just before it runs.
- **Parity.** Ollaya's Rust runtime matches upstream Kev exactly: identical token rows and option positions, and the same decision on every test question, on CPU and CUDA.
- **fp16 tags.** `kev:4b-fp16` and `kev:9b-fp16` run the trunk in fp16 on NVIDIA GPUs ([how](https://github.com/ollaya-dev/ollaya/blob/main/docs/decisions/0008-fp16-decoder-tags.md)). On the RTX 5090: {{result kev:4b-fp16 gpu.p50_ms 0}} and {{result kev:9b-fp16 gpu.p50_ms 0}} ms for five questions (fp32: {{result kev:4b gpu.p50_ms 0}} and {{result kev:9b gpu.p50_ms 0}} ms), {{result kev:4b-fp16 parity.cuda.decisions_same 0}} and {{result kev:9b-fp16 parity.cuda.decisions_same 0}} of {{result kev:4b-fp16 parity.cuda.questions 0}} parity decisions the same as fp32, {{result kev:4b-fp16 typed.accuracy 3}} and {{result kev:9b-fp16 typed.accuracy 3}} on typed decisions. `kev:0.8b` has no fp16 tag (its fp32 graph is as fast). On a CPU use the fp32 tags.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of option to description (`{"refund": null, ...}`). A plain list is rejected, as upstream Kev does.
- **Context.** A question's row, state included, can be up to 8,192 tokens. Kev was trained on shorter states, so very long states are untested upstream.
- **Memory.** On the GPU, with requests of about 1,400 tokens, `4b` took about 11 GB and `9b` about 18 GB; the weights stay BF16. `9b` needs a 24 GB GPU and peaked at 22 GB on rows of 2,000 tokens, so much longer states do not fit. The downloads are 9.5 GB (`4b`) and 19.5 GB (`9b`).
