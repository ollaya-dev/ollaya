Kev is a family of decision models by [Jared Palmer](https://huggingface.co/jaredpalmer): a LoRA adapter on a Qwen3.5 base model plus a small pointer head, released under Apache-2.0. For each question it lays out the state, the question and one span per option, then scores every option at the end of its own span against the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Checkpoint | Params | Typed-decisions accuracy | Out-of-domain (upstream) |
|---|---|---|---|---|---|
| `kev:latest`, `kev:0.8b` | Qwen3.5-0.8B | round 15 | 0.76B | 0.460 | 0.697 |
| `kev:4b` | Qwen3.5-4B | round 10 | 4.2B | 0.669 | 0.838 |
| `kev:9b` | Qwen3.5-9B | 2026-09-21 | 7.9B | **0.722** | 0.852 |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `decider:2b` scores 0.591, `nli` 0.548 and `laya:en` 0.361. The labels have low annotator agreement, so compare the numbers against each other rather than reading them as absolutes. The last column is upstream's locked out-of-domain test (transfer-v4); Jev scores 0.857 on its development items.

## Usage

```shell
ollaya run kev --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
ollaya run kev:9b --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `kev`, `kev:4b` or `kev:9b`.

## Speed

- **State length:** every question reads the whole state again, so the cost grows with the number of questions times the state length.
- **CPU:** `0.8b` takes about 2 s for five questions with a short state on a 24-core x86 CPU. `4b` and `9b` belong on a GPU.

## How it works

- **One row per question.** The state comes first, then the question, then one span per option, then a decide token. The pointer head compares each option's closing token with the decide token.
- **Calibrated.** The scores are divided by the temperature each checkpoint ships, fitted upstream on development data: 2.35 (`0.8b`), 2.41 (`4b`), 2.30 (`9b`).
- **Weights.** The files download from Hugging Face, pinned to a commit and verified by sha256: the Qwen3.5 base weights from Qwen's repository, and the adapter and `head.pt` from Jared Palmer's. The adapter is applied at run time, not merged, so every file stays byte for byte the author's. Ollaya hosts only the ONNX graph (10 to 14 MB).
- **Precision.** Every model computes in fp32 on the BF16 base weights. `0.8b` widens them when it loads; `4b` and `9b` keep them BF16, half the memory, and widen each layer just before it runs.
- **Parity.** Ollaya's Rust runtime matches upstream Kev exactly: identical token rows and option positions, and the same decision on every test question, on CPU and CUDA.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of option to description (`{"refund": null, ...}`). A plain list is rejected, as upstream Kev does.
- **Context.** A question's row, state included, can be up to 8,192 tokens. Kev was trained on shorter states, so very long states are untested upstream.
- **Memory.** On the GPU, with requests of about 1,400 tokens, `4b` took about 11 GB and `9b` about 18 GB; the weights stay BF16. `9b` needs a 24 GB GPU and peaked at 22 GB on rows of 2,000 tokens, so much longer states do not fit. The downloads are 9.5 GB (`4b`) and 19.5 GB (`9b`).
