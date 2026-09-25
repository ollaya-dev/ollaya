Kev is a family of decision models by [Jared Palmer](https://huggingface.co/jaredpalmer): a LoRA adapter on a Qwen3.5 base model plus a small pointer head, released under Apache-2.0. For each question it lays out the state, the question and one span per option, then scores every option at the end of its own span against the question, in one forward pass. It never generates text.

## Models

| Tag | Base | Params | Typed-decisions accuracy |
|---|---|---|---|
| `kev:latest`, `kev:0.8b` | Qwen3.5-0.8B | 0.76B | 0.447 |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `decider:0.8b` scores 0.506, `nli` 0.548, `gliclass` 0.477 and `laya:en` 0.361. The labels have low annotator agreement, so compare the numbers against each other rather than reading them as absolutes. Upstream reports 0.825 accuracy on its own in-distribution development set and 0.652 out of domain.

## Usage

```shell
ollaya run kev --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `kev`.

## Speed

- **RTX 4090, end to end:** a five-question request with a short state takes about 185 ms at the median, close to `decider:0.8b`.
- **State length:** every question reads the whole state again, so the cost grows with the number of questions times the state length. Five questions about a 500-token state take about 0.65 s, and about a 2,000-token state 2.5 s.
- **CPU:** slow, about 2 s for five questions with a short state, on a 24-core x86 CPU.

## How it works

- **One row per question.** The state comes first, then the question, then one span per option, then a decide token. The pointer head compares each option's closing token with the decide token.
- **Calibrated.** The scores are divided by the temperature Kev ships (2.41), fitted upstream on development data.
- **Weights.** Three files download from Hugging Face, pinned to a commit and verified by sha256: the Qwen3.5-0.8B base weights from Qwen's repository, and the adapter and `head.pt` from Jared Palmer's. The adapter is applied at run time, not merged, so every file stays byte for byte the author's. Ollaya hosts only the ONNX graph (10 MB).
- **Parity.** Ollaya's Rust runtime matches upstream Kev exactly: identical token rows and option positions, and the same decision on every test question, on CPU and CUDA.

## Limits

- **Choice criteria.** Give `choice` criteria as an object of option to description (`{"refund": null, ...}`). A plain list is rejected, as upstream Kev does.
- **Context.** A question's row, state included, can be up to 8,192 tokens. Kev was trained on states of about 384 tokens, so very long states are untested upstream.
- **Accuracy.** At 0.8B, out-of-domain accuracy is modest. Upstream's larger Kev models (4B, 9B) are not converted yet.
