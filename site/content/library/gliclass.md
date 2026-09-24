GLiClass is an instruction-following zero-shot classifier by [Knowledgator](https://huggingface.co/knowledgator). Ollaya writes a question's options as labels and its instructions as the task prompt, and the model scores all the labels in a single pass. The cost therefore barely grows with the number of options.

## Models

| Tag | Backbone | Params | License | Typed-decisions accuracy |
|---|---|---|---|---|
| `gliclass:latest`, `gliclass:large` | DeBERTa-v3-large | 439M | Apache-2.0 | 0.477 |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `nli` scores 0.548 and `laya:en` 0.361.

## Usage

```shell
ollaya run gliclass --preset triage "I was charged twice for my subscription this month and want a refund."
```

## How it works

- **One sequence per question.** Label markers go first, then the task prompt, then the state, up to 1,024 tokens.
- **Scoring.** The graph pools every label's span and scores it against the text.
- **Mapping to question types.** The mapping of `choice`, `score` and `noul` onto labels is Ollaya's, chosen by testing on typed-decisions.
- **Weights.** They are Knowledgator's own `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256.
- **Parity.** Ollaya's Rust runtime matches the Python reference exactly on CPU and CUDA.

## Limits

- **Yes/no questions without criteria are its weakest point.** Such a question is scored as one label with a sigmoid, and it can be confidently wrong. Give `noul` questions `criteria` with both `true` and `false` descriptions, or use `nli` for them.
- **Option count.** A question with more options than fit in 1,024 tokens is rejected with `TOO_MANY_OPTIONS`.
- **Uncalibrated.** The model is not calibrated. Fit a `CALIBRATION` layer before you rely on thresholds.
