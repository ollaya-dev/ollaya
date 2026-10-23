GLiClass is an instruction-following zero-shot classifier by [Knowledgator](https://huggingface.co/knowledgator). Ollaya writes a question's options as labels and its instructions as the task prompt, and the model scores all the labels in a single pass, so a question takes one pass however many options it has. Each label adds tokens to that pass, though, so a question with dozens of options takes many times as long as a yes/no one.

## Models

| Tag | Backbone | Params | License | Typed-decisions accuracy |
|---|---|---|---|---|
| `gliclass:latest`, `gliclass:large` | DeBERTa-v3-large | 439M | Apache-2.0 | {{result gliclass:large typed.accuracy 3}} |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `nli` scores {{result nli:deberta-v3-large typed.accuracy 3}} and `laya:en` {{result laya:en typed.accuracy 3}}.

## Usage

```shell
ollaya run gliclass --preset triage "Your app crashed during checkout and I was charged twice. Please refund one of the charges."
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
