GLiClass is a family of instruction-following zero-shot classifiers by [Knowledgator](https://huggingface.co/knowledgator). Ollaya writes a question's options as labels and its instructions as the task prompt, and the model scores all the labels in a single pass, so a question takes one pass however many options it has. Each label adds tokens to that pass, though, so a question with dozens of options takes many times as long as a yes/no one.

## Models

| Tag | Backbone | Params | License | Typed-decisions accuracy |
|---|---|---|---|---|
| `gliclass:latest`, `gliclass:large` | DeBERTa-v3-large | 439M | Apache-2.0 | {{result gliclass:large typed.accuracy 3}} |
| `gliclass:v4-edge` | DeBERTa-v3-large, with a calibrator | 440M | Apache-2.0 | {{result gliclass:v4-edge typed.accuracy 3}} |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. `gliclass:v4-edge` is Knowledgator's GLiClass v4 Edge, released in October 2026: it reads each label at its own marker and rescales every label's score with a small calibrator network. Its download is half the size of `large`'s (0.88 GB, bf16). For comparison, `nli` scores {{result nli:deberta-v3-large typed.accuracy 3}} and `laya:en` {{result laya:en typed.accuracy 3}}.

## Usage

```shell
ollaya run gliclass --preset triage "Your app crashed during checkout and I was charged twice. Please refund one of the charges."
```

## How it works

- **One sequence per question.** Label markers go first, then the task prompt, then the state, up to 1,024 tokens.
- **Scoring.** The graph pools every label (`large`: its span; `v4-edge`: its marker token) and scores it against the text. `v4-edge`'s calibrator then rescales each label's score, as Knowledgator's own pipeline does.
- **Mapping to question types.** The mapping of `choice`, `score` and `noul` onto labels is Ollaya's, chosen by testing on typed-decisions.
- **Weights.** They are Knowledgator's own `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256.
- **Parity.** Ollaya's Rust runtime matches the Python reference on CPU and CUDA: the same decision on every test question.
- **Apple silicon.** `large` runs on the GPU through MLX; `v4-edge` runs on the CPU.

## Limits

- **Yes/no questions without criteria are its weakest point.** Such a question is scored as one label with a sigmoid, and it can be confidently wrong. Give `noul` questions `criteria` with both `true` and `false` descriptions, or use `nli` for them.
- **Option count.** A question with more options than fit in 1,024 tokens is rejected with `TOO_MANY_OPTIONS`.
- **Calibration.** `large` is not calibrated. `v4-edge`'s calibrator was trained by Knowledgator; its calibration error on typed decisions is {{result gliclass:v4-edge typed.ece 3}}, against {{result gliclass:large typed.ece 3}} for `large`. Fit a `CALIBRATION` layer before you rely on thresholds.
