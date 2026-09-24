Zero-shot NLI classifiers by [Moritz Laurer](https://huggingface.co/MoritzLaurer). Ollaya turns every option of a question into a hypothesis and asks the model whether the state entails it. Choice and score questions take the most-entailed option, and a yes/no question takes the probability that its statement is entailed.

## Models

| Tag | Backbone | Params | License | Typed-decisions accuracy |
|---|---|---|---|---|
| `nli:latest`, `nli:deberta-v3-large` | DeBERTa-v3-large | 435M | MIT | **0.548** |
| `nli:modernbert-large` | ModernBERT-large | 396M | Apache-2.0 | 0.515 |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `laya:en` scores 0.361 and `gliclass` 0.477. The labels have low annotator agreement, so compare the numbers to each other rather than reading them as absolutes.

## Usage

```shell
ollaya run nli --preset triage "I was charged twice for my subscription this month and want a refund."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `nli`.

## How it works

- **One sequence pair per option.** Each option is a premise–hypothesis pair (the state, then the option's hypothesis), so the cost grows with the number of options. A question with 20 options runs 20 sequences.
- **Batching.** All sequences in a request share one batched forward pass.
- **Hypothesis templates.** They come from each model's recommended phrasing and live in the model's `decision` layer.
- **Weights.** They are the authors' own `model.safetensors`, downloaded from Hugging Face, pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (about 5 MB).
- **Parity.** Ollaya's Rust runtime reproduces the Python reference exactly: the same token ids and the same decision on every test question, on CPU and CUDA.

## Limits

- **Uncalibrated.** These are zero-shot entailment models, not trained decision models. Their probabilities are not calibrated, so fit a `CALIBRATION` layer on your own data before you rely on thresholds.
- **English.** They work best on English text.
- **Training data licenses.** The model card notes that part of the DeBERTa model's training data carries non-commercial licenses. The author publishes a `-c` variant trained on commercially usable data only; Ollaya does not ship it yet.
