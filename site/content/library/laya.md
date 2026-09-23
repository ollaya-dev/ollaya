Laya is a family of open **decision models** by [Convai Innovations](https://huggingface.co/convaiinnovations), released under the Apache-2.0 license. A decision model reads a *state* — a message, an email, a support ticket, a JSON object — together with a set of typed questions, and returns a typed answer with calibrated probabilities for every question in a single forward pass. It never generates text.

## Models

| Tag | Backbone | Params | Context | Languages | Best for |
|---|---|---|---|---|---|
| `laya:latest` | router | — | 512 / 1024 | auto | Picks `en` or `multilingual` from the detected script and language |
| `laya:en` | ModernBERT-large | 421M | 512 | English | Guardrails, email triage |
| `laya:multilingual` | mmBERT-base | 322M | 1024 | 100+ | Non-English and mixed-language input; up to ~2.2× faster on batched calls |
| `laya:typed-decisions` | ModernBERT-large | 421M | 1024 | English | Typed-decisions workflows (fine-tuned) |

Tags without a suffix resolve to fp16 on a GPU and fp32 on CPU. Add `-fp16` or `-fp32` to choose explicitly, for example `laya:en-fp32`.

## Usage

```shell
ollaya run laya --preset triage "I was charged twice for my subscription this month."
```

Or call the local API:

```shell
curl http://localhost:11435/api/decide \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laya",
    "state": "I was charged twice for my subscription this month.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoices and refunds",
          "technical": "Bugs, errors and outages",
          "account": "Login, profile and settings"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this?",
        "criteria": ["Not urgent", "Normal", "Urgent"]
      },
      "refund": {
        "type": "noul",
        "instructions": "Is the customer asking for a refund?"
      }
    }
  }'
```

Example response:

```json
{
  "model": "laya",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.78,
      "probabilities": { "billing": 0.852, "account": 0.087, "technical": 0.061 }
    },
    "urgency": {
      "type": "score",
      "score": 1.2,
      "confidence": 0.34,
      "legend": { "0": "Not urgent", "1": "Normal", "2": "Urgent" },
      "probabilities": { "0": 0.12, "1": 0.56, "2": 0.32 }
    },
    "refund": { "type": "noul", "noul": 0.91 }
  },
  "routing": { "model": "laya:en" },
  "usage": { "input_tokens": 71, "output_tokens": 0 },
  "total_duration": 41250000,
  "load_duration": 0
}
```

`/api/decide` returns the TypeSafe response shape plus `routing` (the checkpoint the `laya` router picked) and timings in nanoseconds. See the [API reference](/docs/api#decide).

## Question types

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | Object of option → description, up to 255 options | `choice`, `confidence`, `probabilities` |
| `score` | Ordered list of 2–10 levels | `score` (expected value), `confidence`, `legend`, `probabilities` |
| `noul` | Optional `{"true": "…", "false": "…"}` | `noul`: probability that the answer is true |

`confidence` is the normalized top probability, (K · p<sub>max</sub> − 1) / (K − 1) for K options: 0 when every option is equally likely, 1 when one option has all the probability.

## Performance

Figures published on the Laya model card, measured on an NVIDIA Tesla T4:

| | Latency, 1 question | Latency, 10 questions (batched) | Calibration error (ECE) |
|---|---|---|---|
| Laya (English) | 39.5 ms | 158.6 ms | 0.081 after temperature fitting |
| Laya multilingual | 32.8 ms | 72.3 ms | — |
| TypeSafe Jev | 236–276 ms p50 (third-party) | — | 0.246 |

On batched calls `laya:multilingual` is up to ~2.2× faster than `laya:en`; for a single question the two are close. `laya:typed-decisions` reaches 0.766 accuracy on typed-decisions, against 0.727 published for Jev 1.13. Jev latencies come from [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) and [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark); setups differ.

## ONNX export

Ollaya runs Laya as ONNX. Across 2,383 questions per checkpoint (`en`, `multilingual` and `typed-decisions`), the ONNX export chose the same answer as the PyTorch fp32 reference 100% of the time, with a maximum probability difference of 1.1 × 10⁻⁴.

## Limitations

- **Zero-shot typed decisions.** The base checkpoints (`en`, `multilingual`) are near chance zero-shot on typed-decisions (0.362). Use `laya:typed-decisions` for those workflows.
- **Many options.** Choice questions with more than ~20 options are weaker: 0.425 on Banking77, against 0.870 for Jev.
- **Calibration.** The raw checkpoints are over-confident unless their temperatures are refit. For thresholds you rely on, refit calibration on your own labelled data and bake it in with `CALIBRATION` in a [Modelfile](/docs/modelfile).

## License

Apache-2.0. Laya is developed by Convai Innovations — see the [model card on Hugging Face](https://huggingface.co/convaiinnovations/laya). Ollaya packages the published weights for local serving.
