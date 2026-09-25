Laya is a family of open **decision models** by [Convai Innovations](https://huggingface.co/convaiinnovations), released under the Apache-2.0 license. A decision model reads a *state* (a message, an email, a support ticket, a JSON object) together with a set of typed questions, and returns a typed answer with calibrated probabilities for every question in a single forward pass. It never generates text.

## Models

| Tag | Backbone | Params | Context | Languages | Best for |
|---|---|---|---|---|---|
| `laya:latest` | router | – | 512 / 1024 | auto | Sends English text to `en` and everything else to `multilingual` |
| `laya:en` | ModernBERT-large | 421M | 512 | English | Guardrails, email triage |
| `laya:multilingual` | mmBERT-base | 322M | 1024 | 100+ | Non-English and mixed-language input; up to ~2.2× faster on batched calls |
| `laya:typed-decisions` | ModernBERT-large | 421M | 1024 | English | Typed-decisions workflows (fine-tuned) |

Each model carries an fp16 and an fp32 graph over one weights file. It loads fp16 on a CUDA GPU and fp32 on the CPU; the `-fp16` and `-fp32` tags pin one, for example `laya:en-fp32`. `laya` routes Turkish and other non-English text to `laya:multilingual`, and pulling it pulls both targets.

## Usage

```shell
ollaya run laya --preset triage "I was charged twice for my subscription this month and want a refund."
```

Or call the local API:

```shell
curl http://localhost:11435/api/decide -d '{
  "model": "laya",
  "state": "I was charged twice for my subscription this month. Please refund the second charge.",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "technical": "Bugs, errors and outages",
        "account": "Login, profile and settings"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "How urgent is this ticket?",
      "criteria": ["Can wait", "Needs attention this week", "Needs attention today"]
    },
    "refund": {
      "type": "noul",
      "instructions": "The customer asks for money back."
    }
  }
}'
```

```json
{
  "model": "laya:en",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.7781,
      "probabilities": {"billing": 0.8521, "technical": 0.0611, "account": 0.0868}
    },
    "urgency": {
      "type": "score",
      "score": 1.1982,
      "confidence": 0.3418,
      "legend": {"0": "Can wait", "1": "Needs attention this week", "2": "Needs attention today"},
      "probabilities": {"0": 0.1203, "1": 0.5612, "2": 0.3185}
    },
    "refund": {"type": "noul", "noul": 0.9127}
  },
  "usage": {"input_tokens": 118, "output_tokens": 0},
  "routing": {"router": "laya:latest", "model": "laya:en", "route": "english", "reason": "English Latin text"},
  "state_truncated": false,
  "done_reason": "decide",
  "created_at": "2026-09-24T09:30:12.418Z",
  "total_duration": 18734512,
  "load_duration": 0,
  "eval_duration": 16302117
}
```

`model` is the checkpoint that answered, and `routing` says why. See the [API reference](/docs/api#decide) for every field, and [TypeSafe compatibility](/docs/typesafe-compatibility) for `/v1/systemone`.

## Question types

| Type | `criteria` | Answer |
|---|---|---|
| `choice` | Object of option → description, or a list of labels; 2–255 options | `choice`, `confidence`, `probabilities` |
| `score` | Ordered list of 2–10 levels | `score` (expected level), `confidence`, `legend`, `probabilities` |
| `noul` | Optional `{"true": "…", "false": "…"}` | `noul`: the probability that the statement holds |

`confidence` is the normalized top probability, (K · p<sub>max</sub> − 1) / (K − 1) for K options: 0 when every option is equally likely, 1 when one option has all the probability. Every option needs room in the context: about 125 options fit `laya:en`, and 250 fit `laya:multilingual`.

## Performance

On an RTX 4090 at fp16, a request with five questions takes 16.3 ms on `laya:en` and 9.1 ms on `laya:multilingual` (median, Ollaya's Phase 0 benchmark).

Figures published on the Laya model card, measured on an NVIDIA Tesla T4:

| | Latency, 1 question | Latency, 10 questions (batched) | Calibration error (ECE) |
|---|---|---|---|
| Laya (English) | 39.5 ms | 158.6 ms | 0.081 after temperature fitting |
| Laya multilingual | 32.8 ms | 72.3 ms | – |
| TypeSafe Jev | 236–276 ms p50 (third-party) | – | 0.246 |

On batched calls `laya:multilingual` is up to ~2.2× faster than `laya:en`; for a single question the two are close. `laya:typed-decisions` reaches 0.766 accuracy on typed-decisions, against 0.727 published for Jev 1.13. Jev latencies come from [AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks) and [nibzard/decision-model-benchmark](https://github.com/nibzard/decision-model-benchmark); setups differ.

## ONNX export

Ollaya runs Laya as ONNX. Across 2,383 questions per checkpoint (`en`, `multilingual` and `typed-decisions`), the ONNX export chose the same answer as the PyTorch fp32 reference 100% of the time, with a maximum probability difference of 1.1 × 10⁻⁴. The fp16 graph, used on CUDA GPUs by default, can differ from fp32 on near-ties.

## Limitations

- **Zero-shot typed decisions.** The base checkpoints (`en`, `multilingual`) are near chance zero-shot on typed-decisions (0.362). Use `laya:typed-decisions` for those workflows.
- **Many options.** Choice questions with more than ~20 options are weaker: 0.425 on Banking77, against 0.870 for Jev.
- **Calibration.** The raw checkpoints are over-confident unless their temperatures are refit, and `laya:multilingual` ships without refit temperatures. For thresholds you rely on, refit calibration on your own labelled data and bake it in with `CALIBRATION` in a [Modelfile](/docs/modelfile).

## Weights and license

Apache-2.0. Laya is developed by Convai Innovations; see the [model card on Hugging Face](https://huggingface.co/convaiinnovations/laya). Ollaya downloads the weights from that repository, pinned to a commit and checked against sha256; it never re-hosts them.
