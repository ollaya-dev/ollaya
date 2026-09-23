---
title: Modelfile
description: Bake a question set, calibration, precision and license into a model you can run by name.
order: 4
---

# Modelfile

A Modelfile describes a derived model: a base model plus the questions it should always ask, and optionally a refit calibration, a pinned precision, a license and a description. Build it with `ollaya create`, then run it by name with just a state.

```shell
ollaya create triage -f Modelfile
ollaya run triage "I was charged twice for my subscription this month."
```

## Example

```dockerfile
# Support ticket triage
FROM laya:en

QUESTIONS """
{
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
    "instructions": "Does the customer ask for money back?"
  }
}
"""

CALIBRATION ./calibration.json
PARAMETER precision fp32
DESCRIPTION Support ticket triage
LICENSE ./LICENSE
```

## Instructions

| Instruction | Required | Value |
|---|---|---|
| `FROM` | yes | The base model, e.g. `laya:en`, or a router such as `laya` |
| `QUESTIONS` | no | The built-in questions: the same object as `questions` in the [API](/docs/api#questions) |
| `CALIBRATION` | no | Temperatures that replace the base model's |
| `PARAMETER` | no | `precision fp16` or `precision fp32`, to pin one graph |
| `DESCRIPTION` | no | One line, shown by `ollaya show` and `/v1/models` |
| `LICENSE` | no | License text for the derived model |

- **Syntax.** Directives are case-insensitive; `#` starts a comment line.
- **Values.** `QUESTIONS`, `CALIBRATION` and `LICENSE` take a path relative to the Modelfile (`~/` works), inline JSON on one line, or a block between `"""` spanning several lines.
- **Unknown input.** Any other directive or parameter is an error.
- **Pulling.** `ollaya create` pulls `FROM` first when it is not on this machine.

Anything you leave out is inherited from the base model. When a request sends its own `questions`, they replace the built-in ones for that request.

### FROM

The model to build on. A router works too; the derived model then routes like its base:

```dockerfile
FROM laya
```

### QUESTIONS

The questions the model asks when a request brings none, validated like a decision request (1–256 questions):

```dockerfile
QUESTIONS ./questions.json
```

### CALIBRATION

Probabilities are only useful for thresholds if they are calibrated. Ollaya calibrates with temperature scaling: logits are divided by a temperature per question type and number of options. `CALIBRATION` replaces the base model's temperatures with ones refit on your own labelled data:

```json
{
  "temperature": [1.6, 1.25, 1.98],
  "temperature_by_options": {
    "choice:2": 1.9,
    "choice:3-5": 1.76,
    "score:3-5": 1.25,
    "noul:2": 1.98
  }
}
```

`temperature` holds one fallback per question type (choice, score, noul). `temperature_by_options` keys are `<type>:<2|3-5|6-10|11+>`, by the question's number of options.

### PARAMETER

The only parameter is `precision`. A bare model carries an fp16 and an fp32 graph and picks one when it loads (fp16 on a CUDA GPU, fp32 on the CPU). `PARAMETER precision fp32` pins the fp32 graph everywhere, for example to match an fp32 reference exactly.

```dockerfile
PARAMETER precision fp32
```

### DESCRIPTION

```dockerfile
DESCRIPTION Support ticket triage
```

### LICENSE

The license text shipped with the model and shown by `ollaya show --license`. When you build on an Apache-2.0 model such as Laya, keep its license and attribution.
