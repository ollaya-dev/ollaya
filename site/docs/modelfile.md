---
title: Modelfile
description: Bake a question schema, calibration and license into a derived model you can run by name.
order: 4
---

# Modelfile

A Modelfile describes a derived model: a base model plus the questions it should always ask, an optional refit calibration and a license. Build it with `ollaya create`, then run it by name with just a state.

```shell
ollaya create triage -f Modelfile
ollaya run triage "I was charged twice for my subscription this month."
```

## Example

```dockerfile
# Modelfile
FROM laya:en

QUESTIONS """
{
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
"""

CALIBRATION ./calibration.json

LICENSE """
Apache License, Version 2.0
"""
```

## Instructions

| Instruction | Required | Description |
|---|---|---|
| `FROM` | yes | The base model, e.g. `laya:en` or `laya:multilingual` |
| `QUESTIONS` | no | The question schema — the same object as `questions` in the [API](/docs/api#questions) |
| `CALIBRATION` | no | Calibration temperatures that replace the base model's |
| `PARAMETER` | no | A runtime parameter for the model |
| `LICENSE` | no | License text for the derived model |

Instructions are written in upper case. Multi-line values are wrapped in `"""`.

### FROM

The model to build on. Any local or pullable model works:

```dockerfile
FROM laya:multilingual
```

### QUESTIONS

The questions the model asks when you call it without any. It takes a JSON object, inline between `"""` or as a path to a `.json` file:

```dockerfile
QUESTIONS ./questions.json
```

### CALIBRATION

Decision models report probabilities, and those probabilities are only useful if they are calibrated. Ollaya calibrates with temperature scaling. `CALIBRATION` points to a JSON file with temperatures refit on your own labelled data, which replaces the base model's calibration:

```json
{
  "temperature_by_options": {
    "choice:2": 1.9,
    "choice:3-5": 1.8,
    "score:3-5": 1.3,
    "noul:2": 2.0
  }
}
```

Keys are question type and option count; values are the temperatures to divide logits by. Refitting on data from your own workflow is the most reliable way to make thresholds like "escalate when P > 0.9" behave.

### PARAMETER

Sets a runtime parameter for the model:

```dockerfile
PARAMETER <name> <value>
```

The list of supported parameters will be documented here before the first release.

### LICENSE

The license text shipped with the model and shown by `ollaya show`. When you build on an Apache-2.0 model such as Laya, keep its license and attribution.
