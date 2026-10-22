---
license: apache-2.0
language:
- en
base_model: jhu-clsp/ettin-encoder-{{size}}
pipeline_tag: zero-shot-classification
library_name: onnx
tags:
- ollaya
- laya
- decision
- classification
- modernbert
{{tags}}
---

# {{title}}

{{title}} is a small English decision model by [Ollaya](https://ollaya.dev). Give it a **state** (a message, a
support ticket, a JSON object, an agent's next tool call) and **questions**, and it answers every question in one
forward pass with calibrated probabilities:

- `noul`: yes or no, as the probability of yes;
- `choice`: one of up to 255 options, each described in plain words;
- `score`: a level on a scale of 2 to 10 described levels.

{{variant}}

## Run it

**In a web page** (`npm install @ollaya/web`): the model downloads once ({{web_size}}) and runs on the device.

```js
import { fromHub, load } from '@ollaya/web';

const model = await load(fromHub('{{repo}}'));
const { answers } = await model.decide({
  state: "I was charged twice for order A-104. Refund the duplicate today or I'm cancelling.",
  questions: {
    refund: { type: 'noul', instructions: 'Is the customer asking for a refund?' },
    team: { type: 'choice', instructions: 'Which team handles this?',
            criteria: { billing: 'Payments, charges, refunds', technical: 'Bugs, outages, errors' } },
    urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['Can wait', 'This week', 'Today'] },
  },
});
```

**With [Ollaya](https://ollaya.dev)** (Linux, macOS, Windows; CPU, CUDA, Metal):

```sh
curl -fsSL https://ollaya.dev/install.sh | sh
ollaya run {{name}}:{{tag}} --ask "Is the customer asking for a refund?" < ticket.txt
```

or `POST http://localhost:11435/api/decide` with `{"model": "{{name}}:{{tag}}", "state": …, "questions": …}`
([API](https://ollaya.dev/docs/api)). Ollaya also speaks TypeSafe's `/v1/systemone`, so Jev clients work unchanged.

**In Python** with [`laya`](https://pypi.org/project/laya/) (`pip install laya`):

```python
import laya

agent = laya.load("{{repo}}")
agent.system_one("I was charged twice for order A-104.", {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
})
```

The browser, Ollaya and `laya` give the same answers: Ollaya checks its runtime and the browser's against this
checkpoint's PyTorch outputs on 483 questions (every decision the same, probabilities within 3e-6).

## Results

| Model | Weights | typed-decisions | Public benchmark |
|---|---|---:|---:|
{{results}}

- **typed-decisions**: accuracy on the 400 rows of `LocalLLaMA/typed-decisions`' test split, which no model here
  trained on.
- **Public benchmark**: macro accuracy over the 3,880 records of the eleven datasets Ollaya's bench uses (MNLI,
  BoolQ, PAWS, VitaminC, civil_comments, Aegis 2.0, HelpSteer2, PubMedQA, SQuAD 2.0, MASSIVE, SummEval).
  **In-domain**: the model also trained on the *train* splits of ten of them (never their test records; rows that
  share any text with a test record were dropped). **Zero-shot**: trained without them.
- For scale: `pplx-decider:flash`, the 4B decoder this model learned from, scores 0.726 and 0.824 zero-shot;
  0.8B decoders score 0.46–0.51 and 0.61–0.67.

## Training

- **Network**: Laya's (`laya` 0.3.7): the question, a `[MASK]` before every option and the state in one
  sequence of up to 512 tokens; a two-layer transformer head scores each option at its marker. Encoder:
  Ettin {{size}} (ModernBERT architecture, MIT).
- **Data**: about 350,000 English questions from permissively licensed decision datasets (Apache-2.0, MIT,
  CC BY, CC BY-SA, CC0, BSD; every source and its license is in `DATA_SOURCES.tsv`), plus 180,000 from the
  benchmark datasets' train splits. Rows from datasets with a non-commercial or unknown license were dropped.
- **Labels**: each question's gold answer where the data has one, mixed with the full probability distribution
  of `pplx-decider:flash` (Apache-2.0), computed through Ollaya.
- **Recipe**: soft cross-entropy plus the ranked probability score on score questions; options shuffled half the
  time; one temperature per question type fitted on held-out data. {{recipe}}

## Limits

- English only. Other languages are not refused but are not measured.
- 512 tokens for the state and the question together: messages, tickets and small JSON states, not long
  documents or agent traces.
- `score` questions are the weakest kind for every model of this size; read the probabilities, not only the
  expected level.
- Probabilities are calibrated on average, not per domain: check them on your own data before acting on them
  automatically, and keep a person in the loop for decisions that matter.

## License

Apache-2.0. The encoder started from Ettin (MIT; see `NOTICE`). The training data's sources and their licenses
are in `DATA_SOURCES.tsv`; CC BY and CC BY-SA sources are credited there.
