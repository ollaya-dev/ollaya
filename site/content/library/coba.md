Coba is Ollaya's own family of English **decision models**: Laya's decision network on the Ettin encoders, from 75M to 1.1B parameters, distilled from `pplx-decider:flash` on about 350,000 permissively licensed questions. It reads a *state* (a message, a ticket, a JSON object, an agent's next tool call) and typed questions, and returns calibrated probabilities for every question in one forward pass. It never generates text. The ternary tags also run in a web page, on the visitor's device: [try one]({{SITE_ORIGIN}}/try).

## Models

| Tag | Weights | Params | Trained | Best for |
|---|---|---|---|---|
| `coba:1b` | fp32 | 1.1B | zero-shot | The most accurate size, on a GPU |
| `coba:latest`, `coba:400m` | fp32 | 421M | zero-shot | Close to the 1b at 40% of its size; fine on a CPU |
| `coba:400m-ternary` | ternary | 421M | in-domain | The most accurate browser model (147 MB) |
| `coba:150m-ternary` | ternary | 164M | in-domain | The browser default ([try it]({{SITE_ORIGIN}}/try)): 66 MB |
| `coba:68m-ternary` | ternary | 75M | in-domain | The smallest: 35 MB in a browser |

All tags read 512 tokens and English only. Each carries an fp16 and an fp32 graph over its own weights file: fp16 on a CUDA GPU, fp32 on the CPU and, through MLX, on the Apple GPU; the `-fp16` and `-fp32` tags pin one. A ternary tag's weights are −1, 0 or +1 times a scale per output row (as BitNet b1.58), stored as float32 for Ollaya; the 2-bit graph the browser runs is in the same Hugging Face repository.

**Zero-shot** tags trained without any of the public benchmark's datasets. **In-domain** tags also trained on the train splits of ten of its eleven datasets (never their test records), so their public-benchmark score is not a held-out measurement, and Ollaya Bench's edition score leaves those tasks out.

## Usage

```shell
ollaya run coba --ask "Is the customer asking for a refund?" "I was charged twice for order A-104. Refund the duplicate today."
```

Or call the local API with `"model": "coba"` (see [the API](/docs/api)). In a web page:

```js
import { fromHub, load } from '@ollaya/web'

const model = await load(fromHub('ollaya-dev/coba-150m-ternary'))
const { answers } = await model.decide({ state, questions })
```

## Accuracy and speed

Measured by Ollaya on an RTX 5090: typed-decisions is the 400-row test split of `LocalLLaMA/typed-decisions`, which no tag trained on; the public benchmark is Bespoke Labs' 3,880 human-labeled questions (macro accuracy); speed is five questions through the HTTP API.

| Tag | typed-decisions | Public benchmark | GPU | CPU |
|---|---|---|---|---|
| `coba:1b` | {{result coba:1b typed.accuracy 3}} | {{result coba:1b public.macro_accuracy 3}} zero-shot | {{result coba:1b gpu.p50_ms 0}} ms | {{result coba:1b cpu.p50_ms 0}} ms |
| `coba:400m` | {{result coba:400m typed.accuracy 3}} | {{result coba:400m public.macro_accuracy 3}} zero-shot | {{result coba:400m gpu.p50_ms 0}} ms | {{result coba:400m cpu.p50_ms 0}} ms |
| `coba:400m-ternary` | {{result coba:400m-ternary typed.accuracy 3}} | {{result coba:400m-ternary public.macro_accuracy 3}} in-domain | {{result coba:400m-ternary gpu.p50_ms 0}} ms | {{result coba:400m-ternary cpu.p50_ms 0}} ms |
| `coba:150m-ternary` | {{result coba:150m-ternary typed.accuracy 3}} | {{result coba:150m-ternary public.macro_accuracy 3}} in-domain | {{result coba:150m-ternary gpu.p50_ms 0}} ms | {{result coba:150m-ternary cpu.p50_ms 0}} ms |
| `coba:68m-ternary` | {{result coba:68m-ternary typed.accuracy 3}} | {{result coba:68m-ternary public.macro_accuracy 3}} in-domain | {{result coba:68m-ternary gpu.p50_ms 0}} ms | {{result coba:68m-ternary cpu.p50_ms 0}} ms |

`score` questions are the weakest kind for every encoder of this size; read their probabilities, not only the expected level.

## How it works

- **Network.** The same network as `laya` (0.3.7): the question, a `[MASK]` before every option and the state go into one sequence, and a two-layer head scores each option at its `[MASK]`, with one temperature per question type.
- **Training.** Each question's gold answer where the data has one, mixed with the full probability distribution of `pplx-decider:flash` (Apache-2.0), computed through Ollaya. Every data source and its license is listed in each repository's `DATA_SOURCES.tsv`; datasets with a non-commercial or unknown license were dropped.
- **Ternary.** Quantization-aware training from the fp model of the same size, distilled from it (logits, attention relations and every layer's hidden states).
- **Parity.** Ollaya's runtime matches laya's own loader on each checkpoint in fp32 on {{result coba:400m parity.cuda.questions 0}} questions, on the CPU, on CUDA and on the Apple GPU: identical token ids and markers, the same decision on every question, probabilities within 2e-5. The fp16 graphs agree on 99.6 % to 99.8 % of decisions; the others are near-ties.

## Limits

- **English only.** Other languages are not refused, but not measured.
- **State.** 512 tokens including the question: messages, tickets and small JSON states, not long documents. `/v1/systemone` answers a longer state with `STATE_TRUNCATED`; `/api/decide` answers from its first tokens.
- **Calibration.** Probabilities are calibrated on average, not per domain. Check them on your own data before acting on them automatically.

## Weights and license

Apache-2.0, by Ollaya. The base encoders, [Ettin](https://huggingface.co/jhu-clsp/ettin-encoder-150m) by JHU CLSP, are MIT. Ollaya downloads the weights from [ollaya-dev](https://huggingface.co/ollaya-dev) (`coba-1b`, `coba-400m`, `coba-400m-ternary`, `coba-150m-ternary`, `coba-68m-ternary`), each pinned to a commit and checked against sha256.
