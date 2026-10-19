CLM is a contrastive decision model by [Contrastive-LM](https://huggingface.co/Contrastive-LM), released under Apache-2.0. Instead of reading answer logits, it embeds the state and every option with the Qwen3-8B encoder, projects them with two small trained heads (one for states, one for options) and picks the option most similar to the state. Options and questions are embedded on their own, so Ollaya caches them: a question you ask again costs only its state.

> Needs Ollaya 0.7.4 or newer, which runs the `clm-v1` layout. Update first (`ollaya --version`); an older version downloads the weights and then refuses to load them.

## Models

| Tag | Encoder | Params | Typed-decisions accuracy |
|---|---|---|---|
| `clm:latest`, `clm:8b` | Qwen3-8B (last-token embedding) | 8.2B | {{result clm:8b typed.accuracy 3}} |

Typed-decisions accuracy is the argmax against the majority label on all 400 typed-decisions states. On these structured workflow states CLM is close to chance and leans towards "yes"; its authors report its strength on computer-use, gaming and tool-calling states, and as a fine-tuned verifier. For general triage, `winnow:e4b` ({{result winnow:e4b typed.accuracy 3}}) is the better pick.

## Usage

```shell
ollaya run clm --preset triage "Third time this year you've double-charged me. Refund it today or I'm cancelling and moving to a competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `clm`. Give choice criteria as an object of option to description (`{"refund": "wants money back", ...}`); CLM embeds each description, so descriptions matter more than labels.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** the triage preset (five questions) on a new short message takes {{result clm:8b gpu.p50_ms 0}} ms at the median once the questions are cached, and under a millisecond for a repeated request. With nothing cached, five questions take about 0.5 s.
- **Memory:** the encoder's weights stay BF16, about 16 GB; use a GPU with 20 GB or more. On a 24-core CPU a request takes about 13 s.

## How it works

- **Texts.** The state is rendered as prose (JSON objects as `key: value` fields), followed by the question's instructions. Each option is its description; a noul question compares "Yes. This is true: …" with "No. This is false: …".
- **Scores.** An option's logit is 100 × the cosine between its projection and the state's; the answer is the softmax over the options. There is no other calibration.
- **Weights.** The Qwen3-8B weights download from Qwen's repository and the heads from Contrastive-LM's `CLM_v0.1-8B.pt`, both pinned to a commit and verified by sha256. Ollaya hosts only the ONNX graph (7 MB).
- **Parity.** Ollaya's Rust runtime matches the reference exactly on texts and tokens, and on the decision of every test question, on CPU and CUDA (probabilities within 3 × 10⁻⁵).

## Limits

- **Texts over 2,048 tokens are rejected** (422); upstream would cut them.
- **Choice criteria must be an object**, as upstream CLM requires.
