Von is an open decision model by Victor Hugo Panisa, built on ModernBERT-large and released under Apache-2.0. It reads a question's instructions and the state, then every option with its own `[MASK]` marker, all in one sequence, and scores each option at its marker. Every question is one forward pass, whatever its number of options.

## Models

| Tag | Backbone | Params | Context | Typed-decisions accuracy |
|---|---|---|---|---|
| `von:latest`, `von:1.1` | ModernBERT-large | 395M | 8,192 tokens | 0.447 |

Accuracy is the argmax against the majority label on all 400 typed-decisions states. For comparison, `decider` scores 0.591, `nli` 0.548, `gliclass` 0.477 and `laya:en` 0.361. The labels have low annotator agreement, so compare the numbers to each other rather than reading them as absolutes. On the author's JevBench harness Von 1.1 scores 0.938 (easy), 0.653 (standard) and 0.351 (hard).

## Usage

```shell
ollaya run von --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `von`.

## Speed

- **RTX 4090, end to end:** a five-question request with a short state takes about 23 ms at the median.
- **CPU:** the same request takes about 0.76 s on a 24-core x86 CPU.
- **State length:** every question reads the whole state, and attention cost grows faster than the sequence. Five questions about a 600-token state take about 0.17 s on the GPU and 5.6 s on the CPU; about a 2,200-token state, 0.78 s and 25 s. A state near 7,500 tokens takes about 7 s on the GPU. On the CPU a full 8,192-token sequence needs about 25 GB of memory.

## How it works

- **One sequence per question.** The instructions and the state come first, then `[SEP]`, then every option's description after its own `[MASK]`. An option is read from its description; a bare label is read as it is.
- **State as text.** A JSON state is written the way Von was trained on it: one `key: value` line per top-level key, with Python's rendering of nested values.
- **Yes/no without criteria.** A `noul` question without `true`/`false` descriptions adds a second sequence without the state. It measures the question's own lean towards yes or no, which is taken off the answer, as Von does.
- **Calibration.** The temperature is computed per question from the answer's entropy, the state's length and the number of options, with Von's own fitted coefficients.
- **Weights.** Von's trained weights exist only in the author's `option_marker.pt`, a PyTorch archive whose tensors are stored uncompressed. Ollaya downloads it from Hugging Face, pinned to a commit and verified by sha256, and the ONNX graph reads the tensors in place by byte offset. Nothing in it is unpickled or executed, and Ollaya hosts only the graph (3 MB).
- **Parity.** Ollaya's Rust runtime reproduces upstream Von computed in float64: the same token ids and option markers, the same decision on every test question, and logits within 4.4e-4 on x86-64 CPU and CUDA. On Apple silicon's CPU one row is 1.1e-3 off, and every decision is still the same.

## Limits

- **English.** Von is trained on English.
- **Context.** A question's instructions, state and options share 8,192 tokens. A longer state is cut to fit; instructions and options that do not fit on their own are rejected.
- **Long policy text.** The model card calls Von weak on long, multi-hop policy text, and quality drops visibly near the context limit.
- **`[MASK]` in the input.** The literal text `[MASK]` in a state, instruction or option is replaced by a space, because Von would read it as an option marker.
- **Von 1.2.** The author has since released Von 1.2, which changes how options attend to each other. It needs a new graph; `von:1.1` stays pinned to 1.1 until then.

## Weights and license

Apache-2.0. Von is developed by Victor Hugo Panisa ([model card on Hugging Face](https://huggingface.co/wfzyx/von), code at [github.com/wfzyx/von](https://github.com/wfzyx/von)). It is built on ModernBERT-large by Answer.AI and LightOn, also Apache-2.0. Ollaya downloads the weights from the author's repository, pinned to a commit and checked against sha256; it never re-hosts them.
