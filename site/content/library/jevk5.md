JevK5 is an open decision model by alibiserikbay, fine-tuned from Qwen3.5-4B and released under Apache-2.0. The author publishes it as GGUF files, and Ollaya runs the 4B Q8_0 file as it is, on llama.cpp. JevK5 puts the state, the question and the options into one JSON message, labels the options `A` to `P`, and reads the probability of each letter as the next token. It never generates text.

> Needs Ollaya 0.7.2 or newer, the first release that runs the `jevk5-v1` prompt. Update first (`ollaya --version`); an older version downloads the weights and then fails to load them.

## Models

| Tag | Base | Weights | JevBench public (231), author's run |
|---|---|---|---|
| `jevk5:latest`, `jevk5:4b` | Qwen3.5-4B, JevK5 v0.3 | Q8_0 GGUF, 4.5 GB | easy 1.000, standard 0.944, hard 0.784 |

The accuracies are the author's, from the [JevK5-GGUF card](https://huggingface.co/alibiserikbay/JevK5-GGUF): the Q8_0 file gives the same answer as the bf16 model on 229 of the 231 public JevBench items, with the same per-tier accuracy. They are the author's own runs, not official JevBench results. On typed-decisions (all 400 states, 2,000 questions, argmax against the majority label), measured by Ollaya, `jevk5:4b` scores {{result jevk5:4b typed.accuracy 3}}, with a calibration error (ECE) of {{result jevk5:4b typed.ece 3}} at the shipped temperature.

## Usage

```shell
ollaya run jevk5 --preset triage "My order never arrived and support ignores me. Refund me today or I'm switching to your competitor."
```

Point any TypeSafe client at `http://localhost:11435` and set the model to `jevk5`.

## Speed

- **RTX 5090, through the HTTP API (Ollaya Bench):** the triage preset (five questions) on a short customer message takes {{result jevk5:4b gpu.p50_ms 0}} ms at the median.
- **RTX 4090, in the runner:** on the parity fixtures, whose states are longer, five questions take about 288 ms at the median. Every question is its own pass over the whole prompt (see below), so the time grows with the state times the number of questions.
- **CPU:** not measured by Ollaya. The author reports about 0.6 s per short decision on a 32- or 48-thread CPU server.

## How it works

- **Prompt.** Ollaya builds JevK5's prompt exactly as the author's runtime does (`jevk5/prompt.py` in [allebee/jevk5](https://github.com/allebee/jevk5) v0.3.3): a fixed system instruction, then the state, the question and the options as JSON, in Qwen3.5's chat template with thinking switched off.
- **One pass per question.** Qwen3.5 mixes attention with recurrent layers, whose cache cannot be cut back to a shared state prefix, so each question is evaluated from the start, as the author's client does. The same request always returns the same probabilities.
- **Options.** Up to 16 per question, labelled `A` to `P`. A yes/no question reads `true` as `A` and `false` as `B`.
- **Calibration.** Temperature 1.22 over the letters, the value the author gives for the v0.3 4B files.
- **Engine.** llama.cpp v0.5.0, ggml-org's own release build, runs inside Ollaya's runner process. It uses an NVIDIA GPU (CUDA) or an Apple silicon GPU (Metal) when the model fits, and the CPU otherwise.
- **Parity.** Ollaya's runner matches stock llama.cpp (`llama-server` of the same build, on the same file) on CUDA (RTX 4090): the same decision on all 593 test questions, probabilities within 1.6e-6. The prompts are byte-identical to the author's `jevk5.prompt` on all 450 fixture prompts. Not checked yet: the CPU, Metal and Windows.

## Limits

- **Options.** 1 to 16 per question. The author's runtime reads more options in several passes and combines them (a knockout); Ollaya does not, and answers a question with more than 16 options with `TOO_MANY_OPTIONS`.
- **Context.** State, question and options share 16,384 tokens. A longer prompt is rejected, not cut.
- **Control tokens.** Text you send can never become one of Qwen's control tokens: `<|im_end|>` in a state stays text. The author's runtime reads such text as a control token, so on those inputs only, Ollaya's tokens differ from the author's.
- **Size.** A 4B language model: a GPU makes a large difference.

## Weights and license

Apache-2.0. JevK5 is developed by alibiserikbay ([JevK5](https://huggingface.co/alibiserikbay/JevK5), [JevK5-GGUF](https://huggingface.co/alibiserikbay/JevK5-GGUF), runtime at [github.com/allebee/jevk5](https://github.com/allebee/jevk5)), from Qwen3.5-4B by the Qwen team, also Apache-2.0; its prompt and readout are adapted from SemIf by TheoLeeCJ (MIT). The author's NOTICE says part of v0.3's training questions were written by an OpenAI model and are subject to OpenAI's terms. Ollaya downloads the GGUF from the author's repository, pinned to a commit and checked against its sha256; it never re-hosts it. llama.cpp is MIT.
