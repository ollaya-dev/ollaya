# jeb (`jebadiah-v1`)

Jason Brashear's **Jebadiah** models (Apache-2.0), built with AINode: rank-16 LoRAs merged into
Qwen3.5-4B, Qwen3.5-9B and Qwen3.8-27B, published by the authors as GGUF next to the bf16 weights.
Jebadiah answers a typed question from the next-token logits of its option labels after AINode's own
decision prompt, then applies a fitted temperature per question type. Requested in #18.

| Tag | Upstream file | Revision | Temperatures (choice / noul / score) |
|---|---|---|---|
| `jeb:4b` | `frontier-infra/jebadiah-4b-v2-GGUF` `jebadiah-4b-v2-Q8_0.gguf` | `7f671f9a` | 1.1167 / 1.3319 / 0.8312 |
| `jeb:9b`, `jeb:latest` | `frontier-infra/jebadiah-9b-v2-1-GGUF` `jebadiah-9b-v2-1-Q8_0.gguf` (v2.1) | `2182dd8d` | 1.0467 / 1.0284 / 0.6805 |
| `jeb:27b` | `frontier-infra/jebadiah-27b-v2-1-GGUF` `jebadiah-27b-v2-1-Q4_K_M.gguf` (v2.1) | `ed71f41b` | 1.0608 / 1.0405 / 0.6524 |

Q8_0 is the file the authors checked against bf16 (4B 256/260, 9B 257/260 held-out answers the same for v2).
The 27B ships as Q4_K_M (17 GB) so it fits a 24 GB GPU; its Q8_0 is 29 GB.

**v2.1 (COB-560).** The authors' v2 cards now say "This is an earlier version. For new work, use v2.1", so
`jeb:9b` and `jeb:27b` moved to v2.1 (there is no 4B v2.1). Until then they were
`frontier-infra/jebadiah-9b-v2-GGUF@adaec6b3` (`jebadiah-9b-v2-Q8_0.gguf`, temperatures 1.1863 / 1.0903 / 0.8329)
and `frontier-infra/jebadiah-27b-GGUF@7451e611` (`jebadiah-27b-Q4_K_M.gguf`, 1.2321 / 1.297 / 0.7558); their
manifests stay in the registry by digest. v2.1 is the same base, the same LoRA hyperparameters and the same prompt
contract, trained on 21,190 questions from about 90 sources (v2: 15,813), still including typed-decisions' train
split, with temperatures refit on a new calibration split. The tokenizer, the chat template and the 68 labels are
v2's; the GGUFs were converted with llama.cpp build 11549 and load and match on the pinned b11146 (below). The
authors' reported v2.1 results use their T1e recipe (the payload repeated in the prompt, and for choices the
forward and reversed option orders blended), which neither their standalone scripts nor Ollaya run.

## Prompt

`convert/ollaya_convert/families/jebadiah/ref.py` ports the authors' `scripts/jebadiah_prompt.py` and
`scripts/ainode_prompt_verbatim.py` (AINode's renderer at `e5c08938`, prompt sha256 `d2660ebe…`):

```text
system = "You are a decision function. Answer with the single letter of the best option and nothing else."
user   = "STATE:\n" + serialize_state(state) + "\n\nQUESTION: " + instructions.strip() + "\n\nOPTIONS:\n"
         + "A. opt_0\nB. opt_1\n..." + "\n\nAnswer with the label of one option and nothing else."
```

- **State.** A string verbatim; anything else compact JSON with sorted keys; `null` as nothing.
- **Options.** A noul reads `true` (A) then `false` (B); a choice its keys in order; a score its levels
  (a list, or an object of level: description). An option with a description reads
  `name: description`, the description on one line.
- **Labels.** `A`..`Z`, `AA`, `AB`, ... (68 single tokens on the Qwen3.5 tokenizer, `A`..`BP`). A
  wider question uses the authors' extended alphabet: `A`..`Z`, then every single-token two-letter
  string in order, up to TypeSafe's 255 options.
- **Template.** Qwen's chat template with thinking off, the same bytes the authors' renderer produces.

`python -m ollaya_convert.families.jebadiah.check <snapshot>` renders every question of the shared case
set both ways: 1,349 prompts identical to the authors' `Renderer`, the same 33 rejections, for v2 and for both
v2.1 snapshots. v2.1's renderer differs in one hunk: when a prompt is still over the budget after the state is cut
away, it raises instead of returning it (one case, `edge/many_options_77`, whose 77 options alone pass 2,048
tokens; Ollaya rejects it either way).

## Differences from upstream

- **Long prompts are rejected, not cut.** The authors' renderer cuts the state when a prompt passes
  2,048 tokens and appends `[truncated]`; Ollaya never answers from a cut state (#16), so such a
  question is a 400.
- **Control tokens.** Specials are parsed only in the template pieces, as for `jevk5-v1`: `<|im_end|>`
  in a state stays text. On requests without control-token text the tokens are the authors'.
- **List-form choices.** A choice given as a list of labels reads as `{label: null}`, as for every
  Ollaya model; the authors' route rejects it.
- **One pass per question.** Qwen3.5's recurrent layers cannot be cut back to a shared prefix, so every
  question is one cold pass (`plan: cold`), as with `jevk5-v1`.

## Parity (measured 2026-09-30)

Goldens: `export_llama.py jebadiah` sends every test request (40 typed-decisions rows, Laya's and the
decoder edge cases, the extra cases) through `ref.py` to a stock `llama-server` of the pinned build
(b11146) with the authors' file, one cold pass per question. 123 cases, 19 rejected by both, 494
questions.

| Model | Device | Decisions | Option logits max | Probabilities max | Five questions (runner, p50) |
|---|---|---|---|---|---|
| 4b Q8_0 | CUDA, RTX 4090 | 494/494 | 7.6e-6 | 2.1e-6 | 91 ms |
| 9b Q8_0 | CUDA, RTX 4090 | 494/494 | 7.7e-6 | 2.1e-6 | 118 ms |
| 27b Q4_K_M | CUDA, RTX 4090 | 494/494 | 7.7e-6 | 2.4e-6 | 287 ms |

v2.1 (2026-10-24, hope; the 4B row above is still v2's):

| Model | Device | Decisions | Option logits max | Probabilities max |
|---|---|---|---|---|
| 9b v2.1 Q8_0 | CUDA, RTX 5090 | 494/494 | 7.6e-6 | 2.8e-6 |
| 27b v2.1 Q4_K_M | CUDA, RTX 5090 | 494/494 | 7.6e-6 | 2.8e-6 |
| 9b v2.1 Q8_0 | CPU, Threadripper 3960X (smoke cases, goldens replayed on the CPU) | 21/21 | 5.6e-6 | 2.8e-6 |

```sh
uv run python -m ollaya_convert.families.llm_common.export_llama jebadiah --server <b11146>/llama-server \
    --gguf <snapshot>/jebadiah-9b-v2-Q8_0.gguf --slug 9b-q8_0 --repo frontier-infra/jebadiah-9b-v2-GGUF \
    --revision adaec6b3d1f0421706deb49fa275ab49093982ff --file jebadiah-9b-v2-Q8_0.gguf \
    --temperatures <snapshot>/temperatures.json --n-ctx 4096
OLLAYA_LIBRARY_PATH=/usr/local/lib/ollaya cargo run --release -p ollaya-runner --features ollaya-runner/cuda \
    --example parity_llama -- convert/out/jebadiah-9b-q8_0 convert/out/jebadiah-9b-q8_0/goldens-cuda.jsonl cuda --latency
```

## Quality

Typed-decisions (all 400 test states, 2,000 questions, argmax against the majority label, measured here
through the goldens' path, `quality_llama.py`):

| Model | Accuracy | ECE (shipped temperatures) |
|---|---|---|
| 4b Q8_0 | 0.801 | 0.137 |
| 9b Q8_0 | 0.786 | 0.125 |
| 27b Q4_K_M | 0.802 | 0.132 |

**These are in-distribution numbers.** Jebadiah's training pool includes the typed-decisions train split
(`data/convert_data.py` in getainode/jebadiah: "LocalLLaMA/typed-decisions train"), so, like
`laya:typed-decisions`, its typed-decisions score is not comparable with the zero-shot models' and the
catalog leaves it out of the comparison. The authors report held-out results on the Jevals sets, Nimble's
public subsets and Kev's transfer-v4 in their repository.

## Limits

- **Prompt.** Up to 2,048 tokens per question, state included (the authors' budget).
- **Options.** A choice needs 2 or more options, a score 2 to 10 levels; up to 255 options.
- **Instructions.** Must be a non-empty string (AINode's rule): structured instructions are rejected.
