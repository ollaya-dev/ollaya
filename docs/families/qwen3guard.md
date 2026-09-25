# qwen3guard (`qwen3guard-gen-v1`): a fixed-preset guard model

**Qwen/Qwen3Guard-Gen-0.6B** (Apache-2.0, `LICENSE` in the repo) is a Qwen3-0.6B fine-tune that
*generates* a safety verdict:

```
Safety: Safe|Unsafe|Controversial
Categories: <comma-separated list or None>
```

Its whole task prompt (policy, the 9 categories, the output format) is hard-coded in its chat template.
The user message is only the text to judge.

## Decision: preset only (no arbitrary questions)

Qwen3Guard cannot answer arbitrary typed questions:

- the template ignores custom instructions;
- its only "options" are its own labels.

It fits Ollaya as a **fixed-preset model** that ships an embedded question schema
(`decision.json.questions`, i.e. a `vnd.ollaya.questions` layer) and runs as `ollaya run qwen3guard
"<text>"`. Any other question gets a 422. The preset mirrors Laya's `guard` preset where the model can
express it:

| Question | Type | Option logits (runner → server) |
|---|---|---|
| `safety` | choice `[safe, controversial, unsafe]` | first-token logits at `Safety:` → `[ Safe, Cont, Unsafe]` |
| `unsafe` | noul | `[logsumexp(safe, controversial), unsafe]`: p(true) = p(Unsafe) |
| `unsafe_strict` | noul | `[safe, logsumexp(controversial, unsafe)]`: p(true) = p(Unsafe or Controversial) |
| `category` | choice over `none, violent, non_violent_illegal, sexual, pii, suicide_self_harm, unethical, politically_sensitive, copyright, jailbreak` | first-token logits after the teacher-forced `Safety: Unsafe\nCategories:` |

**Semantics to keep in mind.**
- `category` is the first category the model *would list if it judged the text unsafe*. It is a
  conditional distribution, meaningful only when `unsafe` is high. For a safe weather question it still
  spreads mass over categories.
- Multi-token labels are represented by their first token, which is unique in both candidate sets:
  ` Controversial` → ` Cont`, ` PII` → ` P`, ` Unethical` → ` Un`, and so on. That is P(first token), not
  the full sequence probability. After the first token the continuation is effectively forced.
- Response moderation (a `{prompt, response}` pair with `Refusal: Yes|No`) uses the second template
  branch. It is not in v1 because it needs teacher-forcing the safety line first.

## Recommended engine: ONNX (single forward)

- **Why ONNX.** The model is a 0.6B dense Qwen3. One request is two causal rows plus a 13-token logit
  gather, with no generation, so option (a) is cheap.
- **Export.** `llm_common/qwen3.py` is an explicit trunk with full attention and RoPE, with no sequence
  restriction.
- **Why not llama.cpp.** Only third-party GGUFs exist, and ONNX references the author's own safetensors.

### Contract

| Tensor | dtype | shape | meaning |
|---|---|---|---|
| `input_ids` | int64 | `[rows, seq]` | right-padded, any pad id |
| `last_pos` | int64 | `[rows]` | index of the last real token |
| `cand_logits` | float32 | `[rows, 13]` | `[0:3]` = ` Safe`, ` Cont`, ` Unsafe`; `[3:13]` = category first tokens (ids in `decision.json.candidates`) |

### Rows

```
head, tail = the chat template rendered around the user message (decision.json.templates, verbatim)
base = head + state_text + tail          # tail ends "<|im_start|>assistant\n<think>\n\n</think>\n\n"
r0   = tok(base + "Safety:")
r1   = tok(base + "Safety: Unsafe\nCategories:")
```

- `tok` is `tokenizer.json`, whole string, `add_special_tokens=false`, with special tokens parsed as the
  HF chat-template path does. Text like `<|im_end|>` in the state becomes a control token upstream too.
- `state_text` is a string verbatim, otherwise `json.dumps(ensure_ascii=False)`.
- The limit is 32k positions. There is no truncation: longer input gets a 422.

### Calibration

`[1, 1, 1]`: raw model probabilities. There is no upstream temperature.

## Files

| Layer | Source |
|---|---|
| `model.onnx` (5.2 MB) | derived, hosted by Ollaya |
| `model.safetensors` (BF16, 1.5 GB) | `Qwen/Qwen3Guard-Gen-0.6B@fada3b2f655b89601929198343c94cd2f64d93cc`; 310 tensors referenced. The one unused tensor is the stored copy of the tied `lm_head`; the graph reads the tied embedding rows. |
| `tokenizer.json` | same repo and commit, used as-is |
| `LICENSE` | same repo (Apache-2.0) |
| `decision.json`, `calibration.json` | derived, hosted by Ollaya |

## Parity

See "Parity results" at the end, from `families/qwen3guard/parity.py`: ONNX Runtime CPU vs transformers
fp32, which also writes `convert/out/goldens-qwen3guard-gen-0.6b.jsonl`.

## Quality notes

- Qwen reports state-of-the-art safety-benchmark results for the Gen series; see the model card and
  technical report. No independent benchmark was run here.
- Smoke results match expectations:
  - "How can I make a bomb?" → unsafe 0.999, violent;
  - weather → safe 1.0;
  - the prompt-injection edge state → controversial 0.80 with jailbreak as the category;
  - PII text → PII 0.98;
  - self-harm → suicide & self-harm 0.97.
- An angry Turkish billing complaint scored controversial 0.55, which suggests Qwen3Guard is stricter
  than a moderation gate needs. Use `unsafe` (loose) rather than `unsafe_strict` for gating.

## Attribution

Qwen3Guard is by the Qwen team (Alibaba Cloud), Apache-2.0: https://github.com/QwenLM/Qwen3Guard, arXiv
2510.14276.

## Parity results

The weightless graph on ONNX Runtime 1.30 CPU, against transformers fp32 (CPU). The report is
`convert/out/qwen3guard-gen-0.6b/parity.json`.

| | value |
|---|---|
| states | 74: 6 guard probes, 18 distinct edge and decoder states, 50 typed-decisions states |
| preset questions | 296 (4 per state) |
| token rows, port vs `apply_chat_template` + encode | identical, 148 / 148 |
| argmax agreement, ONNX vs fp32 | **100 %** |
| max \|Δ candidate logit\| | 3.6e-5 |
| max \|Δ probability\| | 3.8e-6 (safety), 8.9e-7 (unsafe), 3.8e-6 (unsafe_strict), 9.2e-6 (category) |

Goldens: `convert/out/goldens-qwen3guard-gen-0.6b.jsonl` holds the rows, `last_pos`, the 13 candidate
logits per row, option logits and probabilities.

## ONNX Runtime CUDA EP note (measured)

- **The bug.** ORT 1.30 CUDA EP runs these raw dynamo exports only with
  `SessionOptions.enable_mem_reuse = false`. With memory reuse on, `Reshape` reads shape values from
  reused buffers and fails, for example with "requested shape {3,3,-1,128}". The CPU EP is unaffected;
  every parity number above is from the CPU EP.
- **Verified workaround.** `enable_mem_reuse=false` matches CPU to 1e-5, checked on the Qwen3Guard graph.
- **Alternative.** Re-export with `OLLAYA_FOLD_LIMIT=64`, which runs the onnxscript optimizer with
  folding capped at 64 elements. The CUDA EP then works with default options (checked on Qwen3Guard:
  CUDA vs CPU 1.3e-5). But it folds a few tiny weight-derived tensors into the graph (`-exp(A_log)`,
  16 floats per DeltaNet layer), and those graphs have not been through full parity.
- **Rust runner.** ORT 1.28 through the C API, which has no `enable_mem_reuse`. The runner uses the
  decider's CUDA session options instead (parallel execution mode, which reuses no buffers of the main
  graph); see `crates/ollaya-runner/src/decider.rs`. Measured below: CUDA matches the goldens.

## Rust runtime parity (measured)

`crates/ollaya-runner/examples/parity_qwen3guard.rs` against `goldens-qwen3guard-gen-0.6b.jsonl`: 74
states, 148 rows, 296 preset questions. It first checks the preset rules (part of the preset builds
only the rows it reads; other questions are rejected), then both rows' token ids and last positions,
then all 13 candidate logits of both rows and every preset question's option logits (tolerance 1e-3),
and the decisions.

```sh
cargo run --release -p ollaya-runner --example parity_qwen3guard -- convert/out/qwen3guard-gen-0.6b convert/out/goldens-qwen3guard-gen-0.6b.jsonl cpu
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_qwen3guard -- convert/out/qwen3guard-gen-0.6b convert/out/goldens-qwen3guard-gen-0.6b.jsonl cuda
```

Results on choso-wsl (24 cores; RTX 4090, CUDA 13), `ort` 2.0.0-rc.13 with ONNX Runtime 1.28:

| | CPU | CUDA |
|---|---|---|
| token rows and last positions identical | 148 / 148 | 148 / 148 |
| max \|Δ candidate logit\| (13 per row) | 5.7e-5 | 7.8e-5 |
| max \|Δ option logit\| | 5.7e-5 | 7.8e-5 |
| decisions agree | **100 %** (296 / 296) | **100 %** (296 / 296) |
| max \|Δ probability\| | 1.4e-5 (p99 4.3e-6) | 9.1e-6 (p99 4.3e-6) |

- **Serving.** The registry ships the preset as the model's built-in questions (a
  `vnd.ollaya.questions` layer, the same one a Modelfile's `QUESTIONS` writes), so `ollaya run
  qwen3guard "<text>"` and requests without `questions` get all four answers. A request may also send
  a subset of the preset unchanged; the category row is then built only when `category` is asked.
