# clef (`clef-joint-v1`)

Cloudflare's **Clef** models ([Cloudflare/clef-flash](https://huggingface.co/Cloudflare/clef-flash),
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef), Apache-2.0, released 2026-09-30) are fully
post-trained Qwen models with a **joint schema head**: a small transformer that reads the backbone's last hidden
states, routes evidence from the state to every question and scores every option of every question together,
in one forward pass per request. They never generate text, and the probabilities are a plain softmax of the
head's logits (no fitted temperature). The repository's `joint_schema_model.py` holds the authors' encoder, model
and `/v1/systemone` adapter; Clef speaks the Jev/SystemOne wire format.

| Tag | Weights | Status in Ollaya |
|---|---|---|
| `clef:flash`, `clef:latest` | `Cloudflare/clef-flash@17f0b0ad`: Qwen3.5-9B post-trained, four BF16 shards, and `joint_head.safetensors` (BF16, 122 tensors) | **converted, ONNX**, text only (weights stay BF16 in memory) |
| `Cloudflare/clef` | Qwen3.8-27B post-trained, twelve shards (about 54 GB in BF16) | not converted: it does not fit the GPUs the parity gate runs on |

The vision tower (in the last shard) is not exported, so requests with images or video are rejected; text and
JSON states work as upstream.

## The sequence

`convert/ollaya_convert/families/clef/layout.py` ports `encode_record` (and `systemone`'s request checks) at the
pinned revision. All questions share one sequence:

```text
ids    = tok(prefix) ⧺ tok(render(state))[..budget] ⧺ schema ⧺ tok(suffix)
schema = tok("\n\nSCHEMA FIELDS:\n") ⧺ for question i:
           tok("\nFIELD {i+1}\nID: {id}\nTYPE: {type}\nINSTRUCTION: ") ⧺ tok(render(instructions or id))    question span
           ⧺ tok("\nALLOWED OPTIONS:\n")
           ⧺ for option j: tok("OPTION {j+1}: ") ⧺ tok(render({"option_id", "description"})) ⧺ tok("\n")    option span
           ⧺ tok("END FIELD\n")
prefix = "<|im_start|>system\n" + SYSTEM_PROMPT + "<|im_end|>\n<|im_start|>user\nSTATE:\n"
suffix = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
```

- **Pieces.** Each piece is tokenized on its own with special tokens parsed; upstream escapes nothing, so neither
  does the port. `render` keeps a string as is and writes anything else as
  `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`; a null description is left out.
- **Options.** noul is `[true, false]`, with upstream's default descriptions replaced key by key by the
  question's criteria; a choice's labels are sorted by code point; score levels keep their order. The runtime
  maps the logits back to the order answers use (the caller's labels, noul `[false, true]`).
- **Length.** The state is cut so the sequence fits 4,096 tokens (`max_tokens`, upstream's `max_length`); a
  schema longer than that is rejected. Upstream's default is 16,384: see the memory note below.
- **Tokenizer.** The repository's `tokenizer.json`, loaded by HF `tokenizers`, gives the same ids as
  transformers' tokenizer (upstream's): it is byte-identical to `jaredpalmer/kev-9b`'s.

`python -m ollaya_convert.families.clef.check <snapshot>` compares the port with upstream on the shared case set
(the edge cases and 300 typed-decisions rows), both at the 4,096-token limit: 365 requests with identical ids,
question spans and option spans, 13 rejected by both, 3 refused first by Ollaya's shared question rules.

## The graph

`convert/ollaya_convert/families/clef/export.py`, 11.4 MB, weightless:

| Input | Shape | |
|---|---|---|
| `input_ids` | int64 [1, seq] | the request's ids, right-padded to a multiple of 64 (up to 16,384) |
| `token_positions` | int64 [n] | 0..n-1, the unpadded positions |
| `question_spans` | int64 [q, 2] | each question's instruction tokens |
| `question_types` | int64 [q] | noul 0, choice 1, score 2 (the head's type embedding) |
| `option_spans` | int64 [o, 2] | each option's tokens, questions in order |
| `option_question` | int64 [o] | the question each option belongs to |
| → `logits` | float32 [o] | one raw logit per option |

- **Backbone.** transformers' Qwen3.5 text model with Clef's weights, recomputed by `llm_common/qwen35.py` (the
  scan-based Gated DeltaNet export, see [decider.md](decider.md)).
- **Head.** The authors' `JointSchemaHead` modules and weights, with its Python loops over questions and options
  written as masked matrix products: span means as a span-mask product, the per-question softmax over options
  as a masked softmax, gathers by `option_question`. On the export sample it matches the authors' forward to
  3.6e-7. The lexical option vectors read rows of the untied LM head, which stays BF16 (a Gather, then a Cast).
- **Weights.** 549 initializers reference the four shards and `joint_head.safetensors` by byte offset (548 BF16
  widened by a Cast, the LM head read as stored); 100 KB of masks stay inline; the 333 vision tensors are unused.
  `weights_in_memory` is `bf16`: about 18.2 GB.

## Differences from upstream

- **Text only.** No images or video (no vision tower in the graph).
- **Length.** 4,096 tokens per request instead of 16,384, so a request fits next to the weights on a 24 GB GPU;
  a longer state is cut where upstream would cut a longer one. Every request in the parity and quality sets is
  shorter (at most 3,032 and 1,195 tokens), so the measurements below are unaffected.
- **Questions.** Ollaya's shared question rules apply first: `instructions` is required (upstream falls back to
  the question id; an empty string still does), score criteria must be a list. As upstream, choice criteria
  must be an object. A question whose instructions render to no token is rejected (upstream returns NaN).
- **Confidence.** Probabilities are upstream's. `confidence` follows Ollaya's convention for `/v1/systemone`
  (the top probability normalized over K options); upstream's own adapter reports the top probability.

## Parity

Measured 2026-10-02. Goldens: the authors' `joint_schema_model.py` in fp32 on the CPU (`ref.load`: their
`ClefModel`, `JointSchemaHead` and `encode_record` on transformers' Qwen3.5 model; their own `load_release_model`
does the same through `AutoProcessor`, which needs torchvision for its video half), over the shared case set (the
edge cases and 40 typed-decisions rows): 131 requests, 13 rejected upstream, 571 questions.

| | sequences | decisions | logits max | probabilities max (p99) | five questions (runner, p50) |
|---|---|---|---|---|---|
| CUDA, RTX 4090 | 118 / 118 identical (ids and every span), 0 rejection mismatches | 100 % | 4.3e-5 | 6.3e-6 (3.6e-6) | 581 ms on the fixtures |

```bash
python -m ollaya_convert.families.clef.goldens out/clef-flash --model SNAPSHOT --td-limit 40
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_clef -- \
    out/clef-flash out/goldens-clef-flash.jsonl cuda --latency
```

Not run: the CPU and Metal.

## fp16 tag

`clef:flash-fp16` runs the same weights with the trunk in fp16 on NVIDIA GPUs ([ADR 0008](../decisions/0008-fp16-decoder-tags.md)).
The RMSNorms, the gated delta rule, the attention softmax, the rotary frequencies, the masks and the heads stay fp32,
as the authors' code computes them. The gate (`parity_clef --fp16`, run by `parity_run`) keeps the encodings exact and
requires at least 99.5 % of the decisions to match the fp32 reference. On the RTX 4090
(`results/runs/2026-10-19-parity-rtx4090-cuda.json`):

| Tag | Decisions the same | Max option logit diff | Max probability diff (p99) |
|---|---|---:|---:|
| `clef:flash-fp16` | 570 / 571 (99.82 %) | 2.9e-02 | 4.5e-03 (2.2e-03) |

The fp32 tag is unchanged and still matches the authors' code. Speed and typed-decisions accuracy of
the fp16 tag are on the model page and on bench.ollaya.dev. On a CPU an fp16 tag runs, slowly; use the fp32 tag there.

## Quality and speed

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label, through the
  Rust runtime on CUDA with `llm_common/quality_runtime.py`): **0.703** (choice 0.703, score 0.619, noul 0.815),
  ECE **0.020** with no temperature at all. Cloudflare's card does not list the training data; it reports
  results on TypeSafe's workflow evals (invoices, customer service, security incidents, agent traces), the kinds
  of workflows typed-decisions covers.
- **RTX 4090, HTTP API** (Ollaya 0.9.0, `clef` pulled from the registry): the triage preset (five questions)
  takes 532 ms at the median of 15 warm requests on one message, and 525 ms over 20 different messages
  (`results/runs/2026-10-02-latency-rtx4090-cuda-extra.json`). In the runner, on the parity fixtures: 581 ms
  (p95 792 ms); the 400 typed-decisions requests took 284 s.
- **Memory.** The weights take about 18 GB kept BF16, so a 24 GB GPU. On the RTX 4090, requests of mixed length up
  to the 4,096-token limit ran back to back at a peak of 22.8 GB (a 4,096-token request in about 3.0 s). At 6,000
  and 8,000 tokens the GPU ran out of memory and the driver spilled into host memory, so a request took minutes;
  that is why the limit is 4,096 rather than upstream's 16,384.
