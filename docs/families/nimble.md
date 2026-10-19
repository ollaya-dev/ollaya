# nimble (`nimble-codes-v1`)

Bespoke Labs' **Bespoke-Nimble-9B-v2** (Apache-2.0) is a LoRA adapter (r=16, α=32) on
**Qwen/Qwen3.5-9B** (Apache-2.0), trained on contrastive pairs to answer typed questions from the
next-token logits of one-letter option codes. It never generates text. The repository ships the
reference scorer (`serving_schema.py`, `extended_schema.py`, `parallel_schema.py`, `inference.py`) and
the recommended temperature; the TypeSafe mapping is the author's `/v1/systemone` server
([bespokelabsai/nimble](https://github.com/bespokelabsai/nimble) `nimble/serving/compiler.py`, with
openjev-sglang's request models for the defaults).

| Model | Checkpoint | Base | Temperature | Status in Ollaya |
|---|---|---|---|---|
| `bespokelabs/Bespoke-Nimble-9B-v2` | `4b8c04d1` (2026-09-23) | Qwen/Qwen3.5-9B @ `c2022362` (4 shards) | 2.179 | **converted, ONNX** (weights stay BF16 in memory) |
| `bespokelabs/Bespoke-Nimble-9B-v3` | `8e927b9b` (2026-10) | same base | 1 (none shipped) | **converted, ONNX** (`nimble:9b-v3`; the adapter is CC BY-NC 4.0, see [v3](#v3-nimble9b-v3)) |

## How it decides

One chat row per question (upstream `prepare_prompts`). Every row carries the whole request; only the
requested field at the end differs:

```text
row    = pre + safe_json({"context": serialize(state), "schema": fields})
             + "\n\nRequested field: " + safe_json(qid) + post
fields = [{"name": qid, "description": serialize(instructions),
           "choices": [{"code": "A", "value": ..., "description": ...}, ...]}, ...]
```

- **TypeSafe mapping** (the author's compiler): `noul` becomes a boolean field with values
  `[false, true]` and the descriptions of `criteria.false` and `criteria.true` (`"No"` and `"Yes"`
  when absent); `choice` becomes an enum of the labels, described by their descriptions or, when null,
  the label; `score` becomes an enum `"0".."n-1"` described by the levels.
- **Serialization.** `serialize` is the string itself, or `json.dumps(ensure_ascii=False)` for any
  other JSON. `safe_json` writes `<` and `>` as `<` and `>`, so user text cannot form a chat
  control token; the row is tokenized whole with special tokens parsed, as upstream does.
- **Template.** `pre` and `post` are Qwen3.5's chat template rendered around the user turn with the
  training system prompt and `enable_thinking=False` (stored in `decision.json`). When any question has
  more than 26 options, every row uses the serving extension: the system prompt says "short code"
  instead of "one-letter code", and codes continue after `Z` with 229 two-letter codes (`AA`, `AB`, ...),
  each one token at the answer boundary (`serving_config.json`).
- **Readout.** The graph returns the next-token logits of all 255 code tokens at each row's last token;
  a question reads its first k. Probabilities are `softmax(logits / 2.179)`, the author's recommended
  temperature (`temperature_config.json`: fitted on the original Nimble-9B and transferred to v2).

## The graph

One graph (13.6 MB), weightless: `input_ids [rows, seq]` (right-padded to a multiple of 64) and
`last_pos [rows]` → `cand_logits [rows, 255]`. It reuses the Qwen3.5 export
(`llm_common/qwen35.py`); the LoRA is not merged (every adapted Linear runs `x·Wᵀ + 2·(x·Aᵀ)·Bᵀ`), so
the base shards and `adapter_model.safetensors` stay byte for byte the authors'. The untied `lm_head` is
read only at the 255 code rows. The vision tower and MTP tensors (348) are unused. `weights_in_memory`
is `bf16`: about 18 GB.

## Differences from upstream

- **Precision.** Upstream serves in BF16 autocast; Ollaya computes in fp32 on the BF16 weights, as for
  every Qwen3.5 model here. The goldens are upstream's code in fp32.
- **Rejections.** As upstream: blank state, blank instructions or question ids, blank choice labels,
  noul criteria with keys other than `"true"` and `"false"`, and a row over 8,192 tokens (422, never
  truncated). `instructions` is required, as for every Ollaya model. Ollaya also accepts `choice`
  criteria given as a list of labels, and non-string descriptions (serialized as JSON), which openjev's
  strict models refuse.

## Parity

Goldens: `ollaya_convert.families.nimble.goldens` runs the author's code (serving_schema.prepare_prompts and
inference.candidate_logits, the LoRA unmerged) in fp32 with TF32 off, the part of the 9B that does not fit a
24 GB GPU streamed from CPU memory, over the shared case set (20 typed-decisions rows, Laya's and the decoder
edge cases): 104 requests, 4 rejected upstream, 492 questions. The layout port is also checked against
upstream on 277 requests without the model (`layout.py`): identical token rows, the same rejections.

| | token rows | decisions | option logits max | probabilities max (p99) | wire answers |
|---|---|---|---|---|---|
| CUDA, RTX 4090 | 492 / 492 identical, 0 rejection mismatches | 100 % | 1.1e-4 | 6.5e-6 (5.6e-6) | max diff 5.5e-5 |

The graph uses a 4,096-token budget per `session.run`: the weights take 18 GB of a 24 GB GPU, and on WSL
the driver spills a larger working set into system memory, which slowed the first parity run tenfold.

```sh
uv run --with peft==0.21.0 python -m ollaya_convert.families.nimble.export --out out/nimble-9b-v2
uv run --with peft==0.21.0 python -m ollaya_convert.families.nimble.goldens out/nimble-9b-v2 --device offload
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_nimble -- convert/out/nimble-9b-v2 convert/out/goldens-nimble-9b-v2.jsonl cuda --latency
```

## fp16 tag

`nimble:9b-fp16` runs the same weights with the trunk in fp16 on NVIDIA GPUs ([ADR 0008](../decisions/0008-fp16-decoder-tags.md)).
The RMSNorms, the gated delta rule, the attention softmax, the rotary frequencies, the masks and the heads stay fp32,
as the authors' code computes them. The gate (`parity_nimble --fp16`, run by `parity_run`) keeps the encodings exact and
requires at least 99.5 % of the decisions to match the fp32 reference. On the RTX 4090
(`results/runs/2026-10-19-parity-rtx4090-cuda.json`):

| Tag | Decisions the same | Max option logit diff | Max probability diff (p99) |
|---|---|---:|---:|
| `nimble:9b-fp16` | 492 / 492 (100.00 %) | 4.5e-02 | 4.9e-03 (2.5e-03) |

The fp32 tag is unchanged and still matches the authors' code. Speed and typed-decisions accuracy of
the fp16 tag are on the model page and on bench.ollaya.dev. On a CPU an fp16 tag runs, slowly; use the fp32 tag there.

## Quality and speed

- **Typed-decisions** (all 400 test states, 2,000 questions, argmax against the majority label, through the
  Rust runtime on CUDA): **0.665** (choice 0.572, score 0.626, noul 0.810), ECE 0.084 at the shipped
  temperature 2.179. Nimble was not trained on typed-decisions.
- **Bespoke Labs' public benchmark** (13 human-labeled subsets, 3,880 questions), Ollaya against Ollama on the
  same GPU: see the site's comparison and `convert/ollaya_convert/bench_public.py`.

## v3 (`nimble:9b-v3`)

Bespoke-Nimble-9B-v3 is a LoRA (r=64) on the same Qwen3.5-9B, licensed **CC BY-NC 4.0** (no commercial use). The
model repository holds the adapter and its contract (`schema_config.json`: the 255 codes, their token ids and the
sha256 of the two prompt files). The code ships with the author's Decision Index run, the dataset
`bespokelabs/bespoke-nimble-9b-v3-decision-index` @ `654b124d` (`code/nimble/evaluation`); its prompt files are
byte-identical to v2's, so the graph and the layout are v2's. `decision.json` declares `mapping: "decision-index"`,
the engine's TypeSafe mapping (`decision_index_engine.NimbleEngine`), which differs from v2's in four places:

- a choice option or noul side without a description has none (v2 wrote the label, or "No" / "Yes");
- noul criteria keyed exactly `{"yes", "no"}` read as `{"true", "false"}`;
- a blank string state is sent as `json.dumps(state)` instead of being rejected;
- rows run to 32,768 tokens (v2: 8,192), never truncated.

Probabilities are the softmax of the code logits at temperature 1, as the engine reads them (no temperature file).
The tokenizer is v2's `tokenizer.json`: the engine loads `AutoTokenizer` for the base, which transformers 5
rebuilds from the vocabulary byte for byte as v2's file, not as the base repository's (which splits combining
marks differently). Reference: `families/nimble/ref_v3.py` imports the engine from the pinned dataset snapshot.

Goldens (`goldens.py --checkpoint v3`, fp32, TF32 off, offloaded): 105 requests (the shared case set and four v3
cases: a blank state, yes/no keys, options without descriptions, a 10,000-token state), 3 rejected upstream.

| | token rows | decisions | option logits max | probabilities max (p99) | wire answers |
|---|---|---|---|---|---|
| CUDA, RTX 4090 | 498 / 498 identical, 0 rejection mismatches | 100 % | 1.4e-4 | 1.1e-5 (6.7e-6) | max diff 5.6e-5 |

Latency on the RTX 4090, five-question requests: p50 2.4 s, p95 6.5 s (n = 68), as for v2. A row of about 10,000
tokens (the long-state case) overflows the GPU: the BF16 weights take 18 GB and full attention grows with the square
of the row, so the WSL driver spills into system memory, and the session's CUDA arena keeps that size. In the parity
run that included it, later five-question requests took 20.9 s at the median. On a 24 GB GPU, v3 rows above
8,192 tokens run, but slowly, and slow the model until it is unloaded.

```sh
uv run --with peft==0.21.0 python -m ollaya_convert.families.nimble.export --checkpoint v3 --out out/nimble-9b-v3
uv run --with peft==0.21.0 --with accelerate python -m ollaya_convert.families.nimble.goldens out/nimble-9b-v3 --checkpoint v3 --device offload
cargo run --release -p ollaya-runner --features ollaya-runner/cuda --example parity_nimble -- convert/out/nimble-9b-v3 convert/out/goldens-nimble-9b-v3.jsonl cuda --latency
```
