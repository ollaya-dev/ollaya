# 0010: Adapters as a model kind: N LoRA fine-tunes over one loaded base

- Status: proposed, 2026-11-19 (issue #172). Design only; the code follows once this is agreed.
- Applies to: `crates/ollaya-registry/src/manifest.rs`, `crates/ollaya/src/modelfile.rs`,
  `ModelConfig` and `Ollaya::create` in `crates/ollaya-server`, `scheduler.rs`, `models.rs`, and the MLX engine in
  `crates/ollaya-runner/src/mlx`.
- Not covered: batching in the runner (its own record, measured on its own) and fp16 on MLX (`PARAMETER precision`,
  its own record against [0008](0008-fp16-decoder-tags.md)'s bar).

## Context

A LoRA fine-tune of a base the MLX engine runs can be imported today only as a merged model: a full checkpoint of
its own, which the runner loads as its own process. Four fine-tunes of `laya:en` cost four runners of about 2.6 GB
each (#172), although they differ in a few small matrices. The pieces for something better exist:

- `Ollaya::create` copies the base's manifest and replaces layers; blobs are shared by digest.
- The MLX engine reads a peft adapter as one more weights file and, with `lora` in the arch layer, merges it into
  the backbone at load (`W + scaling · B A`, float32; [0001](0001-mlx-engine.md)). The kev ONNX graphs keep the
  adapter unmerged (`docs/families/kev.md`).
- A runner is one process per manifest digest, and the scheduler's only limit is a count
  (`OLLAYA_MAX_LOADED_MODELS`). `/api/ps` `size` is the manifest's size.

What is missing is a model that is *an adapter on a named base* and a runner that holds one base for several of them.

## Decision

All names and values below are proposals unless the text says the code already does it.

### 1. Two new layers, one new `model_format`

- `application/vnd.ollaya.adapter` (`media::ADAPTER`): a small JSON file, like the arch layer. It holds no weights:

  ```json
  {"schema": 1, "format": "peft-lora",
   "base": {"manifest": "sha256:...", "arch": "sha256:...", "weights": ["sha256:..."]},
   "rank": 16, "alpha": 32, "scaling": 2.0, "dtype": "F32",
   "prefix": "base_model.model.",
   "targets": ["model.layers.0.self_attn.q_proj", "..."]}
  ```

  `targets` lists each adapted module of the base by the name of its `.weight` tensor in the arch layer, without the
  suffix. `scaling` is `alpha / r`, as `arch.py` computes it today (`use_rslora`, `use_dora`, `rank_pattern` and
  `alpha_pattern` are refused, as there).
- `application/vnd.ollaya.adapter.weights` (`media::ADAPTER_WEIGHTS`): the adapter's safetensors file, byte for byte
  as peft wrote it (`lora_A.weight` `[r, in]`, `lora_B.weight` `[out, r]`).
- Digests are as everywhere: sha256 of the blob's bytes. The spec is written with sorted keys and no insignificant
  whitespace so that the same adapter on the same base gives the same blob. The model's digest is the sha256 of its
  manifest, as now.
- The manifest keeps every layer of the base (as any `create` does), so it is self-contained and removing the base's
  tag does not break it. `base` in the spec is what the adapter was checked against. **The sharing key is the
  base's arch digest plus its weights digests, not its manifest digest**: a base with other questions or calibration
  has another manifest digest and the same tensors, and should share the runner. (Alternative: the manifest digest
  only, simpler, but then a recalibrated base would load a second copy.)
- `ModelConfig.model_format` is `adapter`; `engine`, `family` and `layout` are the base's. A client that does not
  know the kind stops in `Scheduler::run_check` and `models::resolve_entry` ("is an adapter model, which this version
  cannot run; upgrade ollaya"), rather than serving the plain base under the adapter's name.
- Weights stay out of the registry: an adapter is a local file made by the user. There is no push, and `ollaya pull`
  never meets an adapter model. The "never re-host weights" rule is not touched.

### 2. `ollaya create`

```text
FROM laya:en
ADAPTER ./ft-a            # a PEFT directory: adapter_config.json + adapter_model.safetensors
QUESTIONS ./q.json        # the other directives work as before
```

`ADAPTER` takes one directory and may appear once. `PARAMETER precision` is refused with an adapter for now.
Because `/api/create` takes only inline JSON ("no `files` map and no `/api/blobs` upload", `docs/api.md` 7.9) and an
adapter is tens to hundreds of MB, this needs the file to reach the daemon. Proposal: implement the reserved
`HEAD`/`POST /api/blobs/:digest` (today `501`): the CLI hashes the safetensors, uploads it if the daemon lacks it, and
`/api/create` gets `adapter: {"config": {...}, "weights": "sha256:..."}`. (Alternative: a path the daemon reads
itself; no new endpoint, but it only works when client and daemon share a disk.)

Validation happens in `create`, before anything is written, and refuses with `422` and a message:

1. `FROM` has an arch layer, so MLX can run it (an ONNX-only base: "has no arch layer"), and is not a router or an
   adapter model itself (no stacking).
2. The adapter config's own checks (`r`, `lora_alpha`, no rslora/dora/patterns, `peft_type` is `LORA`).
3. Every `lora_A`/`lora_B` pair names a module that exists in the arch layer, as a linear weight `[out, in]`, with
   `A = [r, in]` and `B = [out, r]` for the same `r` as the config. A tensor that is not a pair (peft's
   `modules_to_save`, a trained head) is refused in this first version.
4. The dtype is F32, F16 or BF16.

A mismatch names the module and both shapes. There is no check that the adapter was *trained* on this base; shapes
cannot show that, and `adapter_config.json`'s `base_model_name_or_path` is a free string. The record's guarantee is
shape compatibility plus the parity test below.

### 3. Scheduler

Today a runner is keyed by manifest digest and the limit is a count. Proposal:

- **A runner is keyed by the base's sharing key** (section 1). It loads the base once and serves the base itself and
  any number of adapter models. Models with equal keys that have no adapter share a runner as they do now.
- **A runner takes adapters at run time**: `POST /adapters` loads an adapter's tensors (paths of the blobs, like the
  other runner arguments) and `DELETE /adapters/<digest>` drops it; `/decide` gains an optional `adapter` (the adapter
  model's digest; absent means the base). An unknown adapter is `404`.
- **What is counted.** `OLLAYA_MAX_LOADED_MODELS` keeps counting runners, that is bases, because that is where the
  memory is. Adapters are bounded per runner by a new `OLLAYA_MAX_ADAPTERS` (proposal: default 16).
  Memory is *reported*, not enforced, as today: each runner reports the bytes its tensors occupy (the engine knows
  them; the base is fp32 after the upcast of [0001](0001-mlx-engine.md), an adapter is its elements times 4).
  Nothing here adds a byte budget; if you want one, it is a later change that this layout does not block.
- **`/api/ps`.** One row per loaded model name. The base's row carries the base's `size`, an adapter's row only its
  own, so the rows sum to the real total. A base reached only through adapters is listed under the name in
  `details.parent_model`. (Today a runner appears once per digest.)
- **Eviction.** Adapters go before bases. When a runner's adapter limit is reached, its least recently used idle adapter
  is dropped (no process dies). When the runner limit is reached, the least recently used idle *runner* goes, adapters
  with it, as now. Loading an adapter never evicts its own base; a base never goes while it has a lease or an adapter
  loaded.
- **Keep-alive.** A request's `keep_alive` applies to the model that answered, as documented. An adapter unloads on its
  own clock. The base's clock is the latest of its own and its adapters', so the base stays loaded while any adapter
  is; `-1` on one adapter keeps the base forever. `OLLAYA_PRELOAD` takes adapter names.
- **Failure.** A runner that dies takes its adapters with it; the scheduler forgets all of them and the next request
  reloads the base and the adapter it needs. A load that fails because of one adapter leaves the base and the other
  adapters loaded, and answers `MODEL_LOAD_FAILED` for that adapter only.

### 4. Runner: unmerged, per request, unbatched

The adapter is applied per request, unmerged: each adapted linear computes `x·Wᵀ + scaling · (x·Aᵀ)·Bᵀ` in float32.
Merging on load, as `with_lora` does for one model, would write a full copy of every adapted matrix per adapter and
bring back the memory cost this record exists to remove. (Alternative: merge into the shared weights and undo after
each request; it is not safe with a second request in flight and drifts in float32.) The unmerged form is what the kev
graphs run, and `docs/families/kev.md` records it within rounding of the merged result.

Requests are served by the runner's one MLX worker, in order, as now, whatever their adapter. **This record does not
promise throughput.** In #172 one runner answered 4.1 to 4.3 requests a second and four runners 10.7 to 11.8, so
at three to four in flight an adapter model on one runner may be slower than four merged runners; what it gives is
memory (about 2.6 GB per merged checkpoint against the adapter's own size). Batching rows of different adapters is
the next record. The first slice covers what MLX already merges today (the Qwen3 and Qwen3.5 decoders); the encoder
engines (`laya`, `nli`, `gliclass`) have no LoRA path yet and are the second slice, which is what #172 needs.

### 5. Compatibility

- Existing models, merged or not, resolve and run exactly as before; their manifests are untouched, so their digests
  are too. Nothing changes for the kev ONNX graphs, whose adapter is part of the graph.
- Without `ADAPTER`, `create` writes what it writes today. `model_format: "adapter"` appears only on models that use it.
- An older client reading an adapter model's manifest fails with the "upgrade ollaya" error (section 1).
- On a device other than `metal`, or in a build without the `mlx` feature, an adapter model fails to load with a
  message saying it runs on MLX only (`auto` does not fall back to the CPU for it), as `metal` does for any model MLX
  cannot run.

## Parity and tests

Parity for an adapter model is the existing fp32 gate, with the merged model as the reference: for the same
requests, **every decision the same and option logits within 1e-3** of the base with the adapter merged by peft's
`merge_and_unload` in float32, which is what a user would otherwise have imported. Tolerances are not loosened. In
addition, and bit for bit: a request to adapter A returns the same answer when adapters B and C are loaded and in use
between requests, and a request to the base is unchanged by loaded adapters. Planned:

- The parity example takes `--adapter <dir>` and runs the golden set against the merged reference; goldens come from
  `convert/`, not from the Rust code. Like the other MLX parity this runs by hand on a Mac (CI has no Metal,
  [0001](0001-mlx-engine.md), Risks); CI runs the rest.
- Unit tests: the spec's canonical form and digest; the create checks of section 2, one test per refusal, with a
  small synthetic adapter; `resolve_entry` for an adapter manifest; the scheduler's choice of victim and the base's
  clock as pure functions over a list of runners, so eviction order needs no processes.
- Memory: with the base and four adapters loaded, `/api/ps` rows sum to the runner's reported bytes, and that is
  within the adapters' own size of one base alone. Measured on a Mac and recorded under `results/` as
  [0009](0009-measurement-per-release.md) asks, not typed into pages.

## Questions for the maintainers

1. **Upload or path** for the adapter file at create (section 2)? We lean to implementing `/api/blobs`.
2. **`modules_to_save`.** A fine-tune that also trains a head (peft's `modules_to_save`) cannot be an adapter in this
   version. Should a later one let an adapter replace named head tensors, which would also change what the sharing key
   must cover?
3. **A byte budget** instead of, or beside, the runner count and `OLLAYA_MAX_ADAPTERS`?
4. **Throughput first or memory first.** If one worker per base makes the adapter kind slower than N runners at
   a few requests in flight, is memory alone enough to land it before batching, or should the two ship together?
5. **The second slice** (encoders) is where our use is; is it right to land the decoder path first?
