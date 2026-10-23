# Model lifecycle: from a new model to the library, the bench and the site

This is the one procedure for every change that reaches users:

- a new model;
- a new tag of a model (another size, a precision);
- a new graph or engine for an existing tag;
- a runtime change;
- a release.

Each stage below has the same parts:

- **Done when:** what has to be true before the next stage starts.
- **Commands:** what to run.
- **Machine:** where it runs.
- **Budget:** how long it should take. A stage that runs well over its budget is a reason to stop and ask why.

The rules on what to measure are in [ADR 0009](decisions/0009-measurement-per-release.md). The parity rules are in
[PROJECT_NOTES](../.claude/skills/PROJECT_NOTES.md). The release mechanics are in [distribution.md](distribution.md#release-process).

```text
0 Intake → 1 Port → 2 Gate → 3 Catalog → 4 Measure → 5 PR → 6 Release → 7 Live
                                             ↑ bench_plan decides what   ↑ bench_edition --update, deploy
```

## Machines

Results name machines by hardware (`results/machines.json`), never by host name.

| Role | Machine | Why this one |
|---|---|---|
| Export, goldens, parity | RTX 4090 desktop, Linux (`rtx-4090`) | 24 GB fits the fp32 reference of a 9B decoder; every parity number in `docs/families/` was measured on it |
| Ollaya Bench | RTX 5090 desktop with a Threadripper 3960X, Linux (`rtx-5090`) | The machine of edition 2026.10. Rows of one edition compare only when they come from its machine. It can power off unexpectedly, so every bench tool resumes where it stopped |
| Release builds | GitHub Actions (`release.yml`) | About 20 minutes: `release.yml` (four platforms, the GPU packs, the images) and `desktop.yml` run in parallel; 0.13.0 took 16 and 19 |
| Registry and site deploy | The machine with `registry/blobs` and a `wrangler` login | The blobs are not in git; `node site/scripts/fetch-blobs.mjs` restores them from the live site |

The hosts behind `rtx-4090` and `rtx-5090`, what else not to run on them, and `scripts/remote-job.sh`, which sends them
a job in tmux and waits for it: [Remote jobs](remote-jobs.md).

A contributor can run stages 2 and 4 on their own hardware. Their results go in as a `community-<gpu>` machine
(see [Run it yourself](#run-it-yourself)).

## 0 Intake

**Done when** every answer below is yes, and the model has a line in `docs/backlog.md` (or a stub in
`docs/families/<family>.md`) that records the answers.

- **License.** It is known, and it allows serving the model and distributing a graph derived from it. With no license,
  stop: Tev1 waits for one. A non-commercial license is allowed, but `latest` must not point at it
  (`nimble:9b-v3`).
- **Weights.** They are on Hugging Face at a commit we can pin, and Ollaya fetches them unmodified. We never re-host
  weights (`*.safetensors`, `*.onnx.data`) in git, a release, an image or the site.
- **Reference.** The authors' own inference code is available, and it is what the goldens come from. A model card
  alone is not a reference.
- **Fit.** The fp32 reference fits the 24 GB parity GPU. Today that means up to about 9B parameters for a decoder;
  Clef (27B) does not fit. A model that doesn't fit waits for a bigger parity machine. Its gate is never loosened
  to make it fit.
- **Engine.** Either an ONNX port under `convert/ollaya_convert/families/`, or llama.cpp on the authors' GGUF
  ([ADR 0003](decisions/0003-llama-cpp-runtime.md)).

**Budget:** under an hour of reading.

## 1 Port

**Done when** the family's files exist and the layout port reproduces every upstream row of the goldens' requests.

| File | What |
|---|---|
| `convert/ollaya_convert/families/<f>/ref.py` | The authors' code, wrapped: the PyTorch fp32 reference |
| `…/<f>/layout.py` | The request → rows layout, ported from the authors' code |
| `…/<f>/export.py` | The ONNX export (fp32; `--fp16` for a GPU tag, [ADR 0008](decisions/0008-fp16-decoder-tags.md)) |
| `…/<f>/goldens.py` | The goldens: the reference's outputs on the shared edge cases plus the family's own |
| `crates/ollaya-decision/src/<f>.rs` | Questions → rows in Rust |
| `crates/ollaya-runner/src/<f>.rs`, `engine.rs` | The engine and its registration under the manifest's `engine` name |
| `crates/ollaya-runner/examples/parity_<f>.rs` | The runtime gate; it prints one `record:` JSON line before its checks |
| `docs/families/<f>.md` | Files, layout, ONNX contract, parity, limits, attribution |

**Budget:** days. It is the only stage without a fixed budget.

## 2 Gate

**Done when:**

- the export matches the PyTorch reference;
- the Rust runtime matches the goldens on CUDA, and on the CPU where the model is meant to run there;
- each run is recorded in `results/runs/<date>-parity-<machine>-<device>.json`.

```sh
cd convert
python -m ollaya_convert.families.<f>.export <tag> out/<dir>        # checks itself against the reference
python -m ollaya_convert.families.<f>.goldens <tag> out/<dir>
python -m ollaya_convert.parity_run --machine rtx-4090 --device cuda <model>:<tag> …
python -m ollaya_convert.parity_run --machine rtx-4090 --device cpu  <model>:<tag> …
```

`parity_run` maps each tag to its export directory, goldens and example through `catalog.py`. It records the numbers
and the pass, and names the export it checked. A CUDA run needs the CUDA libraries on `LD_LIBRARY_PATH`, the same as
running the examples directly.

Rules:

- **fp32 gate.** The encodings are identical, every decision is the same, the option logits are within the family's
  tolerance (1e-3 or tighter), and the wire answers equal the authors'.
- **Never loosen the gate.** If a change fails, fix the change. Don't widen a tolerance, drop a golden case or
  regenerate goldens.
- **Reduced precision is a new tag.** An `-fp16` tag gets its own gate (ADR 0008): identical encodings and at
  least 99.5 % of decisions the same. The fp32 tag stays as it is.
- **A graph change re-runs the gate** for every tag of that family. The same numbers as before mean the outputs
  did not change, and stage 4 carries the accuracy forward.

**Budget:**

- Export: 10–40 minutes per tag.
- Goldens: 10–60 minutes.
- CUDA parity: 1–5 minutes per tag.
- CPU parity: up to 2 hours for a 9B model. Run it for new models, not for every graph change.

## 3 Catalog

**Done when:**

- the tag is in `convert/ollaya_convert/catalog.py`;
- its overlay is in `site/src/data/catalog.ts` (title, summary, and stats as a tag to look up);
- its page `site/content/library/<model>.md` exists;
- `cd site && npm run build` passes.

**Numbers are never typed into these files.** A page refers to a measurement and the build fills it in from
`results/`. For example, `{{result kev:4b typed.accuracy 3}}` becomes kev:4b's typed-decisions accuracy, to three
decimals, from the current edition. The build fails on a reference to a result that doesn't exist. A number that
isn't in `results/` doesn't go on the site, with one exception: a figure someone else published (an author's model
card, TypeSafe's number for Jev) may be quoted when the sentence says whose it is.

In code (`catalog.ts`, the home page) the same lookup is `result('kev:4b', 'typed.accuracy', 3)` from
`site/src/data/numbers.ts`, which lists the paths. The README can't run code, so its numbers sit between
`<!-- results:<name> -->` markers, written by `site/src/data/readme.ts`. After a results update, run
`npm run readme` in `site/`; `npm run build` fails while a block is stale.

**Budget:** under an hour.

## 4 Measure (before the release)

**Done when:**

- the raw outputs of everything `bench_plan` lists are on the bench machine;
- `bench_edition --update` has turned them into result files.

The release candidate is measured from a test registry, so nothing is published before it passes.

1. On the parity machine, package the changed models into the repository's `registry/`:
   `python -m ollaya_convert.package <model> …`.
2. Ask what to measure (on any machine, from the repository): `python -m ollaya_convert.bench_plan --out plan.json`.
   It compares every tag's manifest with the digest the current edition measured, and lists the suites per tag
   with an estimate. It fails when a changed graph has no passing parity record. When that happens, go back to stage 2.
3. On the bench machine, serve the candidate registry and run the plan:

   ```sh
   python -m http.server 8899 -d registry &
   python -m ollaya_convert.bench_run --plan plan.json --machine rtx-5090 --out ~/bench/0.14.0 \
       --ollaya <branch build>/bin/ollaya --registry http://127.0.0.1:8899 --blobs registry/blobs \
       --typed-data typed-decisions-test.jsonl --public-data public/
   ```

   The candidate's manifests point their blobs at ollaya.dev, where the new graphs are not published yet;
   `--blobs` links them into the model store first, so the manifests (and the digests the bench records) are the
   ones that will ship.

   It resumes where it stopped. The plan, not habit, decides the suites (ADR 0009):

   - parity-identical graph changes get latency only;
   - new tags and new precisions get typed-decisions;
   - only new models or new weights get the 3,880-question public suite.

4. Write the result files:
   `python -m ollaya_convert.bench_edition --update <release> --edition <current> --raw ~/bench/0.14.0 …`
   (the same arguments as a full edition, with `--models` from the plan).

**Budget:** `bench_plan` prints the estimate. The tables below give typical times.

Rough time per model:

| Suite | Time per model |
|---|---|
| Latency | 1–3 minutes on the GPU, up to 15 on the CPU |
| typed-decisions | 2–10 minutes |
| Public | 30–90 minutes |

Rough total per release:

| Release | Bench time |
|---|---|
| Graph change only | Under an hour |
| A new model | About 2 hours |
| A full edition | About 7.5 hours |

A full edition runs on a schedule (once a quarter) or after a runtime-wide change, never for one release.

## 5 PR

**Done when** CI passes and the PR is squash-merged. Run the checks locally first:

```sh
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
python3 -m unittest discover -s convert/tests
cd site && npm run build
```

The PR carries the code, the docs, the parity records and the result files from stage 4. It does not carry the
registry: manifests go in at release time.

**Budget:** CI takes 5–10 minutes.

## 6 Release

**Done when:**

- the release is published;
- the installed release passes the integration corpus;
- new models have their corpus file.

Releases are batched: several models and fixes go into one release, so the release workflows run once.

1. On the parity machine, package the changed models and commit their manifests as "Registry: …" commits (one per
   model). The blobs stay on that machine's disk until the deploy.
2. `scripts/bump-version.sh X.Y.Z`, commit "Ollaya X.Y.Z", tag `vX.Y.Z` and push the tag. The release workflow
   builds and publishes everything ([distribution.md](distribution.md#release-process)).
3. Install the release on a GPU machine and run
   `python -m ollaya_convert.integration check`. A changed decision fails the release.
4. For each new model, run `python -m ollaya_convert.integration record --model <tag>` against the installed
   release, and commit the file.

**Budget:** about 40 minutes, half of it the workflows.

## 7 Live

**Done when:**

- `ollaya pull <tag>` works from ollaya.dev;
- bench.ollaya.dev shows the new rows, stamped with the release that measured them.

Deploy the site after the release, so a manifest never points at an engine that no published binary has.

```sh
cd site
npm run deploy                    # ollaya.dev: pages and the registry under /v2/
node scripts/backup-blobs.mjs <bucket>
npm run deploy:bench              # bench.ollaya.dev
```

Then close the model's items in `docs/backlog.md`.

**Budget:** 15 minutes.

## Run it yourself

Anyone can measure the library on their own machine with the same tools. You need:

- an Ollaya install;
- this repository (the tools need only the Python standard library);
- the two datasets:
  - `typed-decisions-test.jsonl`, from `python -m ollaya_convert.bench_typed export`, run once in the `convert/`
    environment;
  - Bespoke Labs' public benchmark, built with their converters and checked against their committed checksums
    (`bench_public.py`).

```sh
ollaya serve &
python -m ollaya_convert.bench_run --machine community-rtx-4070 --out runs/ \
    --models kev:4b d1:3b --suites latency-cuda typed \
    --typed-data typed-decisions-test.jsonl
```

To submit your results:

1. Open a PR that adds your machine to `results/machines.json` (`kind: community`, `by`, `source`).
2. Add the run files that `bench_edition` writes from your outputs.

Community runs appear on ollaya.dev/results, labeled with your machine. They are not ranked in an edition: an
edition's rows come from its own machine.

## Checklists

**New model:**

- [ ] Intake answers recorded.
- [ ] Port done: the layout reproduces upstream.
- [ ] Parity recorded on CUDA and CPU.
- [ ] Catalog, overlay and library page added, with no typed numbers.
- [ ] `bench_plan` lists it with the full set of suites, and `bench_run` has measured them.
- [ ] `bench_edition --update` has run.
- [ ] PR merged.
- [ ] Release published; integration corpus checked, and a corpus file recorded for the model.
- [ ] Site and bench deployed.

**Graph or engine change:**

- [ ] Parity re-run for every tag of the family, with the same numbers as before (otherwise the change is a new
      model or a new precision).
- [ ] `bench_plan` shows latency only.
- [ ] Measured, updated, released and deployed as above.

**Runtime-wide change (an ORT bump, the scheduler):**

- [ ] Parity for every family on CUDA and CPU.
- [ ] A latency sweep, or a new edition when decisions moved.
