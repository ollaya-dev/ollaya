# Contributing to Ollaya

Thanks for helping. Most contributions are one of three kinds: a fix or feature in the runtime, a docs or site change,
or a new model. New models are the most welcome: several families in the library were ported by people outside the
project (Arbiter, Credence, d1 and Winnow's vision tags, among others).

Before a large change, open an issue so we can agree on the approach. For a new model, use the
[new model](https://github.com/ollaya-dev/ollaya/issues/new?template=new-model.yml) template.

## Setup

```sh
cargo test --workspace                 # the Rust workspace: CLI, daemon, runners
cd convert && uv sync                  # build-time Python: export, parity, goldens (not needed at runtime)
cd site && npm ci && npm run build     # the website
```

## Rules that every change keeps

- **Parity tests must pass.** Never loosen a tolerance, drop a golden case or regenerate goldens to make a change
  pass. If a change fails parity, fix the change.
- **Python is build-time only.** Nothing at runtime depends on `convert/`.
- **Never re-host model weights.** They come unmodified from the author's Hugging Face repository, pinned to a commit
  and checked by sha256. Don't commit, upload or bundle `*.safetensors`, `*.gguf` or `*.onnx.data` files.
- **No typed numbers on the site.** Pages refer to measurements with `{{result <tag> <path>}}` and the build fills
  them in from `results/`. A figure someone else published may be quoted when the sentence says whose it is.

## Porting a model

The full process, with the commands and the time each stage takes, is
[docs/model-lifecycle.md](docs/model-lifecycle.md). In short:

1. **Intake.** The model has a known license that allows serving it and distributing a graph derived from it, its
   weights are on Hugging Face at a commit we can pin, and the author's own inference code is available (a model card
   alone isn't a reference). The fp32 reference has to fit a 24 GB GPU, so a decoder of up to about 9B parameters.
2. **Port.** Either an ONNX export under `convert/ollaya_convert/families/<family>/`, or the author's GGUF file on
   llama.cpp ([ADR 0003](docs/decisions/0003-llama-cpp-runtime.md)). The layout (how a request becomes the model's
   input rows) is ported from the author's code, in Python for the reference and in Rust in `crates/ollaya-decision`.
   Look at a family with a similar design in [docs/families/](docs/families/) first: many new models reuse an
   existing layout.
3. **Gate.** Goldens from the author's code, then the Rust runtime against them on CUDA and on the CPU: identical
   inputs, the same decision on every case, and logits within 1e-3 or tighter. Record the runs. If you can only
   run one of the two devices, say so in the pull request.
4. **Pages.** `docs/families/<family>.md` (files, layout, parity, limits, attribution), the library page
   `site/content/library/<model>.md`, and the tag in `convert/ollaya_convert/catalog.py`.

The maintainer measures every model on the bench machines, then adds the registry entry and ships it in a release.
Your pull request doesn't need GPUs beyond the one you ran parity on, and it doesn't carry `registry/` or bench
results.

### A good first model

A good first model reuses what Ollaya already runs: a new checkpoint of a supported family, or a GGUF model that
reads its answer from option letters, as `snap`, `jevk5` and `winnow` do. A new architecture with its own head is a
bigger first step. Ask in an issue if you're not sure which kind a model is.

## Pull requests

Run the checks CI runs, in this order (CI stops at the first failure, and formatting comes first):

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace
python3 -m unittest discover -s convert/tests
cd site && npm ci && npm run build
```

Keep a pull request to one change, describe what you ran, and say what you couldn't test. Contributions are under
the project's Apache-2.0 license; each model keeps its own.

## Credit

Every model author is credited on their model's page and on the [contributors page](https://ollaya.dev/contributors),
together with the people who brought each model to Ollaya.
