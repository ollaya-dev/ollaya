<!-- One change per pull request. See CONTRIBUTING.md. -->

## What and why



## Checks

- [ ] `cargo fmt --all --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, `cargo test --workspace`
- [ ] `python3 -m unittest discover -s convert/tests` (if `convert/` changed)
- [ ] `cd site && npm ci && npm run build` (if `site/` or a library page changed)
- [ ] No tolerance loosened, golden dropped or golden regenerated
- [ ] No weights committed; nothing at runtime depends on `convert/`

## New model (delete if not)

- [ ] License allows serving and a derived graph; weights pinned to a commit of the author's repository
- [ ] Ported from the author's own inference code (linked here)
- [ ] Parity recorded on CUDA and the CPU: inputs identical, same decisions, max logit difference
- [ ] `docs/families/<family>.md`, `site/content/library/<model>.md` and the tag in `convert/ollaya_convert/catalog.py`

## Not tested

