#!/bin/sh
# Set the release version everywhere it is written (docs/model-lifecycle.md, stage 6):
#   scripts/bump-version.sh 0.14.0
# The workspace and desktop crates, their lock files (Ollaya's own crates only), the desktop app's package files and
# its UI mock. Fails if any file did not change, so a new place the version lives is noticed.
set -eu
new=${1:?usage: scripts/bump-version.sh X.Y.Z}
cd "$(dirname "$0")/.."
old=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)
[ "$old" != "$new" ] || { echo "already $new"; exit 1; }
for f in Cargo.toml desktop/src-tauri/Cargo.toml; do
  perl -pi -e "s/^version = \"\\Q$old\\E\"/version = \"$new\"/" "$f"
done
for f in Cargo.lock desktop/src-tauri/Cargo.lock; do
  perl -0pi -e "s/(name = \"ollaya[^\"]*\"\\nversion = )\"\\Q$old\\E\"/\$1\"$new\"/g" "$f"
done
perl -pi -e "s/^  \"version\": \"\\Q$old\\E\"/  \"version\": \"$new\"/" desktop/package.json desktop/src-tauri/tauri.conf.json
perl -0pi -e "s/^(  \"version\": )\"\\Q$old\\E\"/\$1\"$new\"/m; s/(\"\": \\{\\n      \"name\": \"[^\"]*\",\\n      \"version\": )\"\\Q$old\\E\"/\$1\"$new\"/" desktop/package-lock.json
perl -pi -e "s/version: '\\Q$old\\E'/version: '$new'/g" desktop/ui/src/mock.ts
for f in Cargo.toml Cargo.lock desktop/package.json desktop/package-lock.json desktop/src-tauri/Cargo.toml \
         desktop/src-tauri/Cargo.lock desktop/src-tauri/tauri.conf.json desktop/ui/src/mock.ts; do
  git diff --quiet -- "$f" && { echo "$f: no $old found"; exit 1; }
done
git diff --stat
echo "$old -> $new. Commit as \"Ollaya $new\", then tag v$new."
