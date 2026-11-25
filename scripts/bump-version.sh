#!/bin/sh
# Set the release version everywhere it is written (docs/model-lifecycle.md, stage 6):
#   scripts/bump-version.sh 0.14.0
# The workspace and desktop crates, their lock files (Ollaya's own crates only), the desktop app's packaging
# configuration (desktop/Packager.toml), and the MCP Registry entry (server.json). Fails if any file did not change, so
# a new place the version lives is noticed.
set -eu
new=${1:?usage: scripts/bump-version.sh X.Y.Z}
cd "$(dirname "$0")/.."
old=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -1)
[ "$old" != "$new" ] || { echo "already $new"; exit 1; }
for f in Cargo.toml desktop/Cargo.toml desktop/Packager.toml; do
  perl -pi -e "s/^version = \"\\Q$old\\E\"/version = \"$new\"/" "$f"
done
for f in Cargo.lock desktop/Cargo.lock; do
  perl -0pi -e "s/(name = \"ollaya[^\"]*\"\\nversion = )\"\\Q$old\\E\"/\$1\"$new\"/g" "$f"
done
# The MCP Registry entry can already name the coming release (0.17.0 was its first image), so it is set and checked by
# content, not by diff.
perl -pi -e "s/^  \"version\": \"[^\"]*\"/  \"version\": \"$new\"/; s{(\"ghcr\\.io/ollaya-dev/ollaya:)[^\"]*\"}{\${1}$new\"}" server.json
grep -q "^  \"version\": \"$new\"" server.json && grep -q "\"ghcr\\.io/ollaya-dev/ollaya:$new\"" server.json \
  || { echo "server.json: version or image tag not set to $new"; exit 1; }
for f in Cargo.toml Cargo.lock desktop/Cargo.toml desktop/Cargo.lock desktop/Packager.toml; do
  git diff --quiet -- "$f" && { echo "$f: no $old found"; exit 1; }
done
git diff --stat
echo "$old -> $new. Commit as \"Ollaya $new\", then tag v$new."
