#!/bin/sh
# Wait until another job of this workflow run uploads an artifact, then unpack it into a directory.
#
#   scripts/wait-artifact.sh <name> <dir> [minutes]
#
# For jobs that need one file of a long job (release.yml: the Windows engine, the linux-amd64
# archives) without waiting for the whole job with `needs`. Gives up after <minutes> (default 60)
# or as soon as a job of the run has failed. Needs gh with GH_TOKEN (actions: read), and
# GITHUB_REPOSITORY and GITHUB_RUN_ID, which Actions sets.
set -eu

[ $# -ge 2 ] || { echo "usage: $0 <name> <dir> [minutes]" >&2; exit 2; }
name=$1 dir=$2 minutes=${3:-60}
run=repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID

waited=0
while [ "$(gh api "$run/artifacts?name=$name" --jq .total_count)" = 0 ]; do
    failed=$(gh api "$run/jobs" --paginate --jq '.jobs[] | select(.conclusion == "failure") | .name')
    [ -z "$failed" ] || { echo "$name: not waiting, a job failed: $failed" >&2; exit 1; }
    [ "$waited" -lt $((minutes * 60)) ] || { echo "$name: not uploaded within $minutes minutes" >&2; exit 1; }
    sleep 20
    waited=$((waited + 20))
done
echo "$name: uploaded after ${waited}s of waiting"
gh run download "$GITHUB_RUN_ID" --repo "$GITHUB_REPOSITORY" --name "$name" --dir "$dir"
