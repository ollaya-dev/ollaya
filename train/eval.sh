#!/bin/bash
# Score a checkpoint on both held-out suites: typed-decisions test and Bespoke Labs' public benchmark.
#   train/eval.sh <checkpoint> <bench-dir> [port]
# <bench-dir> holds bench_typed.py, bench_public.py, typed-decisions-test.jsonl, public/ and nimble-src/.
# Writes <checkpoint>/eval/{typed,public}/ and prints both reports.
set -euo pipefail
CKPT=$(realpath "$1"); B=$(realpath "$2"); PORT=${3:-11600}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$HERE/.venv/bin/python}
NAME=$(basename "$CKPT")
mkdir -p "$CKPT/eval"
"$PY" "$HERE/serve.py" "$CKPT" --port "$PORT" > "$CKPT/eval/serve.log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
until grep -q serving "$CKPT/eval/serve.log"; do
  kill -0 $SP 2>/dev/null || { cat "$CKPT/eval/serve.log"; exit 1; }
  sleep 1
done
URL=http://127.0.0.1:$PORT
cd "$B"
"$PY" bench_typed.py run --url $URL --model "$NAME" --data typed-decisions-test.jsonl --out "$CKPT/eval/typed" > /dev/null
NIMBLE_SRC=$B/nimble-src "$PY" bench_public.py run --url $URL --model "$NAME" --data public --out "$CKPT/eval/public" > /dev/null
"$PY" bench_typed.py report --data typed-decisions-test.jsonl --json "$CKPT/eval/typed.json" "$CKPT/eval/typed"
NIMBLE_SRC=$B/nimble-src "$PY" bench_public.py report "$CKPT/eval/public" --json "$CKPT/eval/public.json"
