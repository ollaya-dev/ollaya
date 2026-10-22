"""Teacher labels from an Ollaya daemon's /v1/systemone (build-time only, stdlib).

Resumable: appends {"id", "teacher": {qid: [probabilities in option order, noul as [false, true]]}}
or {"id", "error": ...} for a rejected row to --out, and skips every id already there.

  python3 train/data/label.py --in data/main.jsonl --out data/teacher.jsonl \
      --url http://127.0.0.1:11500 --model pplx-decider:flash-fp16
"""

import argparse
import concurrent.futures as cf
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.request

MAX_QUESTIONS = 64  # the teacher's per-request limit


class Rejected(Exception):
    pass


def post(url, body, retries=8):
    data = json.dumps(body, ensure_ascii=False).encode()
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code < 500:  # 4xx: the request itself is wrong, retrying cannot help
                try:
                    err = json.load(e)
                except ValueError:
                    err = {"code": str(e.code)}
                raise Rejected(f"{err.get('code')}: {err.get('error', '')[:300]}")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(min(2**attempt, 30))
    raise RuntimeError("server unavailable")


def probs(q, a):
    if q["type"] == "noul":
        return [round(1 - a["noul"], 4), a["noul"]]
    p = a["probabilities"]
    keys = list(q["criteria"]) if q["type"] == "choice" else [str(i) for i in range(len(q["criteria"]))]
    return [p[k] for k in keys]


def label(row, url, model):
    qs = list(row["questions"].items())
    out = {}
    for b in range(0, len(qs), MAX_QUESTIONS):
        part = dict(qs[b : b + MAX_QUESTIONS])
        res = post(url, {"model": model, "state": row["state"], "questions": part})
        for qid, q in part.items():
            out[qid] = probs(q, res["answers"][qid])
    return out


def selftest():
    assert probs({"type": "noul"}, {"noul": 0.75}) == [0.25, 0.75]
    assert probs({"type": "choice", "criteria": {"b": None, "a": None}}, {"probabilities": {"a": 0.1, "b": 0.9}}) == [0.9, 0.1]
    assert probs({"type": "score", "criteria": ["x", "y"]}, {"probabilities": {"0": 0.3, "1": 0.7}}) == [0.3, 0.7]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:11500", help="one or more daemons, comma-separated")
    ap.add_argument("--model", default="pplx-decider:flash-fp16")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="label at most this many new rows")
    args = ap.parse_args()
    selftest()
    # A runner does one forward pass at a time, so throughput scales with daemons, not workers.
    urls = itertools.cycle(u.rstrip("/") + "/v1/systemone" for u in args.url.split(","))

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass  # a line cut off by a crash: that row is labelled again

    def todo():
        n = 0
        with open(args.inp) as f:
            for line in f:
                row = json.loads(line)
                if row["id"] in done:
                    continue
                if args.limit and n >= args.limit:
                    return
                n += 1
                yield row

    with open(args.inp) as f:
        total_rows = sum(1 for _ in f)
    remaining = total_rows - len(done)
    t0, rows_n, q_n, rejected = time.time(), 0, 0, 0
    with open(args.out, "a") as out, cf.ThreadPoolExecutor(args.workers) as ex:
        pending = set()
        it = todo()
        while True:
            while len(pending) < args.workers * 4:
                row = next(it, None)
                if row is None:
                    break
                fut = ex.submit(label, row, next(urls), args.model)
                fut.row = row
                pending.add(fut)
            if not pending:
                break
            finished, pending = cf.wait(pending, return_when=cf.FIRST_COMPLETED)
            for fut in finished:
                row = fut.row
                try:
                    rec = {"id": row["id"], "teacher": fut.result()}
                    q_n += len(row["questions"])
                except Rejected as e:
                    rec = {"id": row["id"], "error": str(e)}
                    rejected += 1
                    print(f"rejected {row['id']}: {e}", file=sys.stderr, flush=True)
                except Exception as e:  # server down or a bad answer: not recorded, the next run retries it
                    print(f"failed {row['id']}: {e!r}", file=sys.stderr, flush=True)
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows_n += 1
                if rows_n % 500 == 0:
                    out.flush()
                    dt = time.time() - t0
                    left = (remaining - rows_n) / (rows_n / dt)
                    print(
                        f"{rows_n} rows {q_n} q in {dt:.0f}s: {rows_n / dt:.1f} rows/s {q_n / dt:.1f} q/s, "
                        f"{rejected} rejected, ETA {left / 3600:.1f} h",
                        flush=True,
                    )
    dt = time.time() - t0
    print(f"done: {rows_n} rows {q_n} questions in {dt:.0f}s ({rows_n / max(dt, 1e-9):.1f} rows/s, {q_n / max(dt, 1e-9):.1f} q/s), {rejected} rejected")


if __name__ == "__main__":
    main()
