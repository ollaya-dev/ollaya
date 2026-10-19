"""Measure models for Ollaya Bench on this machine: the raw outputs `bench_edition` turns into result files.

    python -m ollaya_convert.bench_run --machine rtx-5090 --out OUT --plan plan.json \\
        --typed-data typed-decisions-test.jsonl --public-data PUBLIC_DIR [--ollaya PATH] [--registry URL]
    python -m ollaya_convert.bench_run ... --models kev:4b --suites latency-cuda typed

The suites per model come from `bench_plan` (--plan) or from --models and --suites; anyone can run either on their
own machine (docs/model-lifecycle.md, "Run it yourself"). It starts `ollaya serve` (one model loaded at a time),
pulls the models, then runs, in this order so a partial run already covers most:

    latency-cuda   bench_latency.py --memory, 5 warm-up and 20 timed requests per model  -> OUT/latency-<machine>-cuda.json
    typed          bench_typed.py, the 400 typed-decisions states                         -> OUT/typed/<model>/rows.jsonl
    public         bench_public.py, Bespoke Labs' 13 public subsets                       -> OUT/public/<model>/<subset>/
    latency-cpu    the server restarted with OLLAYA_DEVICE=cpu; 2 warm-up and 10 timed      -> OUT/latency-<machine>-cpu.json

Run it again after an interruption: it skips what OUT already holds. --registry points `ollaya pull` at a test
registry (a release candidate's `registry/`, served with `python -m http.server`). Standard library only.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ORDER = ["latency-cuda", "typed", "public", "latency-cpu"]
CONVERT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def log(msg):
    print("[%s] %s" % (time.strftime("%F %T"), msg), flush=True)


def model_dir(model):
    return model.replace(":", "-")


def suites_of(plan_path, models, suites):
    """model -> its suites, in ORDER."""
    if plan_path:
        wanted = {m: e["suites"] for m, e in json.load(open(plan_path, encoding="utf-8"))["models"].items()}
    else:
        wanted = {m: suites for m in models}
    bad = {s for v in wanted.values() for s in v} - set(ORDER)
    if bad:
        sys.exit("unknown suites: %s" % " ".join(sorted(bad)))
    return {m: [s for s in ORDER if s in v] for m, v in wanted.items()}


def done(out, model, suite):
    """Whether OUT already holds this suite of this model (the latency files resume on their own)."""
    d = model_dir(model)
    if suite == "typed":
        p = os.path.join(out, "typed", d, "rows.jsonl")
        return os.path.exists(p) and sum(1 for _ in open(p, encoding="utf-8")) >= 400
    if suite == "public":
        p = os.path.join(out, "public", d)
        return os.path.isdir(p) and sum(os.path.exists(os.path.join(p, s, "summary.json")) for s in os.listdir(p)) >= 13
    return False


class Server:
    def __init__(self, ollaya, env, out):
        self.ollaya, self.env, self.out, self.proc = ollaya, env, out, None

    def start(self, host, device):
        self.stop()
        self.url = "http://" + host
        try:
            urllib.request.urlopen(self.url + "/api/version", timeout=2)
            sys.exit("something already answers on %s; stop it or pick another --port" % host)
        except OSError:
            pass
        env = dict(self.env, OLLAYA_HOST=host, OLLAYA_DEVICE=device)
        self.proc = subprocess.Popen([self.ollaya, "serve"], env=env, stdin=subprocess.DEVNULL,
                                     stdout=open(os.path.join(self.out, "serve-%s.log" % device), "w"),
                                     stderr=subprocess.STDOUT)
        for _ in range(60):
            try:
                urllib.request.urlopen(self.url + "/api/version", timeout=2)
                return
            except OSError:
                time.sleep(1)
        sys.exit("the server on %s did not start (see serve-%s.log)" % (host, device))

    def unload(self, model):
        req = urllib.request.Request(self.url + "/api/decide", data=json.dumps({"model": model, "keep_alive": 0}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=120).read()
        except OSError:
            pass

    def stop(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=60)
            self.proc = None


def tool(name, *args):
    """Run one of the bench tools in this package with the same Python."""
    env = dict(os.environ, PYTHONPATH=CONVERT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    return subprocess.run([sys.executable, "-m", "ollaya_convert." + name, *args], env=env).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", required=True, help="the machine's id in results/machines.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", help="a plan from bench_plan")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--suites", nargs="*", default=["latency-cuda", "typed", "public"], choices=ORDER)
    ap.add_argument("--ollaya", default=shutil.which("ollaya"), help="the ollaya binary (default: the one on PATH)")
    ap.add_argument("--registry", help="pull from this registry URL instead of ollaya.dev")
    ap.add_argument("--blobs", help="a candidate registry's blobs/ directory: its manifests point at ollaya.dev, where the new "
                    "blobs are not published yet, so they are linked into the model store first")
    ap.add_argument("--typed-data", help="the typed-decisions test split (jsonl)")
    ap.add_argument("--public-data", help="the directory holding <subset>/all.jsonl")
    ap.add_argument("--port", type=int, default=11460, help="the GPU server's port; the CPU server uses the next")
    ap.add_argument("--note", default="", help="added to the latency files' note")
    ap.add_argument("--cpu-only", action="store_true", help="no NVIDIA GPU here: every suite runs on the CPU server")
    a = ap.parse_args()
    if not (a.plan or a.models):
        sys.exit("give --plan or --models")
    if not a.ollaya:
        sys.exit("no ollaya on PATH; pass --ollaya")
    work = suites_of(a.plan, a.models, a.suites)
    needs = {s for v in work.values() for s in v}
    if a.cpu_only and "latency-cuda" in needs:
        sys.exit("--cpu-only cannot time latency-cuda")
    if "typed" in needs and not a.typed_data or "public" in needs and not a.public_data:
        sys.exit("the typed suite needs --typed-data and the public suite --public-data")

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.join(out, "run.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        sys.exit("another bench_run holds %s/run.lock" % out)
    env = dict(os.environ, OLLAYA_MODELS=os.environ.get("OLLAYA_MODELS", os.path.join(out, "models")),
               OLLAYA_KEEP_ALIVE="30m", OLLAYA_MAX_LOADED_MODELS="1")
    if a.registry:
        env["OLLAYA_REGISTRY"] = a.registry
    machine = a.machine.replace("-", "")
    server = Server(a.ollaya, env, out)
    gpu, cpu = "127.0.0.1:%d" % a.port, "127.0.0.1:%d" % (a.port + 1)

    def latency(device, models):
        if models:
            log("latency %s: %s" % (device, " ".join(models)))
            extra = ["--warm", "2", "--n", "10"] if device == "cpu" else []
            tool("bench_latency", "--url", server.url, "--machine", a.machine, "--device", device, "--memory",
                 "--resume", *extra, "--models", *models,
                 "--out", os.path.join(out, "latency-%s-%s.json" % (machine, device)), "--note", note)

    try:
        # Pulls and the accuracy suites go through the first server: the GPU's, unless --cpu-only.
        server.start(cpu if a.cpu_only else gpu, "cpu" if a.cpu_only else "cuda")
        version = json.load(urllib.request.urlopen(server.url + "/api/version", timeout=10))["version"]
        note = ("Ollaya %s. %s" % (version, a.note)).strip()
        log("START Ollaya %s, %d models, plan %s" % (version, len(work), a.plan or "from --models"))
        if a.blobs:
            store = os.path.join(env["OLLAYA_MODELS"], "blobs")
            os.makedirs(store, exist_ok=True)
            for f in os.listdir(a.blobs):
                if f.startswith("sha256-") and not os.path.exists(os.path.join(store, f)):
                    os.link(os.path.join(a.blobs, f), os.path.join(store, f))
        for m in work:
            p = subprocess.run([a.ollaya, "pull", m], env=dict(env, OLLAYA_HOST=server.url[len("http://"):]),
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            if p.returncode:
                sys.exit("pull %s failed: %s" % (m, p.stderr.strip()[-500:]))
            log("pulled %s" % m)
        if not a.cpu_only:
            latency("cuda", [m for m, v in work.items() if "latency-cuda" in v])
        for m, v in work.items():
            if "typed" in v and not done(out, m, "typed"):
                log("typed %s" % m)
                tool("bench_typed", "run", "--url", server.url, "--model", m, "--data", a.typed_data,
                     "--out", os.path.join(out, "typed", model_dir(m)))
            if "public" in v and not done(out, m, "public"):
                log("public %s" % m)
                shutil.rmtree(os.path.join(out, "public", model_dir(m)), ignore_errors=True)
                tool("bench_public", "run", "--url", server.url, "--model", m, "--data", a.public_data,
                     "--out", os.path.join(out, "public", model_dir(m)))
            server.unload(m)
        cpu_models = [m for m, v in work.items() if "latency-cpu" in v]
        if cpu_models:
            if not a.cpu_only:
                server.start(cpu, "cpu")
            latency("cpu", cpu_models)
    finally:
        server.stop()
    log("DONE")


if __name__ == "__main__":
    main()
