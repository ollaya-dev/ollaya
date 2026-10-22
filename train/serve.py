"""Serve a checkpoint over `/v1/systemone`, so convert/'s benchmark runners can score it during training.

    uv run serve.py runs/fp150 --port 11600
    python -m ollaya_convert.bench_typed run --url http://127.0.0.1:11600 --model fp150 --data ... --out ...

A development aid only: the answers are `laya.Agent.system_one`'s, the network Ollaya's laya runtime
matches; released numbers are measured through Ollaya itself.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import laya


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--port", type=int, default=11600)
    ap.add_argument("--device")
    a = ap.parse_args()
    agent = laya.load(a.checkpoint, device=a.device)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/v1/systemone":
                return self.reply(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:
                    res = agent.system_one(req["state"], req["questions"])
            except (ValueError, KeyError, TypeError) as e:
                return self.reply(400, {"error": str(e)})
            res["model"] = req.get("model", res.get("model"))
            self.reply(200, res)

    print("serving %s on 127.0.0.1:%d" % (a.checkpoint, a.port), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
