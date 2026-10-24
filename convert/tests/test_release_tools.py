"""The release tools of docs/model-lifecycle.md on hand-made data: what bench_plan asks a change to measure, how
bench_edition adds a release to an edition, and how parity_run records a parity run; no network, weights or GPU."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ollaya_convert import bench_edition, bench_plan, bench_run, parity_run

GRAPH = "application/vnd.ollaya.graph.onnx"
WEIGHTS = "application/vnd.ollaya.weights"


def manifest(graph="g1", weights="w1", precision="fp32", config="c1"):
    return {"layers": [
        {"mediaType": GRAPH, "digest": "sha256:" + graph, "annotations": {"org.ollaya.precision": precision}},
        {"mediaType": WEIGHTS, "digest": "sha256:" + weights},
        {"mediaType": "application/vnd.ollaya.calibration", "digest": "sha256:" + config},
    ]}


class Plan(unittest.TestCase):
    def test_new_tag_gets_every_suite(self):
        self.assertEqual(bench_plan.decide("kev:4b", manifest(), None, {}, False)[0], bench_plan.FULL)

    def test_new_fp16_tag_gets_typed_and_gpu_latency(self):
        self.assertEqual(bench_plan.decide("kev:4b-fp16", manifest(precision="fp16"), None, {}, False)[0],
                         ["typed", "latency-cuda"])

    def test_same_layers_measure_nothing(self):
        self.assertEqual(bench_plan.decide("kev:4b", manifest(), manifest(), {}, False)[0], [])

    def test_new_weights_are_a_new_model(self):
        self.assertEqual(bench_plan.decide("kev:4b", manifest(weights="w2"), manifest(), {}, False)[0], bench_plan.FULL)

    def test_graph_change_needs_parity(self):
        suites, why = bench_plan.decide("kev:4b", manifest(graph="g2"), manifest(), {}, False)
        self.assertIsNone(suites)
        self.assertIn("no parity", why)
        failed = {"kev:4b": {"pass": False, "gate": "fp32"}}
        self.assertIsNone(bench_plan.decide("kev:4b", manifest(graph="g2"), manifest(), failed, False)[0])

    def test_graph_change_with_fp32_parity_carries_accuracy(self):
        parity = {"kev:4b": {"pass": True, "gate": "fp32"}}
        self.assertEqual(bench_plan.decide("kev:4b", manifest(graph="g2"), manifest(), parity, False)[0], ["latency-cuda"])
        self.assertEqual(bench_plan.decide("kev:4b", manifest(graph="g2"), manifest(), parity, True)[0],
                         ["latency-cuda", "latency-cpu"])

    def test_graph_change_with_fp16_gate_measures_accuracy(self):
        parity = {"kev:4b-fp16": {"pass": True, "gate": "fp16"}}
        old, new = manifest(precision="fp16"), manifest(graph="g2", precision="fp16")
        self.assertEqual(bench_plan.decide("kev:4b-fp16", new, old, parity, False)[0], ["typed", "latency-cuda"])

    def test_scope(self):
        for tag in ("decider:2b-vision", "laya:en-fp16", "laya:multilingual-fp32"):
            self.assertTrue(bench_plan.OUT_OF_SCOPE.search(tag), tag)
        for tag in ("kev:4b-fp16", "laya:en", "decider:2b"):
            self.assertFalse(bench_plan.OUT_OF_SCOPE.search(tag), tag)


class EditionUpdate(unittest.TestCase):
    def test_merge(self):
        edition = {"runs": ["a.json"], "models": ["kev:4b", "d1:3b"],
                   "score": {"results": [{"model": "kev:4b", "score": 50.0}, {"model": "d1:3b", "score": 60.0}]}}
        out = bench_edition.merge_update(edition, "0.14.0", "2026-10-20", ["b.json"],
                                         ["kev:4b-fp16", "strands-decider:2b"],
                                         [{"model": "strands-decider:2b", "score": 55.0}])
        self.assertEqual(out["runs"], ["a.json", "b.json"])
        self.assertEqual(out["models"], ["kev:4b", "d1:3b", "strands-decider:2b"])  # -fp16 stays off the leaderboard
        self.assertEqual([r["model"] for r in out["score"]["results"]], ["d1:3b", "strands-decider:2b", "kev:4b"])
        self.assertEqual(out["updates"][0]["release"], "0.14.0")


class Run(unittest.TestCase):
    def test_suites_keep_the_order(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "plan.json")
            json.dump({"models": {"kev:4b": {"suites": ["latency-cpu", "typed", "latency-cuda"]}}}, open(p, "w"))
            self.assertEqual(bench_run.suites_of(p, [], []), {"kev:4b": ["latency-cuda", "typed", "latency-cpu"]})
        self.assertEqual(bench_run.suites_of(None, ["d1:3b"], ["public"]), {"d1:3b": ["public"]})


class ParityRecord(unittest.TestCase):
    def test_command_picks_example_goldens_and_gate(self):
        cmd, gate = parity_run.command("kev", "/x/out/kev-0.8b-r15-fp16", "cuda")
        self.assertEqual(gate, "fp16")
        self.assertIn("parity_kev", cmd)
        self.assertTrue(cmd[cmd.index("--") + 2].endswith("goldens-kev-0.8b-r15.jsonl"))
        self.assertEqual(cmd[-1], "--fp16")
        cmd, gate = parity_run.command("decider", "/x/out/decider-2b-vision", "cpu")
        self.assertEqual((gate, "parity_decider_vision" in cmd, "--features" in cmd), ("fp32", True, False))
        cmd, _ = parity_run.command("decima", "/x/out/decima-base-clean", "cuda", "/pack/cuda_v13")
        self.assertEqual(cmd[cmd.index("--features") + 1], "cuda-dynamic")
        self.assertIn("--target-dir", cmd)

    def test_parse_and_merge(self):
        out = 'numbers: ...\nrecord: {"questions": 10, "decisions_same": 10, "max_logit_diff": 1e-5}\n'
        row = parity_run.parse("kev:4b", out, 0)
        self.assertEqual((row["model"], row["questions"], row["pass"]), ("kev:4b", 10, True))
        self.assertFalse(parity_run.parse("kev:4b", out, 1)["pass"])
        self.assertIsNone(parity_run.parse("kev:4b", "encoding mismatch\n", 1))
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "run.json")
            parity_run.merge(p, [dict(row, device="cuda")], "rtx-4090", "cuda", "2026-10-19", "0.13.0")
            run = parity_run.merge(p, [dict(row, device="cuda", questions=11)], "rtx-4090", "cuda", "2026-10-19", "0.13.0")
            self.assertEqual([r["questions"] for r in run["results"]], [11])  # checked again: replaced
            self.assertEqual((run["suite"], run["machine"]), ("parity", "rtx-4090"))


if __name__ == "__main__":
    unittest.main()
