import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ollaya_convert import integration  # noqa: E402

HAVE_ONNX = all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime", "numpy", "torch"))


class IntegrationDiff(unittest.TestCase):
    def test_names_changed_decisions_and_statuses(self):
        want = {"a": {"status": 200, "decisions": {"q": "billing", "r": "true"}}, "b": {"status": 422, "decisions": {}}}
        got = {"a": {"status": 200, "decisions": {"q": "sales", "r": "true"}}, "b": {"status": 200, "decisions": {}}}
        self.assertEqual(integration.diff("m", want, got),
                         ["m a q: 'sales', expected 'billing'", "m b: HTTP 200, expected 422"])
        self.assertEqual(integration.diff("m", want, want), [])

    def test_decision_is_the_argmax(self):
        qs = {"t": {"type": "choice", "criteria": {"x": None, "y": None}}, "n": {"type": "noul"},
              "s": {"type": "score", "criteria": ["lo", "mid", "hi"]}}
        ans = {"t": {"probabilities": {"x": 0.2, "y": 0.8}}, "n": {"noul": 0.3},
               "s": {"probabilities": {"0": 0.1, "1": 0.2, "2": 0.7}}}
        self.assertEqual(integration.decisions(qs, ans), {"t": "y", "n": "false", "s": "2"})
        self.assertEqual(integration.decisions(qs, None), {"t": None, "n": None, "s": None})


@unittest.skipUnless(HAVE_ONNX, "needs the convert environment (onnx, onnxruntime, torch)")
class SequenceSplits(unittest.TestCase):
    def test_split_and_chunk_become_split_with_the_same_values(self):
        import tempfile

        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch

        from ollaya_convert.families.llm_common.onnx_export import sequence_splits_to_split

        class M(torch.nn.Module):
            def forward(self, x):
                a, b, c = torch.split(x, [3, 3, 4], dim=-1)
                d, e = torch.chunk(x, 2, dim=-1)
                return a * 2 + b, c + d[..., :4], e

        x = torch.randn(2, 5, 10)
        dyn = {"x": {0: torch.export.Dim("rows"), 1: torch.export.Dim("tokens")}}
        with tempfile.TemporaryDirectory() as tmp:   # as export_graph does (FOLD_LIMIT 0), before the rewrite
            program = torch.onnx.export(M().eval(), (x,), dynamo=True, opset_version=20, input_names=["x"],
                                        dynamic_shapes=dyn, optimize=False)
            program.save(tmp + "/m.onnx")
            model = onnx.load(tmp + "/m.onnx")
            before = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x.numpy()})
            n = sequence_splits_to_split(model)
            ops = {node.op_type for node in model.graph.node}
            after = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x.numpy()})
        self.assertGreaterEqual(n, 1)
        self.assertNotIn("SplitToSequence", ops)
        for b, a in zip(before, after):
            np.testing.assert_array_equal(b, a)


if __name__ == "__main__":
    unittest.main()
