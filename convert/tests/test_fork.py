"""Exact state forking (`llm_common/qwen35.py`): the split mirrors the runner's (`crates/ollaya-runner/src/fork.rs`),
and on a small random Qwen3.5 every forked row has the row form's hidden states."""
import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAVE_TORCH = all(importlib.util.find_spec(m) for m in ("torch", "transformers"))
PAD = 9


def row(shared, own, tag):
    return list(range(shared)) + [tag] * own


@unittest.skipUnless(HAVE_TORCH, "needs the convert environment (torch, transformers)")
class ForkTest(unittest.TestCase):
    def test_split_matches_the_runner(self):
        from ollaya_convert.families.llm_common.qwen35 import fork_rows

        # the cases of fork.rs's tests, with the same shapes
        p, b, j = fork_rows([row(150, 30, 1), row(150, 90, 2), row(150, 5, 3)], PAD)
        self.assertEqual((len(p), len(p[0]), len(b), len(b[0]), j), (1, 128, 3, 128, [0, 0, 0]))
        p, b, j = fork_rows([row(10, 120, 1), row(10, 200, 2), row(5, 20, 3)], PAD)
        self.assertEqual((len(p), len(p[0]), len(b[0]), j), (3, 192, 64, [0, 1, 2]))
        p, b, j = fork_rows([row(5, 20, 1), row(5, 50, 2)], PAD)
        self.assertEqual((len(p[0]), len(b[0])), (64, 64))
        self.assertTrue(all(t == PAD for x in b for t in x))
        p, b, j = fork_rows([row(128, 0, 0), row(128, 0, 0)], PAD)
        self.assertEqual((len(p), len(p[0]), len(b[0])), (1, 64, 64))

    def test_fork_is_the_row_form(self):
        import torch
        from transformers import Qwen3_5TextConfig, Qwen3_5TextModel

        from ollaya_convert.families.llm_common.qwen35 import CHUNK, Qwen35Trunk, fork_rows, gather_hidden

        torch.manual_seed(0)
        cfg = Qwen3_5TextConfig(vocab_size=500, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                                num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                                layer_types=["linear_attention"] * 3 + ["full_attention"], linear_num_key_heads=2,
                                linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
        trunk = Qwen35Trunk(Qwen3_5TextModel(cfg).float().eval())
        g = torch.Generator().manual_seed(1)
        state = torch.randint(1, 500, (150,), generator=g).tolist()
        cases = {
            "fork": [state + torch.randint(1, 500, (n,), generator=g).tolist() for n in (30, 90, 5)],
            "own prefixes": [torch.randint(1, 500, (n,), generator=g).tolist() for n in (130, 210, 25)],
            "one chunk": [torch.randint(1, 500, (n,), generator=g).tolist() for n in (25, 50)],
        }
        for name, rows in cases.items():
            T = -(-max(map(len, rows)) // CHUNK) * CHUNK
            ids = torch.tensor([r + [PAD] * (T - len(r)) for r in rows])
            pos = torch.tensor([[0, len(r) // 2, len(r) - 1] for r in rows])
            with torch.no_grad():
                want = trunk(ids)[torch.arange(len(rows))[:, None], pos]
                prefixes, branches, job = map(torch.tensor, fork_rows(rows, PAD))
                hp, hb = trunk.fork(prefixes, branches, job)
                got = gather_hidden(hp, hb, job, pos)
            self.assertLess(float((got - want).abs().max()), 1e-5, name)


if __name__ == "__main__":
    unittest.main()
