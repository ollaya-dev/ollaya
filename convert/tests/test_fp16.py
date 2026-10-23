import unittest

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from ollaya_convert.fp16 import to_fp16


class Fp16Test(unittest.TestCase):
    def test_weights_cast_plainly_and_constants_keep_the_clamp(self):
        w = np.full((64, 128), 0.5, np.float32)
        w[0, 0] = 1e-8  # below fp16's smallest subnormal: a cast gives 0, the converter 5.96e-8
        eps = np.array([1e-12], np.float32)
        graph = helper.make_graph(
            [helper.make_node("MatMul", ["x", "w"], ["y0"]), helper.make_node("Add", ["y0", "eps"], ["y"])], "g",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 128])],
            [numpy_helper.from_array(w, "w"), numpy_helper.from_array(eps, "eps")])
        model = to_fp16(helper.make_model(graph))
        got = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
        self.assertTrue(np.array_equal(got["w"], w.astype(np.float16)))
        self.assertEqual(got["w"][0, 0], 0)
        self.assertGreater(got["eps"][0], 0)


if __name__ == "__main__":
    unittest.main()
