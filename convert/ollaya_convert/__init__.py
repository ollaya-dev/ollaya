"""Build-time tooling that turns decision-model checkpoints into Ollaya model layers.

Nothing here runs on a user's machine: the Ollaya runtime is a Rust binary. This package exports
checkpoints to ONNX, checks the exports against the PyTorch reference, and produces the golden
fixtures the Rust port is tested against.
"""
