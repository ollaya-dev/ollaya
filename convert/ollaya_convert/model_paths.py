"""Shared build-time checkpoint paths, without importing model runtimes."""
import os

DEFAULT_ROOT = os.environ.get("LAYA_ROOT", os.path.expanduser("~/models/laya"))
LARGHA_ROOT = os.environ.get("LARGHA_ROOT", os.path.expanduser("~/models/largha"))
# Ollaya's own laya checkpoints, one folder per tag (a copy of ollaya-dev/coba-<tag> at its pinned commit).
COBA_ROOT = os.environ.get("COBA_ROOT", os.path.expanduser("~/models/coba"))
COBA_TAGS = ("1b", "400m", "400m-ternary", "150m-ternary", "68m-ternary")
