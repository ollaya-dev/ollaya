"""Shared build-time checkpoint paths, without importing model runtimes."""
import os

DEFAULT_ROOT = os.environ.get("LAYA_ROOT", os.path.expanduser("~/models/laya"))
LARGHA_ROOT = os.environ.get("LARGHA_ROOT", os.path.expanduser("~/models/largha"))
