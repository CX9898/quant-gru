"""Compatibility entry point for the canonical GRU regression script."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests" / "test_quant_gru.py"),
        run_name="__main__",
    )
