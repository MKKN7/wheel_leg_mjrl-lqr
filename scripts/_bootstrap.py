"""Run a packaged project module from a repository checkout."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "wheel_leg_mjrl"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def run(module: str) -> None:
    runpy.run_module(module, run_name="__main__")
