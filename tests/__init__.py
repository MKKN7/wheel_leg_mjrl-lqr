"""Regression tests for the locomotion sandbox."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parents[1] / "src" / "wheel_leg_mjrl"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
