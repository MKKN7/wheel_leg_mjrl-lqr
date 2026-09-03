from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "wheel_leg_mjrl"))
runpy.run_module("build_rm_train_ground", run_name="__main__")
