"""Render a deterministic side view for lower guide-wheel placement review."""

from __future__ import annotations

from pathlib import Path

import mujoco
from PIL import Image


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "wheeled_infantry.xml"
OUTPUT_PATH = ROOT / "artifacts" / "guide_wheel_lower_preview.png"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.0, 0.22)
    camera.distance = 0.90
    camera.azimuth = 90.0
    camera.elevation = -10.0
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(OUTPUT_PATH)
    finally:
        renderer.close()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
