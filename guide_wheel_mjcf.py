"""Render and synchronize the passive rolling guide-wheel MJCF block."""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "guide_wheel_training_model.yaml"
DEFAULT_XML = ROOT / "wheeled_infantry.xml"
BEGIN_MARKER = "      <!-- GUIDE_WHEEL_TRAINING_BEGIN -->"
END_MARKER = "      <!-- GUIDE_WHEEL_TRAINING_END -->"
ROOT_BODY_CLOSE = re.compile(r"(?m)^    </body>\r?\n  </worldbody>")
IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class GuideWheel:
    name: str
    side: str
    position_m: tuple[float, float, float]


@dataclass(frozen=True)
class GuideWheelModel:
    root_body: str
    radius_m: float
    half_width_m: float
    axle_axis: tuple[float, float, float]
    cylinder_quat: tuple[float, float, float, float]
    density_kg_m3: float
    joint_damping_nms: float
    joint_armature_kg_m2: float
    friction: tuple[float, float, float]
    condim: int
    contype: int
    conaffinity: int
    rgba: tuple[float, float, float, float]
    expected_actuator_count: int
    expected_sensor_data_count: int
    wheels: tuple[GuideWheel, ...]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{path} must be an MJCF-safe identifier")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{path} must be finite and >= {minimum}")
    return result


def _vector(value: Any, path: str, size: int, *, minimum: float | None = None) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{path} must contain exactly {size} values")
    return tuple(
        _number(item, f"{path}[{index}]", minimum=-math.inf if minimum is None else minimum)
        for index, item in enumerate(value)
    )


def load_guide_wheel_model(path: str | Path = DEFAULT_CONFIG) -> GuideWheelModel:
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = _mapping(raw, "root")
    if root.get("schema_version") != 1:
        raise ValueError("guide wheel config schema_version must be 1")
    if root.get("mode") != "rolling_passive_guide_wheels":
        raise ValueError("guide wheel config must select rolling_passive_guide_wheels")
    geometry = _mapping(root.get("geometry"), "geometry")
    contracts = _mapping(root.get("contracts"), "contracts")
    wheels_raw = root.get("wheels")
    if not isinstance(wheels_raw, list) or len(wheels_raw) != int(contracts.get("expected_wheel_count", 0)):
        raise ValueError("wheels must match contracts.expected_wheel_count")
    wheels: list[GuideWheel] = []
    for index, value in enumerate(wheels_raw):
        wheel = _mapping(value, f"wheels[{index}]")
        name = _identifier(wheel.get("name"), f"wheels[{index}].name")
        side = wheel.get("side")
        if side not in {"left", "right"}:
            raise ValueError(f"wheels[{index}].side must be left or right")
        wheels.append(GuideWheel(name, side, _vector(wheel.get("position_m"), f"wheels[{index}].position_m", 3)))
    if len({wheel.name for wheel in wheels}) != len(wheels):
        raise ValueError("guide wheel names must be unique")
    if {wheel.side for wheel in wheels} != {"left", "right"}:
        raise ValueError("guide wheel layout must contain both left and right support")
    return GuideWheelModel(
        root_body=_identifier(root.get("root_body"), "root_body"),
        radius_m=_number(geometry.get("radius_m"), "geometry.radius_m", minimum=1.0e-6),
        half_width_m=_number(geometry.get("half_width_m"), "geometry.half_width_m", minimum=1.0e-6),
        axle_axis=_vector(geometry.get("axle_axis"), "geometry.axle_axis", 3),
        cylinder_quat=_vector(geometry.get("cylinder_quat"), "geometry.cylinder_quat", 4),
        density_kg_m3=_number(geometry.get("density_kg_m3"), "geometry.density_kg_m3", minimum=1.0e-6),
        joint_damping_nms=_number(geometry.get("joint_damping_nms"), "geometry.joint_damping_nms", minimum=0.0),
        joint_armature_kg_m2=_number(geometry.get("joint_armature_kg_m2"), "geometry.joint_armature_kg_m2", minimum=0.0),
        friction=_vector(geometry.get("friction"), "geometry.friction", 3, minimum=0.0),
        condim=int(_number(geometry.get("condim"), "geometry.condim", minimum=1.0)),
        contype=int(_number(geometry.get("contype"), "geometry.contype", minimum=1.0)),
        conaffinity=int(_number(geometry.get("conaffinity"), "geometry.conaffinity", minimum=1.0)),
        rgba=_vector(geometry.get("rgba"), "geometry.rgba", 4, minimum=0.0),
        expected_actuator_count=int(_number(contracts.get("expected_actuator_count"), "contracts.expected_actuator_count", minimum=1.0)),
        expected_sensor_data_count=int(_number(contracts.get("expected_sensor_data_count"), "contracts.expected_sensor_data_count", minimum=0.0)),
        wheels=tuple(wheels),
    )


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.7f}" for value in values)


def render_guide_wheel_block(model: GuideWheelModel) -> str:
    lines = [
        BEGIN_MARKER,
        "      <!-- Passive 40x10 mm guide rollers; no actuator or public sensor. -->",
    ]
    for wheel in model.wheels:
        lines.extend(
            (
                f'      <body name="{wheel.name}" pos="{_fmt(wheel.position_m)}">',
                f'        <joint name="{wheel.name}_spin" type="hinge" axis="{_fmt(model.axle_axis)}" damping="{model.joint_damping_nms:.7f}" armature="{model.joint_armature_kg_m2:.7f}" />',
                f'        <site name="{wheel.name}_site" pos="0 0 0" size="0.004" rgba="{_fmt(model.rgba)}" />',
                f'        <geom name="{wheel.name}_contact" type="cylinder" size="{model.radius_m:.7f} {model.half_width_m:.7f}" quat="{_fmt(model.cylinder_quat)}" density="{model.density_kg_m3:.7f}" friction="{_fmt(model.friction)}" condim="{model.condim}" contype="{model.contype}" conaffinity="{model.conaffinity}" rgba="{_fmt(model.rgba)}" />',
                "      </body>",
            )
        )
    lines.append(END_MARKER)
    return "\n".join(lines)


def _replace_block(source: str, block: str) -> str:
    begin_count = source.count(BEGIN_MARKER)
    end_count = source.count(END_MARKER)
    if begin_count == 0 and end_count == 0:
        match = ROOT_BODY_CLOSE.search(source)
        if match is None or ROOT_BODY_CLOSE.search(source, match.end()) is not None:
            raise RuntimeError("expected exactly one robot-body closing anchor before inserting guide wheels")
        return source[:match.start()] + block + "\n" + source[match.start():]
    if begin_count != 1 or end_count != 1:
        raise RuntimeError("guide-wheel MJCF markers must appear exactly once")
    begin = source.index(BEGIN_MARKER)
    end = source.index(END_MARKER, begin) + len(END_MARKER)
    return source[:begin] + block + source[end:]


def apply_guide_wheel_block(
    xml_path: str | Path = DEFAULT_XML,
    config_path: str | Path = DEFAULT_CONFIG,
) -> bool:
    target = Path(xml_path).expanduser().resolve()
    model = load_guide_wheel_model(config_path)
    source = target.read_text(encoding="utf-8")
    ET.fromstring(source)
    result = _replace_block(source, render_guide_wheel_block(model))
    ET.fromstring(result)
    if result == source:
        return False
    target.write_text(result, encoding="utf-8", newline="\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    model = load_guide_wheel_model(args.config)
    if args.check:
        source = args.xml.read_text(encoding="utf-8")
        expected = render_guide_wheel_block(model)
        if BEGIN_MARKER not in source or expected not in source:
            raise RuntimeError("guide-wheel XML block is absent or out of date")
        ET.fromstring(source)
        print(f"validated guide-wheel MJCF block: wheels={len(model.wheels)}")
        return
    changed = apply_guide_wheel_block(args.xml, args.config)
    print(f"synchronized guide-wheel MJCF block: changed={changed}")


if __name__ == "__main__":
    main()
