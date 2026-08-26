"""Build the synthetic RM2025/2026 standard-terrain training MJCF.

The robot subtree is byte-for-byte derived from ``wheeled_infantry.xml``.
Only static support/obstacle geometry, camera, model metadata, and the
projection plane are replaced.  Official dimensions live in
``official_terrain_geometry.yaml``; fixture-only dimensions are explicitly
labelled there.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parent
SOURCE_XML = ROOT / "wheeled_infantry.xml"
GEOMETRY_PATH = ROOT / "official_terrain_geometry.yaml"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _number(value: Any, path: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{path} must be finite and >= {minimum}")
    return number


def _identifier(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ValueError(f"{path} must be an MJCF-safe identifier")
    return value


def _vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9f}" for value in values)


def _warp_compatibility_settings(
    geometry: Mapping[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Return the deliberately tiny MuJoCo-Warp compatibility delta.

    The canonical CPU MJCF remains the official collision reference.  Warp
    rejects positive margins on the doghole's box-to-box chassis proxy pairs
    while MULTICCD is enabled, so the generated GPU-only variant may only set
    the declared proxy margins to zero.  Everything that defines the field
    itself (dimensions, poses, masks, friction, and contact parameters) is
    asserted unchanged by :func:`validate_warp_model_parity`.
    """

    scene = _mapping(geometry["scene"], "scene")
    compatibility = _mapping(scene.get("warp_compatibility"), "scene.warp_compatibility")
    if set(compatibility) != {"output_xml", "disable_collision_geoms", "zero_margin_geoms"}:
        raise ValueError(
            "scene.warp_compatibility must contain output_xml, disable_collision_geoms, "
            "and zero_margin_geoms"
        )
    output_xml = compatibility["output_xml"]
    if not isinstance(output_xml, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.xml", output_xml):
        raise ValueError("scene.warp_compatibility.output_xml must be a local MJCF filename")
    names_raw = compatibility["zero_margin_geoms"]
    if not isinstance(names_raw, list) or not names_raw:
        raise ValueError("scene.warp_compatibility.zero_margin_geoms must be a non-empty list")
    names = tuple(_identifier(name, "scene.warp_compatibility.zero_margin_geoms") for name in names_raw)
    if len(set(names)) != len(names):
        raise ValueError("scene.warp_compatibility.zero_margin_geoms contains duplicates")
    disabled_raw = compatibility["disable_collision_geoms"]
    if not isinstance(disabled_raw, list) or not disabled_raw:
        raise ValueError("scene.warp_compatibility.disable_collision_geoms must be a non-empty list")
    disabled = tuple(
        _identifier(name, "scene.warp_compatibility.disable_collision_geoms")
        for name in disabled_raw
    )
    if len(set(disabled)) != len(disabled):
        raise ValueError("scene.warp_compatibility.disable_collision_geoms contains duplicates")
    return output_xml, names, disabled


def load_geometry(path: Path = GEOMETRY_PATH) -> dict[str, Any]:
    """Read the canonical standard dimensions and validate their geometry."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = dict(_mapping(raw, "root"))
    if root.get("schema_version") != 1:
        raise ValueError("official terrain geometry schema_version must be 1")
    scene = _mapping(root.get("scene"), "scene")
    for key in ("id", "mjcf_model", "follow_camera", "support_prefix", "obstacle_prefix"):
        _identifier(scene.get(key), f"scene.{key}")
    output_xml = scene.get("output_xml")
    if not isinstance(output_xml, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*\.xml", output_xml):
        raise ValueError("scene.output_xml must be a local MJCF filename")
    collision = _mapping(scene.get("collision"), "scene.collision")
    for key in ("track_width_m", "support_thickness_m", "projection_plane_size_m"):
        _number(collision.get(key), f"scene.collision.{key}", minimum=1e-9)
    friction = collision.get("friction")
    solref = collision.get("solref")
    if not isinstance(friction, list) or len(friction) != 3:
        raise ValueError("scene.collision.friction must contain three values")
    if not isinstance(solref, list) or len(solref) != 2:
        raise ValueError("scene.collision.solref must contain two values")
    for index, value in enumerate([*friction, *solref]):
        _number(value, f"scene.collision coefficient {index}", minimum=0.0)
    features = _mapping(root.get("features"), "features")
    if not features:
        raise ValueError("features must not be empty")
    for feature_id, raw_feature in features.items():
        _identifier(feature_id, f"features key {feature_id!r}")
        feature = _mapping(raw_feature, f"features.{feature_id}")
        kind = feature.get("kind")
        if kind not in ("ramp", "fly_ramp", "stepped_platform", "doghole"):
            raise ValueError(f"features.{feature_id}.kind is unsupported")
        official = _mapping(feature.get("official"), f"features.{feature_id}.official")
        fixture = _mapping(feature.get("fixture"), f"features.{feature_id}.fixture")
        if kind in ("ramp", "fly_ramp"):
            angle = _number(official.get("angle_deg"), f"features.{feature_id}.official.angle_deg", minimum=1e-9)
            height = _number(official.get("vertical_height_m"), f"features.{feature_id}.official.vertical_height_m", minimum=1e-9)
            run = _number(official.get("horizontal_run_m"), f"features.{feature_id}.official.horizontal_run_m", minimum=1e-9)
            expected_run = height / math.tan(math.radians(angle))
            if not math.isclose(run, expected_run, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"features.{feature_id}.official.horizontal_run_m={run:.9f} "
                    f"does not match height/tan(angle)={expected_run:.9f}"
                )
            _number(fixture.get("x_start_m"), f"features.{feature_id}.fixture.x_start_m", minimum=-math.inf)
            _number(fixture.get("y_m"), f"features.{feature_id}.fixture.y_m", minimum=-math.inf)
            _number(fixture.get("lead_length_m"), f"features.{feature_id}.fixture.lead_length_m", minimum=1e-9)
        elif kind == "stepped_platform":
            _number(official.get("step_height_m"), f"features.{feature_id}.official.step_height_m", minimum=1e-9)
            segments = feature.get("segments")
            if not isinstance(segments, list) or not segments:
                raise ValueError(f"features.{feature_id}.segments must be a non-empty list")
            for index, segment_value in enumerate(segments):
                segment = _mapping(segment_value, f"features.{feature_id}.segments[{index}]")
                _identifier(segment.get("name"), f"features.{feature_id}.segments[{index}].name")
                _number(segment.get("top_height_m"), f"features.{feature_id}.segments[{index}].top_height_m", minimum=1e-9)
                _number(segment.get("length_m"), f"features.{feature_id}.segments[{index}].length_m", minimum=1e-9)
            _number(fixture.get("x_start_m"), f"features.{feature_id}.fixture.x_start_m", minimum=-math.inf)
            _number(fixture.get("y_m"), f"features.{feature_id}.fixture.y_m", minimum=-math.inf)
            _number(fixture.get("lead_length_m"), f"features.{feature_id}.fixture.lead_length_m", minimum=1e-9)
        else:
            _number(official.get("clear_height_m"), f"features.{feature_id}.official.clear_height_m", minimum=1e-9)
            _number(fixture.get("x_start_m"), f"features.{feature_id}.fixture.x_start_m", minimum=-math.inf)
            _number(fixture.get("y_m"), f"features.{feature_id}.fixture.y_m", minimum=-math.inf)
            for key in ("floor_length_m", "tunnel_length_m", "tunnel_start_offset_m", "clear_width_m", "wall_thickness_m", "roof_thickness_m"):
                _number(fixture.get(key), f"features.{feature_id}.fixture.{key}", minimum=1e-9)
        for key in ("support_geoms",):
            names = feature.get(key)
            if not isinstance(names, list) or not names:
                raise ValueError(f"features.{feature_id}.{key} must be a non-empty list")
            for name in names:
                _identifier(name, f"features.{feature_id}.{key}")
        obstacle_names = feature.get("obstacle_geoms", [])
        if not isinstance(obstacle_names, list):
            raise ValueError(f"features.{feature_id}.obstacle_geoms must be a list when supplied")
        for name in obstacle_names:
            _identifier(name, f"features.{feature_id}.obstacle_geoms")
    _, warp_zero_margin_geoms, warp_disabled_collision_geoms = _warp_compatibility_settings(root)
    expected_warp_zero_margin_geoms = {"base_collision"}
    for feature_value in features.values():
        feature = _mapping(feature_value, "feature")
        expected_warp_zero_margin_geoms.update(
            _identifier(name, "feature.obstacle_geoms")
            for name in feature.get("obstacle_geoms", [])
        )
    if set(warp_zero_margin_geoms) != expected_warp_zero_margin_geoms:
        raise ValueError(
            "scene.warp_compatibility.zero_margin_geoms must contain base_collision "
            "and every declared obstacle geom exactly once"
        )
    if set(warp_disabled_collision_geoms) != {"ground"}:
        raise ValueError("scene.warp_compatibility.disable_collision_geoms must contain only ground")
    return root


def _support_box(name: str, *, center: tuple[float, float, float], size: tuple[float, float, float], friction: str, solref: str, quat: tuple[float, float, float, float] | None = None) -> str:
    attributes = [
        f'name="{name}"',
        'type="box"',
        f'pos="{_vector(center)}"',
        f'size="{_vector(size)}"',
        f'friction="{friction}"',
        f'solref="{solref}"',
        'contype="2"',
        'conaffinity="1"',
        'rgba="0.32 0.38 0.32 1"',
    ]
    if quat is not None:
        attributes.append(f'quat="{_vector(quat)}"')
    return "    <geom " + " ".join(attributes) + " />"


def _obstacle_box(name: str, *, center: tuple[float, float, float], size: tuple[float, float, float], friction: str, solref: str) -> str:
    return (
        f'    <geom name="{name}" type="box" pos="{_vector(center)}" '
        f'size="{_vector(size)}" friction="{friction}" solref="{solref}" '
        'contype="2" conaffinity="5" rgba="0.42 0.25 0.18 1" />'
    )


def _flat_box(name: str, x_start: float, length: float, y: float, top_height: float, half_width: float, thickness: float, friction: str, solref: str) -> str:
    return _support_box(
        name,
        center=(x_start + 0.5 * length, y, top_height - thickness),
        size=(0.5 * length, half_width, thickness),
        friction=friction,
        solref=solref,
    )


def _ramp_box(name: str, x_start: float, y: float, run: float, height: float, half_width: float, thickness: float, friction: str, solref: str) -> str:
    angle = math.atan2(height, run)
    slope_length = math.hypot(run, height)
    theta = -angle
    # The top face midpoint must be (x_start + run/2, y, height/2).  MuJoCo
    # stores a box pose at its volume centre, so offset it by its local +Z
    # half-extent through the negative-Y pitch rotation.
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    center = (
        x_start + 0.5 * run - sin_theta * thickness,
        y,
        0.5 * height - cos_theta * thickness,
    )
    quat = (math.cos(0.5 * theta), 0.0, math.sin(0.5 * theta), 0.0)
    return _support_box(
        name,
        center=center,
        size=(0.5 * slope_length, half_width, thickness),
        friction=friction,
        solref=solref,
        quat=quat,
    )


def terrain_world_block(geometry: Mapping[str, Any], newline: bytes) -> bytes:
    scene = _mapping(geometry["scene"], "scene")
    collision = _mapping(scene["collision"], "scene.collision")
    half_width = 0.5 * _number(collision["track_width_m"], "scene.collision.track_width_m")
    thickness = _number(collision["support_thickness_m"], "scene.collision.support_thickness_m")
    friction = _vector(tuple(float(value) for value in collision["friction"]))
    solref = _vector(tuple(float(value) for value in collision["solref"]))
    plane_size = _number(collision["projection_plane_size_m"], "scene.collision.projection_plane_size_m")
    blocks = [
        b"    <!-- The inherited plane is used only during LQR trim projection. -->",
        (
            f'    <geom name="ground" type="plane" pos="0 0 0" size="{plane_size:.3f} {plane_size:.3f} 0.1" '
            f'material="ground_black_mat" friction="{friction}" contype="2" conaffinity="1" rgba="0 0 0 0" />'
        ).encode(),
        b"    <!-- support_* are collision surfaces; obstacle_* are non-support collisions. -->",
    ]
    features = _mapping(geometry["features"], "features")
    for feature_id, raw_feature in features.items():
        feature = _mapping(raw_feature, f"features.{feature_id}")
        kind = str(feature["kind"])
        official = _mapping(feature["official"], f"features.{feature_id}.official")
        fixture = _mapping(feature["fixture"], f"features.{feature_id}.fixture")
        support_names = tuple(str(name) for name in feature["support_geoms"])
        x_start = float(fixture["x_start_m"])
        y = float(fixture["y_m"])
        if kind in ("ramp", "fly_ramp"):
            lead_length = float(fixture["lead_length_m"])
            run = float(official["horizontal_run_m"])
            height = float(official["vertical_height_m"])
            platform_length = float(
                fixture["landing_platform_length_m"] if kind == "fly_ramp" else fixture["platform_length_m"]
            )
            blocks.extend(
                (
                    f"    <!-- {feature_id}: official ramp angle/height/run from geometry YAML. -->".encode(),
                    _flat_box(support_names[0], x_start - lead_length, lead_length, y, 0.0, half_width, thickness, friction, solref).encode(),
                    _ramp_box(support_names[1], x_start, y, run, height, half_width, thickness, friction, solref).encode(),
                )
            )
            landing_start = x_start + run
            if kind == "fly_ramp":
                landing_start += float(official["minimum_landing_distance_m"])
            blocks.append(
                _flat_box(support_names[2], landing_start, platform_length, y, 0.0 if kind == "fly_ramp" else height, half_width, thickness, friction, solref).encode()
            )
        elif kind == "stepped_platform":
            lead_length = float(fixture["lead_length_m"])
            blocks.append(
                f"    <!-- {feature_id}: official vertical rises; tread lengths are fixture-only. -->".encode()
            )
            blocks.append(
                _flat_box(support_names[0], x_start - lead_length, lead_length, y, 0.0, half_width, thickness, friction, solref).encode()
            )
            cursor = x_start
            segments = feature["segments"]
            for index, raw_segment in enumerate(segments):
                segment = _mapping(raw_segment, f"features.{feature_id}.segments[{index}]")
                blocks.append(
                    _flat_box(
                        support_names[index + 1],
                        cursor,
                        float(segment["length_m"]),
                        y,
                        float(segment["top_height_m"]),
                        half_width,
                        thickness,
                        friction,
                        solref,
                    ).encode()
                )
                cursor += float(segment["length_m"])
        else:
            floor_length = float(fixture["floor_length_m"])
            clear_height = float(official["clear_height_m"])
            tunnel_length = float(fixture["tunnel_length_m"])
            tunnel_start_offset = float(fixture["tunnel_start_offset_m"])
            clear_width = float(fixture["clear_width_m"])
            wall_thickness = float(fixture["wall_thickness_m"])
            roof_thickness = float(fixture["roof_thickness_m"])
            obstacle_names = tuple(str(name) for name in feature["obstacle_geoms"])
            blocks.append(b"    <!-- Doghole roof/walls are true collision obstacles, not hfield support. -->")
            blocks.append(
                _flat_box(support_names[0], x_start, floor_length, y, 0.0, half_width, thickness, friction, solref).encode()
            )
            tunnel_x = x_start + tunnel_start_offset + 0.5 * tunnel_length
            wall_half_height = 0.5 * clear_height + 0.5 * roof_thickness
            wall_center_z = wall_half_height - roof_thickness
            wall_y = 0.5 * clear_width + wall_thickness
            blocks.extend(
                (
                    _obstacle_box(obstacle_names[0], center=(tunnel_x, y, clear_height + 0.5 * roof_thickness), size=(0.5 * tunnel_length, 0.5 * clear_width + wall_thickness, 0.5 * roof_thickness), friction=friction, solref=solref).encode(),
                    _obstacle_box(obstacle_names[1], center=(tunnel_x, y - wall_y, wall_center_z), size=(0.5 * tunnel_length, wall_thickness, wall_half_height), friction=friction, solref=solref).encode(),
                    _obstacle_box(obstacle_names[2], center=(tunnel_x, y + wall_y, wall_center_z), size=(0.5 * tunnel_length, wall_thickness, wall_half_height), friction=friction, solref=solref).encode(),
                )
            )
    blocks.append(
        f'    <camera name="{scene["follow_camera"]}" mode="trackcom" target="robot" pos="-2.8 -2.8 1.6" fovy="55" />'.encode()
    )
    return newline.join(blocks)


def _replace_once(raw: bytes, old: bytes, new: bytes, description: str) -> bytes:
    if raw.count(old) != 1:
        raise RuntimeError(f"expected exactly one {description}, found {raw.count(old)}")
    return raw.replace(old, new, 1)


def generate_xml(geometry_path: Path = GEOMETRY_PATH) -> tuple[bytes, dict[str, Any]]:
    geometry = load_geometry(geometry_path)
    scene = _mapping(geometry["scene"], "scene")
    source = SOURCE_XML.read_bytes()
    newline = b"\r\n" if b"\r\n" in source else b"\n"
    result = _replace_once(
        source,
        b'<mujoco model="wheeled_infantry">',
        f'<mujoco model="{scene["mjcf_model"]}">'.encode(),
        "source model tag",
    )
    result = _replace_once(
        result,
        b'  <statistic center="0 0 0.25" extent="1.5" />',
        b'  <statistic center="-4 0 0.35" extent="18" />',
        "source statistic",
    )
    pattern = re.compile(
        rb'    <!-- Static ground: collision group 2 accepts contacts from dynamic group 1\. -->\r?\n'
        rb'    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0\.1" material="ground_black_mat" '
        rb'friction="1\.10 0\.05 0\.01" contype="2" conaffinity="1" />'
    )
    matches = list(pattern.finditer(result))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one static ground block, found {len(matches)}")
    result = pattern.sub(terrain_world_block(geometry, newline), result, count=1)
    robot_marker = b'    <body name="robot"'
    if source[source.index(robot_marker):] != result[result.index(robot_marker):]:
        raise RuntimeError("the robot body suffix changed while generating official terrain")
    return result, geometry


def _set_geom_margin_zero(xml: bytes, geom_name: str) -> bytes:
    """Set one named geom's explicit margin without reformatting the MJCF."""

    escaped_name = re.escape(geom_name.encode("ascii"))
    pattern = re.compile(
        rb'(<geom\b(?=[^>]*\bname="' + escaped_name + rb'")[^>]*?)(/>)'
    )
    matches = tuple(pattern.finditer(xml))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one geom named {geom_name!r} while generating the Warp variant, "
            f"found {len(matches)}"
        )
    match = matches[0]
    attributes = match.group(1)
    if re.search(rb'\bmargin\s*=', attributes):
        raise RuntimeError(
            f"geom {geom_name!r} already has an explicit margin; the Warp compatibility delta must be unambiguous"
        )
    return (
        xml[:match.start()]
        + attributes.rstrip()
        + b' margin="0"'
        + match.group(2)
        + xml[match.end():]
    )


def _disable_geom_collision(xml: bytes, geom_name: str) -> bytes:
    """Set the named fixture geom's collision mask to zero without changing it."""

    escaped_name = re.escape(geom_name.encode("ascii"))
    pattern = re.compile(
        rb'(<geom\b(?=[^>]*\bname="' + escaped_name + rb'")[^>]*?)(/>)'
    )
    matches = tuple(pattern.finditer(xml))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one geom named {geom_name!r} while generating the Warp variant, "
            f"found {len(matches)}"
        )
    match = matches[0]
    attributes = match.group(1)
    for attribute in (b"contype", b"conaffinity"):
        attribute_pattern = re.compile(rb'\b' + attribute + rb'="[^"]*"')
        if len(attribute_pattern.findall(attributes)) != 1:
            raise RuntimeError(
                f"geom {geom_name!r} must have exactly one explicit {attribute.decode('ascii')} attribute"
            )
        attributes = attribute_pattern.sub(attribute + b'="0"', attributes, count=1)
    return xml[:match.start()] + attributes + match.group(2) + xml[match.end():]


def generate_warp_xml(geometry_path: Path = GEOMETRY_PATH) -> tuple[bytes, dict[str, Any]]:
    """Generate the GPU-only standard-field MJCF compatibility variant.

    MuJoCo-Warp 3.11 rejects the original positive-margin doghole box pairs
    with MULTICCD enabled.  This function deliberately preserves the canonical
    scene byte-for-byte except for the YAML-declared proxy-margin and
    projection-plane collision-mask overrides.  The plane is a CPU trim
    projection aid and is disabled by the CPU terrain environment as well.
    """

    result, geometry = generate_xml(geometry_path)
    _, zero_margin_geoms, disable_collision_geoms = _warp_compatibility_settings(geometry)
    for geom_name in disable_collision_geoms:
        result = _disable_geom_collision(result, geom_name)
    for geom_name in zero_margin_geoms:
        result = _set_geom_margin_zero(result, geom_name)
    return result, geometry


def _object_names(model: Any, object_type: Any, count: int) -> tuple[str, ...]:
    import mujoco

    return tuple(
        "" if (name := mujoco.mj_id2name(model, object_type, index)) is None else name
        for index in range(count)
    )


def validate_output(output_path: Path, geometry: Mapping[str, Any]) -> None:
    import mujoco

    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    model = mujoco.MjModel.from_xml_path(str(output_path))
    comparisons = (
        ("body", mujoco.mjtObj.mjOBJ_BODY, source_model.nbody, model.nbody),
        ("joint", mujoco.mjtObj.mjOBJ_JOINT, source_model.njnt, model.njnt),
        ("equality", mujoco.mjtObj.mjOBJ_EQUALITY, source_model.neq, model.neq),
        ("tendon", mujoco.mjtObj.mjOBJ_TENDON, source_model.ntendon, model.ntendon),
        ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, source_model.nu, model.nu),
        ("sensor", mujoco.mjtObj.mjOBJ_SENSOR, source_model.nsensor, model.nsensor),
    )
    for label, object_type, source_count, output_count in comparisons:
        if source_count != output_count:
            raise RuntimeError(f"{label} count changed: {source_count} -> {output_count}")
        if _object_names(source_model, object_type, source_count) != _object_names(model, object_type, output_count):
            raise RuntimeError(f"{label} names changed while adding official terrain")
    scene = _mapping(geometry["scene"], "scene")
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(scene["follow_camera"]))
    if camera_id < 0:
        raise RuntimeError("official follow camera is missing")
    support_names = []
    obstacle_names = []
    for feature_value in _mapping(geometry["features"], "features").values():
        feature = _mapping(feature_value, "feature")
        support_names.extend(feature["support_geoms"])
        obstacle_names.extend(feature.get("obstacle_geoms", []))
    for name in support_names + obstacle_names:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(name))
        if geom_id < 0:
            raise RuntimeError(f"official terrain geom is missing: {name}")
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_BOX:
            raise RuntimeError(f"official terrain geom is not a box: {name}")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for feature_id, feature_value in _mapping(geometry["features"], "features").items():
        feature = _mapping(feature_value, f"features.{feature_id}")
        if feature["kind"] not in ("ramp", "fly_ramp"):
            continue
        official = _mapping(feature["official"], f"features.{feature_id}.official")
        fixture = _mapping(feature["fixture"], f"features.{feature_id}.fixture")
        ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, str(feature["support_geoms"][1]))
        rotation = data.geom_xmat[ramp_id].reshape(3, 3)
        local_x = rotation[:, 0]
        top_normal = rotation[:, 2]
        expected_angle = math.radians(float(official["angle_deg"]))
        measured_slope = math.atan2(abs(local_x[2]), max(1e-12, abs(local_x[0])))
        if not math.isclose(measured_slope, expected_angle, rel_tol=0.0, abs_tol=1e-6):
            raise RuntimeError(f"{feature_id} ramp angle differs from canonical geometry")
        if top_normal[2] <= 0.0:
            raise RuntimeError(f"{feature_id} ramp top normal is inverted")
        x_start = float(fixture["x_start_m"])
        run = float(official["horizontal_run_m"])
        height = float(official["vertical_height_m"])
        half_x = float(model.geom_size[ramp_id, 0])
        half_z = float(model.geom_size[ramp_id, 2])
        center = data.geom_xpos[ramp_id]
        low_top = center + rotation @ np.array((-half_x, 0.0, half_z))
        high_top = center + rotation @ np.array((half_x, 0.0, half_z))
        if not (math.isclose(low_top[0], x_start, abs_tol=2e-6) and math.isclose(low_top[2], 0.0, abs_tol=2e-6) and math.isclose(high_top[0], x_start + run, abs_tol=2e-6) and math.isclose(high_top[2], height, abs_tol=2e-6)):
            raise RuntimeError(f"{feature_id} ramp endpoints do not match canonical dimensions")
        if feature["kind"] == "fly_ramp":
            landing_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, str(feature["support_geoms"][2])
            )
            landing_x0 = (
                float(data.geom_xpos[landing_id, 0])
                - float(model.geom_size[landing_id, 0])
            )
            landing_distance = landing_x0 - float(high_top[0])
            minimum_distance = float(official["minimum_landing_distance_m"])
            if not math.isclose(landing_distance, minimum_distance, abs_tol=2e-6):
                raise RuntimeError(
                    f"{feature_id} landing gap {landing_distance:.6f}m does not match "
                    f"the canonical {minimum_distance:.6f}m"
                )
            if landing_distance + 1e-9 < minimum_distance:
                raise RuntimeError(f"{feature_id} landing gap is below the official minimum")
    for feature_id, feature_value in _mapping(geometry["features"], "features").items():
        feature = _mapping(feature_value, f"features.{feature_id}")
        kind = feature["kind"]
        official = _mapping(feature["official"], f"features.{feature_id}.official")
        if kind == "stepped_platform":
            segments = feature["segments"]
            step_height = float(official["step_height_m"])
            for index, raw_segment in enumerate(segments):
                segment = _mapping(raw_segment, f"features.{feature_id}.segments[{index}]")
                geom_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_GEOM, str(feature["support_geoms"][index + 1])
                )
                top_height = float(data.geom_xpos[geom_id, 2] + model.geom_size[geom_id, 2])
                expected_height = float(segment["top_height_m"])
                if not math.isclose(top_height, expected_height, abs_tol=2e-6):
                    raise RuntimeError(
                        f"{feature_id} segment {segment['name']!r} top is {top_height:.6f}m, "
                        f"not {expected_height:.6f}m"
                    )
                expected_multiple = (index + 1) * step_height
                if not math.isclose(expected_height, expected_multiple, abs_tol=2e-6):
                    raise RuntimeError(
                        f"{feature_id} segment {segment['name']!r} does not preserve its "
                        "official vertical-step increment"
                    )
        elif kind == "doghole":
            roof_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, str(feature["obstacle_geoms"][0])
            )
            roof_underside = float(data.geom_xpos[roof_id, 2] - model.geom_size[roof_id, 2])
            required_clear_height = float(official["clear_height_m"])
            if not math.isclose(roof_underside, required_clear_height, abs_tol=2e-6):
                raise RuntimeError(
                    f"{feature_id} roof underside {roof_underside:.6f}m does not match "
                    f"the canonical clear height {required_clear_height:.6f}m"
                )
    print(
        f"validated {output_path.name}: bodies={model.nbody}, joints={model.njnt}, "
        f"equalities={model.neq}, tendons={model.ntendon}, actuators={model.nu}, sensors={model.nsensor}, "
        f"static_supports={len(support_names)}, static_obstacles={len(obstacle_names)}"
    )


_PRESERVED_GEOM_FIELDS = (
    "geom_type",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_priority",
    "geom_bodyid",
    "geom_dataid",
    "geom_group",
    "geom_sameframe",
    "geom_pos",
    "geom_quat",
    "geom_size",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_solmix",
    "geom_gap",
    "geom_fluid",
)

# Disabling the inherited projection-plane collision mask makes MuJoCo rebuild
# its broad-phase BVH. Those caches are intentionally excluded; every array
# that defines robot, actuator, sensor, terrain, or solver dynamics must stay
# exact between the canonical model and the Warp compatibility variant.
_WARP_VARIANT_BVH_FIELDS = {
    "_sizes",
    "body_bvhadr",
    "body_bvhnum",
    "bvh_aabb",
    "bvh_child",
    "bvh_depth",
    "bvh_nodeid",
    "mesh_bvhadr",
}
_WARP_VARIANT_EXPLICIT_GEOM_FIELDS = {
    "geom_margin",
    "geom_contype",
    "geom_conaffinity",
}
_MODEL_STRUCTURE_FIELDS = (
    "nq", "nv", "na", "nu", "nbody", "njnt", "neq", "ntendon", "nsensor", "ngeom",
    "nsite", "ncam", "nmesh", "nhfield", "ntex", "nmat", "npair", "nexclude", "nkey",
    "nnumeric", "ntext", "ntuple", "nplugin",
)


def _geom_ids(model: Any) -> dict[str, int]:
    import mujoco

    result: dict[str, int] = {}
    for index in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        if name is not None:
            result[name] = index
    return result


def _assert_equal_except_indices(
    field: str,
    cpu_value: np.ndarray,
    warp_value: np.ndarray,
    allowed_indices: set[int],
) -> None:
    if cpu_value.shape != warp_value.shape:
        raise RuntimeError(f"Warp variant changed shape of {field}: {cpu_value.shape} -> {warp_value.shape}")
    comparison = np.array(cpu_value == warp_value, copy=True)
    if comparison.ndim > 1:
        comparison = comparison.reshape(comparison.shape[0], -1).all(axis=1)
    if allowed_indices:
        comparison[list(allowed_indices)] = True
    if not bool(np.all(comparison)):
        mismatch = int(np.flatnonzero(~comparison)[0])
        raise RuntimeError(f"Warp variant changed {field} for non-whitelisted geom index {mismatch}")


def _assert_dynamic_model_parity(canonical: Any, variant: Any) -> None:
    """Reject all non-whitelisted compiled-model differences."""

    for field in _MODEL_STRUCTURE_FIELDS:
        if hasattr(canonical, field) and int(getattr(canonical, field)) != int(getattr(variant, field)):
            raise RuntimeError(
                f"Warp variant changed {field}: {getattr(canonical, field)} -> {getattr(variant, field)}"
            )
    if canonical.names != variant.names:
        raise RuntimeError("Warp variant changed the canonical name table")

    for field in dir(canonical):
        if field.startswith("_") or field in _WARP_VARIANT_BVH_FIELDS:
            continue
        if field in _WARP_VARIANT_EXPLICIT_GEOM_FIELDS:
            continue
        cpu_value = getattr(canonical, field)
        if not isinstance(cpu_value, np.ndarray):
            continue
        warp_value = getattr(variant, field)
        if not isinstance(warp_value, np.ndarray) or cpu_value.shape != warp_value.shape:
            raise RuntimeError(
                f"Warp variant changed dynamic model field {field}: {cpu_value.shape} -> "
                f"{getattr(warp_value, 'shape', None)}"
            )
        if not np.array_equal(cpu_value, warp_value):
            raise RuntimeError(f"Warp variant changed dynamic model field {field}")

    for field in dir(canonical.opt):
        if field.startswith("_"):
            continue
        cpu_value = getattr(canonical.opt, field)
        if not isinstance(cpu_value, (int, float, np.ndarray)):
            continue
        if not np.array_equal(np.asarray(cpu_value), np.asarray(getattr(variant.opt, field))):
            raise RuntimeError(f"Warp variant changed simulation option {field}")


def validate_warp_model_parity(
    canonical_xml_path: Path,
    warp_xml_path: Path,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the Warp scene differs only by its declared delta.

    This is intentionally a model-contract check rather than a trajectory
    parity claim.  It proves that standard-field geometry and all collision
    properties survive the conversion, while keeping the exact four margin and
    one invisible-plane collision-mask differences explicit for later dynamic
    CPU-vs-Warp gates.
    """

    import mujoco

    canonical = mujoco.MjModel.from_xml_path(str(canonical_xml_path))
    variant = mujoco.MjModel.from_xml_path(str(warp_xml_path))
    _assert_dynamic_model_parity(canonical, variant)
    canonical_names = _geom_ids(canonical)
    variant_names = _geom_ids(variant)
    if canonical_names != variant_names:
        raise RuntimeError("Warp variant changed the canonical geom name-to-index map")

    _, zero_margin_names, disabled_collision_names = _warp_compatibility_settings(geometry)
    for name in zero_margin_names + disabled_collision_names:
        if name not in canonical_names:
            raise RuntimeError(f"Warp compatibility whitelist geom is absent from the canonical model: {name}")
    zero_margin_ids = {canonical_names[name] for name in zero_margin_names}
    disabled_collision_ids = {canonical_names[name] for name in disabled_collision_names}

    for field in _PRESERVED_GEOM_FIELDS:
        allowed = disabled_collision_ids if field in {"geom_contype", "geom_conaffinity"} else set()
        _assert_equal_except_indices(
            field,
            np.asarray(getattr(canonical, field)),
            np.asarray(getattr(variant, field)),
            allowed,
        )
    _assert_equal_except_indices(
        "geom_margin",
        np.asarray(canonical.geom_margin),
        np.asarray(variant.geom_margin),
        zero_margin_ids,
    )

    for name in zero_margin_names:
        index = canonical_names[name]
        if float(canonical.geom_margin[index]) <= 0.0 or float(variant.geom_margin[index]) != 0.0:
            raise RuntimeError(f"Warp compatibility margin delta is invalid for {name}")
    for name in disabled_collision_names:
        index = canonical_names[name]
        if int(canonical.geom_contype[index]) == 0 or int(canonical.geom_conaffinity[index]) == 0:
            raise RuntimeError(f"canonical projection plane collision is unexpectedly disabled: {name}")
        if int(variant.geom_contype[index]) != 0 or int(variant.geom_conaffinity[index]) != 0:
            raise RuntimeError(f"Warp projection plane collision was not disabled: {name}")

    base_index = canonical_names["base_collision"]
    for name in zero_margin_names:
        if name == "base_collision":
            continue
        obstacle_index = canonical_names[name]
        canonical_collides = bool(
            (int(canonical.geom_contype[base_index]) & int(canonical.geom_conaffinity[obstacle_index]))
            or (int(canonical.geom_contype[obstacle_index]) & int(canonical.geom_conaffinity[base_index]))
        )
        variant_collides = bool(
            (int(variant.geom_contype[base_index]) & int(variant.geom_conaffinity[obstacle_index]))
            or (int(variant.geom_contype[obstacle_index]) & int(variant.geom_conaffinity[base_index]))
        )
        if not canonical_collides or not variant_collides:
            raise RuntimeError(f"Warp variant removed doghole collision for {name}")

    cpu_data = mujoco.MjData(canonical)
    warp_data = mujoco.MjData(variant)
    mujoco.mj_forward(canonical, cpu_data)
    mujoco.mj_forward(variant, warp_data)
    if not np.array_equal(cpu_data.geom_xpos, warp_data.geom_xpos) or not np.array_equal(
        cpu_data.geom_xmat, warp_data.geom_xmat
    ):
        raise RuntimeError("Warp variant changed a canonical geom world pose")
    return {
        "canonical_xml": str(canonical_xml_path),
        "warp_xml": str(warp_xml_path),
        "zero_margin_geoms": zero_margin_names,
        "disabled_collision_geoms": disabled_collision_names,
        "static_geometry_preserved": True,
    }


def validate_official_warp_scene(
    warp_xml_path: str | Path,
    *,
    geometry_path: str | Path = GEOMETRY_PATH,
) -> dict[str, Any]:
    """Validate an official GPU scene against the immutable canonical MJCF.

    Training launchers should call this before allocating a CUDA batch.  The
    function deliberately fails for renamed or hand-edited variants, so a
    removed doghole, re-enabled ground plane, or broadened compatibility delta
    never silently reaches a GPU rollout.
    """

    geometry_file = Path(geometry_path).expanduser().resolve()
    geometry = load_geometry(geometry_file)
    scene = _mapping(geometry["scene"], "scene")
    warp_output_xml, _, _ = _warp_compatibility_settings(geometry)
    variant_path = Path(warp_xml_path).expanduser().resolve()
    if variant_path.name != warp_output_xml:
        raise RuntimeError(
            f"official Warp scene must be named {warp_output_xml!r}, got {variant_path.name!r}"
        )
    if not variant_path.is_file():
        raise RuntimeError(f"official Warp scene does not exist: {variant_path}")
    canonical_path = geometry_file.parent / str(scene["output_xml"])
    if not canonical_path.is_file():
        raise RuntimeError(f"canonical official scene does not exist: {canonical_path}")
    validate_output(variant_path, geometry)
    return validate_warp_model_parity(canonical_path, variant_path, geometry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=GEOMETRY_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--warp-compatible",
        action="store_true",
        help="generate the YAML-whitelisted MuJoCo-Warp compatibility variant",
    )
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    geometry_path = args.geometry.expanduser().resolve()
    output, geometry = (
        generate_warp_xml(geometry_path) if args.warp_compatible else generate_xml(geometry_path)
    )
    scene = _mapping(geometry["scene"], "scene")
    default_output = (
        _warp_compatibility_settings(geometry)[0] if args.warp_compatible else str(scene["output_xml"])
    )
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else ROOT / default_output
    )
    output_path.write_bytes(output)
    if not args.skip_validation:
        if args.warp_compatible:
            validate_official_warp_scene(output_path, geometry_path=geometry_path)
        else:
            validate_output(output_path, geometry)
    print(f"generated: {output_path}")


if __name__ == "__main__":
    main()
