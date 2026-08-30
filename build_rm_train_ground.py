"""Build the RMUC training-ground MJCF without touching the wheeled robot topology.

The generated file is deliberately text-derived from ``wheeled_infantry.xml``.
Only the static world terrain, visual RMUC assets, model statistic, and camera
are changed.  This avoids an XML reserialization that could alter the existing
closed-chain robot structure.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
SOURCE_XML = ROOT / "wheeled_infantry.xml"
OUTPUT_XML = ROOT / "rm_train_ground.xml"
STAGED_ASSET_DIR = ROOT / "assets" / "rmuc"
# Keep the supplied 2048x1146 image as the visual/source asset.  MuJoCo-Warp's
# hfield narrowphase caps the contact manifold for each geom pair at
# MJ_MAXCONPAIR (50).  A 1024x573 collision grid lets the wheel and guide-wheel
# footprints cover more than that many cells on the RMUC mesh and therefore
# raises a deterministic HFIELD overflow.  The 512x286 grid keeps the same
# surveyed extents/elevation range while bounding the per-pair prism count.
COLLISION_HFIELD_NAME = "hfield_collision_512x286.png"
COLLISION_HFIELD_RESOLUTION = (512, 286)

ASSET_NAMES = (
    "hfield.png",
    "RMUC20261-floor.obj",
    "RMUC20261-floor.mtl",
    "RMUC20261-building_vis.obj",
    "RMUC20261-building_vis.mtl",
)

# The hfield is positioned so the largest surveyed flat RMUC patch is exactly
# at the inherited robot origin. The source OBJ uses Blender Y-up coordinates;
# the visual meshes use a +90 degree X rotation to map it into MuJoCo Z-up.
TERRAIN_HFIELD_POS = (-12.112445, -2.726397, -0.196745)
VISUAL_SCENE_POS = (-27.013445, -10.751400, -0.200000)
HFIELD_SIZE = (14.525, 8.025, 1.730, 0.100)
# The collision hfield needs a contact time constant large enough to keep the
# two wheel manifolds continuous at the 1 kHz controller rate and symmetric
# through a high-jump landing.  Values below 0.12 s amplified the two closed
# chain impact mismatch.  This remains stiffer than the former 0.20 s setting
# while preserving the validated RMUC landing envelope.
TERRAIN_CONTACT_SOLREF = "0.150 1"
# The Y-up Blender OBJ assets need a +90 degree X input rotation. MuJoCo's
# internal mesh principal-axis transform is accounted for by this calibrated
# input quaternion, yielding world coordinates (source_x, -source_z, source_y).
VISUAL_SCENE_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate rm_train_ground.xml from the untouched wheeled robot MJCF."
    )
    parser.add_argument(
        "--asset-source",
        type=Path,
        default=STAGED_ASSET_DIR,
        help=(
            "Directory containing the supplied RMUC PNG/OBJ/MTL assets. "
            "The staged assets directory is used by default after the first build."
        ),
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Write and stage files without compiling the generated MJCF.",
    )
    return parser.parse_args()


def require_single(raw: bytes, needle: bytes, description: str) -> int:
    count = raw.count(needle)
    if count != 1:
        raise RuntimeError(f"expected exactly one {description}, found {count}")
    return raw.index(needle)


def stage_assets(source_dir: Path) -> None:
    source_dir = source_dir.resolve()
    STAGED_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        source = source_dir / name
        destination = STAGED_ASSET_DIR / name
        if not source.is_file():
            raise FileNotFoundError(f"missing RMUC asset: {source}")
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    build_collision_hfield(STAGED_ASSET_DIR / "hfield.png")


def build_collision_hfield(source: Path) -> None:
    """Produce a Warp-safe collision grid while retaining the supplied image."""
    destination = STAGED_ASSET_DIR / COLLISION_HFIELD_NAME
    with Image.open(source) as image:
        if image.size != (2048, 1146):
            raise RuntimeError(f"expected supplied hfield.png to be 2048x1146, got {image.size}")
        collision_image = image.resize(COLLISION_HFIELD_RESOLUTION, Image.Resampling.BOX)
        try:
            collision_image.save(destination)
        finally:
            collision_image.close()


def xml_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def terrain_asset_block(newline: bytes) -> bytes:
    hfield_size = xml_vector(HFIELD_SIZE)
    return newline.join(
        (
            b"    <!-- RMUC assets: hfield supplies collision; OBJ meshes are visual-only. -->",
            f'    <hfield name="rmuc_training_hfield" file="../assets/rmuc/{COLLISION_HFIELD_NAME}" size="{hfield_size}" />'.encode(),
            b'    <mesh name="rmuc_floor_visual" file="../assets/rmuc/RMUC20261-floor.obj" />',
            b'    <mesh name="rmuc_building_visual" file="../assets/rmuc/RMUC20261-building_vis.obj" />',
        )
    )


def terrain_world_block(newline: bytes) -> bytes:
    terrain_pos = xml_vector(TERRAIN_HFIELD_POS)
    visual_pos = xml_vector(VISUAL_SCENE_POS)
    visual_quat = xml_vector(VISUAL_SCENE_QUAT)
    return newline.join(
        (
            b"    <!-- The original flat plane is enabled only while Env projects the LQR working point. -->",
            b'    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0.1" material="ground_black_mat" friction="1.10 0.05 0.01" contype="2" conaffinity="1" rgba="0 0 0 0" />',
            b"    <!-- Env activates this co-located RMUC collision hfield immediately after LQR projection. -->",
            f'    <geom name="rmuc_terrain" type="hfield" hfield="rmuc_training_hfield" pos="{terrain_pos}" friction="1.10 0.05 0.01" solref="{TERRAIN_CONTACT_SOLREF}" contype="2" conaffinity="1" rgba="0.20 0.26 0.20 0" />'.encode(),
            b"    <!-- Source OBJ meshes are rendering-only and never participate in RL collision queries. -->",
            f'    <geom name="rmuc_floor_visual_geom" type="mesh" mesh="rmuc_floor_visual" pos="{visual_pos}" quat="{visual_quat}" contype="0" conaffinity="0" group="0" rgba="0.48 0.51 0.47 1" />'.encode(),
            f'    <geom name="rmuc_building_visual_geom" type="mesh" mesh="rmuc_building_visual" pos="{visual_pos}" quat="{visual_quat}" contype="0" conaffinity="0" group="0" rgba="0.72 0.72 0.72 1" />'.encode(),
            b"    <!-- Viewer scripts select this track-com camera for a stable third-person follow view. -->",
            b'    <camera name="rmuc_follow_camera" mode="trackcom" target="robot" pos="-2.8 -2.8 1.6" fovy="55" />',
        )
    )


def replace_once(raw: bytes, old: bytes, new: bytes, description: str) -> bytes:
    require_single(raw, old, description)
    return raw.replace(old, new, 1)


def generate_xml() -> bytes:
    if not SOURCE_XML.is_file():
        raise FileNotFoundError(f"missing wheeled robot source XML: {SOURCE_XML}")
    source = SOURCE_XML.read_bytes()
    newline = b"\r\n" if b"\r\n" in source else b"\n"

    result = replace_once(
        source,
        b'<mujoco model="wheeled_infantry">',
        b'<mujoco model="rm_train_ground">',
        "source model tag",
    )
    result = replace_once(
        result,
        b'  <statistic center="0 0 0.25" extent="1.5" />',
        b'  <statistic center="0 0 0.30" extent="18" />',
        "source statistic",
    )

    asset_anchor = b'    <mesh name="base_visual_plastic" file="base_visual_plastic.stl" scale="0.001 0.001 0.001" />'
    require_single(result, asset_anchor, "final source mesh asset")
    result = result.replace(
        asset_anchor,
        asset_anchor + newline + terrain_asset_block(newline),
        1,
    )

    ground_pattern = re.compile(
        rb'    <!-- Static ground: collision group 2 accepts contacts from dynamic group 1\. -->\r?\n'
        rb'    <geom name="ground" type="plane" pos="0 0 0" size="100 100 0\.1" material="ground_black_mat" '
        rb'friction="1\.10 0\.05 0\.01" contype="2" conaffinity="1" />'
    )
    matches = list(ground_pattern.finditer(result))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one static ground block, found {len(matches)}")
    result = ground_pattern.sub(terrain_world_block(newline), result, count=1)

    # The robot body and all following source content must remain byte-identical.
    robot_marker = b'    <body name="robot"'
    source_robot = source[source.index(robot_marker) :]
    output_robot = result[result.index(robot_marker) :]
    if source_robot != output_robot:
        raise RuntimeError("the robot body suffix changed while generating RMUC terrain")
    return result


def object_names(model: object, object_type: object, count: int) -> tuple[str, ...]:
    import mujoco

    names: list[str] = []
    for object_id in range(count):
        name = mujoco.mj_id2name(model, object_type, object_id)
        names.append("" if name is None else name)
    return tuple(names)


def validate_output() -> None:
    import mujoco

    source_model = mujoco.MjModel.from_xml_path(str(SOURCE_XML))
    output_model = mujoco.MjModel.from_xml_path(str(OUTPUT_XML))
    comparisons = (
        ("body", mujoco.mjtObj.mjOBJ_BODY, source_model.nbody, output_model.nbody),
        ("joint", mujoco.mjtObj.mjOBJ_JOINT, source_model.njnt, output_model.njnt),
        ("equality", mujoco.mjtObj.mjOBJ_EQUALITY, source_model.neq, output_model.neq),
        ("tendon", mujoco.mjtObj.mjOBJ_TENDON, source_model.ntendon, output_model.ntendon),
        ("actuator", mujoco.mjtObj.mjOBJ_ACTUATOR, source_model.nu, output_model.nu),
        ("sensor", mujoco.mjtObj.mjOBJ_SENSOR, source_model.nsensor, output_model.nsensor),
    )
    for label, object_type, source_count, output_count in comparisons:
        if source_count != output_count:
            raise RuntimeError(f"{label} count changed: {source_count} -> {output_count}")
        if object_names(source_model, object_type, source_count) != object_names(output_model, object_type, output_count):
            raise RuntimeError(f"{label} names changed while adding RMUC terrain")

    ground_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
    terrain_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_GEOM, "rmuc_terrain")
    if ground_id < 0 or output_model.geom_type[ground_id] != mujoco.mjtGeom.mjGEOM_PLANE:
        raise RuntimeError("RMUC scene is missing its flat LQR projection plane")
    if terrain_id < 0 or output_model.geom_type[terrain_id] != mujoco.mjtGeom.mjGEOM_HFIELD:
        raise RuntimeError("RMUC scene is missing its hfield collision terrain")
    camera_id = mujoco.mj_name2id(output_model, mujoco.mjtObj.mjOBJ_CAMERA, "rmuc_follow_camera")
    if camera_id < 0:
        raise RuntimeError("RMUC follow camera is missing")
    if (
        output_model.nhfield != 1
        or output_model.hfield_nrow[0] != COLLISION_HFIELD_RESOLUTION[1]
        or output_model.hfield_ncol[0] != COLLISION_HFIELD_RESOLUTION[0]
    ):
        raise RuntimeError("RMUC collision hfield dimensions do not match the generated training grid")
    print(
        "validated rm_train_ground.xml: "
        f"bodies={output_model.nbody}, joints={output_model.njnt}, equalities={output_model.neq}, "
        f"tendons={output_model.ntendon}, actuators={output_model.nu}, sensors={output_model.nsensor}, "
        f"hfield={output_model.hfield_ncol[0]}x{output_model.hfield_nrow[0]}"
    )


def main() -> None:
    args = parse_args()
    stage_assets(args.asset_source)
    OUTPUT_XML.write_bytes(generate_xml())
    if not args.skip_validation:
        validate_output()
    print(f"generated: {OUTPUT_XML}")


if __name__ == "__main__":
    main()
