"""Regression contracts for the official MuJoCo-Warp scene variant."""

from __future__ import annotations

import os
from pathlib import Path
import unittest
import warnings

import mujoco

from build_official_standard_ground import (
    _assert_dynamic_model_parity,
    generate_warp_xml,
    validate_official_warp_scene,
)


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_SCENE = ROOT / "official_standard_ground.xml"
WARP_SCENE = ROOT / "official_standard_warp_ground.xml"
GEOMETRY = ROOT / "official_terrain_geometry.yaml"
_BATCH_SIZES = {
    "body_mass": 2,
    "body_inertia": 2,
    "body_subtreemass": 2,
    "body_invweight0": 2,
    "dof_damping": 2,
    "dof_invweight0": 2,
    "geom_friction": 2,
    "actuator_gainprm": 2,
    "actuator_biasprm": 2,
    "actuator_acc0": 2,
}


class OfficialWarpSceneContractTest(unittest.TestCase):
    def test_generated_variant_matches_declared_whitelist(self) -> None:
        generated, _ = generate_warp_xml(GEOMETRY)
        self.assertEqual(generated, WARP_SCENE.read_bytes())

        report = validate_official_warp_scene(WARP_SCENE, geometry_path=GEOMETRY)
        self.assertTrue(report["static_geometry_preserved"])
        self.assertEqual(
            report["zero_margin_geoms"],
            (
                "base_collision",
                "obstacle_doghole_roof",
                "obstacle_doghole_left_wall",
                "obstacle_doghole_right_wall",
            ),
        )
        self.assertEqual(report["disabled_collision_geoms"], ("ground",))

        canonical = mujoco.MjModel.from_xml_path(str(CANONICAL_SCENE))
        variant = mujoco.MjModel.from_xml_path(str(WARP_SCENE))
        for geom_name in report["zero_margin_geoms"]:
            index = mujoco.mj_name2id(canonical, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            self.assertGreater(canonical.geom_margin[index], 0.0)
            self.assertEqual(variant.geom_margin[index], 0.0)
        ground_index = mujoco.mj_name2id(canonical, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self.assertGreater(canonical.geom_contype[ground_index], 0)
        self.assertGreater(canonical.geom_conaffinity[ground_index], 0)
        self.assertEqual(variant.geom_contype[ground_index], 0)
        self.assertEqual(variant.geom_conaffinity[ground_index], 0)

    def test_validator_rejects_the_canonical_filename(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be named"):
            validate_official_warp_scene(CANONICAL_SCENE, geometry_path=GEOMETRY)

    def test_dynamic_model_guard_rejects_actuator_change(self) -> None:
        canonical = mujoco.MjModel.from_xml_path(str(CANONICAL_SCENE))
        variant = mujoco.MjModel.from_xml_path(str(WARP_SCENE))
        variant.actuator_gear[2, 0] = 15.7
        with self.assertRaisesRegex(RuntimeError, "actuator_gear"):
            _assert_dynamic_model_parity(canonical, variant)


@unittest.skipUnless(
    os.environ.get("WARP_OFFICIAL_SCENE_CUDA_TEST") == "1",
    "set WARP_OFFICIAL_SCENE_CUDA_TEST=1 for the official MuJoCo-Warp scene test",
)
class OfficialWarpSceneCudaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import mujoco_warp
        import warp

        cls.mujoco_warp = mujoco_warp
        cls.warp = warp
        warp.init()

    def test_canonical_scene_stays_fail_closed_and_variant_is_accepted(self) -> None:
        canonical = mujoco.MjModel.from_xml_path(str(CANONICAL_SCENE))
        with self.assertRaisesRegex(NotImplementedError, "obstacle_doghole_roof, base_collision"):
            self.mujoco_warp.put_model(canonical, batch_sizes=_BATCH_SIZES)

        variant = mujoco.MjModel.from_xml_path(str(WARP_SCENE))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.mujoco_warp.put_model(variant, batch_sizes=_BATCH_SIZES)
        self.assertTrue(
            any("CYLINDER', 'BOX" in str(item.message) for item in caught),
            "the known one-contact Warp limitation must remain visible to the dynamic parity gate",
        )


if __name__ == "__main__":
    unittest.main()
