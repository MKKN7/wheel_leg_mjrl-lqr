"""Regression contracts for the lower 30 x 9 mm guide-wheel training model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

from guide_wheel_mjcf import guide_wheel_mass_kg, guide_wheel_runtime_contract, load_guide_wheel_model
import lqr_deploy
from warp_env import load_warp_batch_config


ROOT = Path(__file__).resolve().parent
SCENES = (
    ROOT / "wheeled_infantry.xml",
    ROOT / "rm_train_ground.xml",
    ROOT / "official_standard_ground.xml",
    ROOT / "official_standard_warp_ground.xml",
)
BATCH_CONFIGS = (
    ROOT / "configs" / "warp_batch_preflight.yaml",
    ROOT / "configs" / "warp_flat_batch.yaml",
    ROOT / "configs" / "warp_rmuc_curriculum_batch.yaml",
    ROOT / "configs" / "warp_official_grade15_batch.yaml",
    ROOT / "configs" / "warp_official_standard_batch.yaml",
)


class LowerGuideWheelModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_guide_wheel_model()
        cls.contract = guide_wheel_runtime_contract()

    def test_only_the_twenty_four_lower_rollers_are_declared(self) -> None:
        self.assertEqual(len(self.config.wheels), 24)
        self.assertEqual(len(self.contract.contact_names), 24)
        self.assertEqual(len(self.contract.joint_names), 24)
        self.assertEqual(len(self.contract.left_indices), 12)
        self.assertEqual(len(self.contract.right_indices), 12)
        self.assertFalse(any("front_" in name or "rear_" in name for name in self.contract.contact_names))

    def test_source_model_reallocates_root_mass_without_changing_total(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "wheeled_infantry.xml"))
        guide_body_ids = []
        guide_joint_ids = []
        for contact_name, joint_name in zip(self.contract.contact_names, self.contract.joint_names):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, contact_name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.assertGreaterEqual(geom_id, 0)
            self.assertGreaterEqual(joint_id, 0)
            self.assertEqual(int(model.geom_bodyid[geom_id]), int(model.jnt_bodyid[joint_id]))
            guide_body_ids.append(int(model.geom_bodyid[geom_id]))
            guide_joint_ids.append(joint_id)
        self.assertEqual(len(set(guide_body_ids)), 24)
        self.assertEqual(len(set(guide_joint_ids)), 24)
        self.assertEqual(model.nq, 45)
        self.assertEqual(model.nv, 44)
        self.assertEqual(model.nu, self.config.expected_actuator_count)
        self.assertEqual(model.nsensordata, self.config.expected_sensor_data_count)
        self.assertFalse(set(guide_joint_ids) & set(np.asarray(model.actuator_trnid[:, 0], dtype=np.int64)))
        self.assertTrue(np.allclose(model.dof_armature[-24:], 0.0, rtol=0.0, atol=1.0e-12))
        self.assertTrue(np.allclose(model.dof_damping[-24:], self.config.joint_damping_nms, rtol=0.0, atol=1.0e-9))
        reallocated_mass = float(model.body_mass[1] + model.body_mass[guide_body_ids].sum())
        self.assertAlmostEqual(reallocated_mass, self.config.baseline_root_inertial.mass_kg, places=6)
        self.assertAlmostEqual(float(model.body_mass[guide_body_ids].mean()), guide_wheel_mass_kg(self.config), places=7)

    def test_every_derived_scene_has_the_same_lower_guide_contract(self) -> None:
        for path in SCENES:
            with self.subTest(scene=path.name):
                model = mujoco.MjModel.from_xml_path(str(path))
                self.assertEqual(model.nq, 45)
                self.assertEqual(model.nv, 44)
                self.assertEqual(model.nu, 6)
                self.assertEqual(model.nsensordata, 40)
                for name in self.contract.contact_names:
                    self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name), 0)
                for name in self.contract.joint_names:
                    self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name), 0)
                names = tuple(
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
                    for index in range(model.ngeom)
                )
                self.assertFalse(any("guide_wheel_front" in name or "guide_wheel_rear" in name for name in names))

    def test_lqr_excludes_only_the_passive_lower_guide_dofs(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(ROOT / "wheeled_infantry.xml"))
        refs = lqr_deploy.build_refs(model)
        excluded = lqr_deploy.passive_guide_dof_addresses(model, refs)
        retained, state_indices = lqr_deploy.lqr_state_indices(model, excluded)
        self.assertEqual(excluded.tolist(), list(range(20, 44)))
        self.assertEqual(retained.tolist(), list(range(20)))
        self.assertEqual(state_indices.size, 40)

    def test_batch_capacity_matches_the_rolling_model_dof_count(self) -> None:
        for path in BATCH_CONFIGS:
            with self.subTest(config=path.name):
                batch = load_warp_batch_config(path)
                model = mujoco.MjModel.from_xml_path(str(batch.xml_path))
                self.assertEqual(model.nv, 44)
                self.assertEqual(batch.capacity.nvmax, model.nv)

    def test_guides_are_allowed_on_supports_but_not_declared_obstacles(self) -> None:
        refs = SimpleNamespace(
            ground_geoms=(100,),
            obstacle_geoms=(200,),
            wheel_geoms=(1, 2),
            guide_wheel_geoms=(3, 4),
        )
        data = SimpleNamespace(
            ncon=2,
            contact=(
                SimpleNamespace(geom1=100, geom2=3),
                SimpleNamespace(geom1=200, geom2=3),
            ),
        )
        support, obstacle = lqr_deploy.nonwheel_static_contact_counts(data, refs)
        self.assertEqual(support, 0)
        self.assertEqual(obstacle, 1)


if __name__ == "__main__":
    unittest.main()
