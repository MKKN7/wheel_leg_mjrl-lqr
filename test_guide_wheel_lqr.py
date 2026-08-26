"""Regression coverage for passive 40 mm guide rollers in the CPU LQR model."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import mujoco
import numpy as np

import lqr_deploy as lqr


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "wheeled_infantry.xml"


class GuideWheelLqrTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        cls.refs = lqr.build_refs(cls.model)

    def test_passive_guide_package_is_complete_and_unactuated(self) -> None:
        self.assertEqual(len(self.refs.guide_wheel_geoms), 4)
        self.assertEqual(len(self.refs.guide_wheel_joints), 4)
        self.assertEqual(
            tuple(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                for geom_id in self.refs.guide_wheel_geoms
            ),
            lqr.GUIDE_WHEEL_CONTACT_NAMES,
        )
        self.assertEqual(
            tuple(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                for joint_id in self.refs.guide_wheel_joints
            ),
            lqr.GUIDE_WHEEL_JOINT_NAMES,
        )

    def test_passive_guide_dofs_are_excluded_from_lqr_but_not_physics(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        controller = lqr.settle_and_relinearize(
            self.model,
            data,
            self.refs,
            speed=0.0,
            acceleration_limit=lqr.DEFAULT_ACCELERATION_MPS2,
        )
        guide_dofs = lqr.passive_guide_dof_addresses(self.model, self.refs)
        expected_state_size = 2 * (self.model.nv - guide_dofs.size) + self.model.na
        self.assertEqual(controller.gain.shape, (self.model.nu, expected_state_size))
        self.assertEqual(controller.state_error(data).shape, (expected_state_size,))
        self.assertFalse(np.isin(guide_dofs, controller.lqr_dof_addresses).any())
        self.assertTrue(np.isfinite(controller.command(data)).all())

    def test_guides_are_safe_on_support_but_not_on_obstacles(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        controller = lqr.settle_and_relinearize(
            self.model,
            data,
            self.refs,
            speed=0.0,
            acceleration_limit=lqr.DEFAULT_ACCELERATION_MPS2,
        )
        data.qpos[:] = controller.qpos_equilibrium
        data.qvel[:] = 0.0
        root_qpos = int(self.model.jnt_qposadr[self.refs.root_joint])
        data.qpos[root_qpos + 2] -= 0.155
        mujoco.mj_forward(self.model, data)

        guide_geoms = set(self.refs.guide_wheel_geoms)
        self.assertTrue(
            any(
                contact.geom1 in guide_geoms or contact.geom2 in guide_geoms
                for contact in data.contact[: data.ncon]
            )
        )
        self.assertEqual(lqr.nonwheel_static_contact_counts(data, self.refs), (0, 0))

        obstacle_refs = replace(self.refs, obstacle_geoms=(self.refs.ground_geom,))
        support_contacts, obstacle_contacts = lqr.nonwheel_static_contact_counts(
            data,
            obstacle_refs,
        )
        self.assertEqual(support_contacts, 0)
        self.assertGreater(obstacle_contacts, 0)


if __name__ == "__main__":
    unittest.main()
