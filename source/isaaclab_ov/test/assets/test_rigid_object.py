# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-OVPhysX integration coverage for rigid objects."""

from __future__ import annotations

import pytest
import torch
import warp as wp

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import RigidObject  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402

pytestmark = pytest.mark.integration


def _sim_context():
    """Build a local CPU OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(physics=OvPhysxCfg(), device="cpu", gravity=(0.0, 0.0, 0.0)),
        auto_add_lighting=False,
    )


def _spawn_rigid_objects() -> RigidObject:
    """Author two local cuboids for partial-write and wrench proofs."""
    for index in range(2):
        sim_utils.create_prim(f"/World/Env_{index}", "Xform", translation=(2.0 * index, 0.0, 0.0))
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/Env_[^/]*/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


def test_rigid_object_real_ovphysx_seams() -> None:
    """Prove partial state, inertial properties, and one real wrench delivery."""
    with _sim_context() as sim:
        rigid_object = _spawn_rigid_objects()
        sim.reset()

        assert rigid_object.is_initialized
        assert rigid_object.num_instances == 2
        assert rigid_object.data.body_mass.torch.shape == (2, 1)
        assert rigid_object.data.body_com_pose_b.torch.shape == (2, 1, 7)
        assert rigid_object.data.body_inertia.torch.shape == (2, 1, 9)

        initial_pose = rigid_object.data.root_link_pose_w.torch.clone()
        target_pose = initial_pose[1:2].clone()
        target_pose[0, :3] += torch.tensor([0.25, -0.1, 0.2])
        rigid_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=[1])
        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[1:2], target_pose)
        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[0:1], initial_pose[0:1])

        raw_mass_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_MASS)).clone()
        rigid_object.set_masses_index(masses=wp.array([[3.0]], dtype=wp.float32, device="cpu"), env_ids=[1])
        expected_raw_mass = raw_mass_before.clone()
        expected_raw_mass[1] = 3.0
        torch.testing.assert_close(
            wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_MASS)), expected_raw_mass
        )

        raw_com_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_COM_POSE)).clone()
        coms = rigid_object.data.body_com_pose_b.torch[1:2].clone()
        coms[0, 0, :3] = torch.tensor([0.01, -0.02, 0.03])
        rigid_object.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), env_ids=[1])
        expected_raw_com = raw_com_before.clone()
        expected_raw_com[1] = coms[0, 0]
        torch.testing.assert_close(
            wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_COM_POSE)), expected_raw_com
        )

        raw_inertia_before = wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_INERTIA)).clone()
        inertias = rigid_object.data.body_inertia.torch[1:2].clone()
        inertias[0, 0, 0] *= 1.5
        rigid_object.set_inertias_index(inertias=wp.from_torch(inertias, dtype=wp.float32), env_ids=[1])
        expected_raw_inertia = raw_inertia_before.clone()
        expected_raw_inertia[1] = inertias[0, 0]
        torch.testing.assert_close(
            wp.to_torch(rigid_object.root_view.get_attribute(TT.RIGID_BODY_INERTIA)), expected_raw_inertia
        )
        torch.testing.assert_close(rigid_object.data.body_mass.torch[:, 0], torch.tensor([1.0, 3.0]))
        torch.testing.assert_close(rigid_object.data.body_com_pose_b.torch[1:2], coms)
        torch.testing.assert_close(rigid_object.data.body_inertia.torch[1:2], inertias)

        initial_velocity = rigid_object.data.root_com_vel_w.torch.clone()
        forces = torch.zeros((2, 1, 3))
        forces[1, 0, 0] = 20.0
        rigid_object.instantaneous_wrench_composer.set_forces_and_torques_index(forces=forces, is_global=True)
        rigid_object.write_data_to_sim()
        sim.step()
        rigid_object.update(sim.cfg.dt)

        assert rigid_object.data.root_com_vel_w.torch[1, 0] > initial_velocity[1, 0]
        torch.testing.assert_close(rigid_object.data.root_com_vel_w.torch[0], initial_velocity[0], atol=1e-6, rtol=0.0)
