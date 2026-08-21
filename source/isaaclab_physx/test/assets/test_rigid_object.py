# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-PhysX integration coverage for rigid objects."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest
import torch
from isaaclab_physx.assets import RigidObject

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration


def _spawn_rigid_objects() -> RigidObject:
    """Author two local cuboids without Nucleus dependencies."""
    for env_index in range(2):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(2.0 * env_index, 0.0, 0.0))
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/Env_[^/]*/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


def test_rigid_object_real_physx_seams() -> None:
    """Prove partial state/inertial writes and one real external-wrench delivery."""
    with build_simulation_context(device="cpu", gravity_enabled=False) as sim:
        rigid_object = _spawn_rigid_objects()
        sim.reset()

        assert rigid_object.is_initialized
        assert rigid_object.num_instances == 2
        assert rigid_object.data.body_mass.torch.shape == (2, 1)
        assert rigid_object.data.body_com_pose_b.torch.shape == (2, 1, 7)
        assert rigid_object.data.body_inertia.torch.shape == (2, 1, 9)

        env_ids = torch.tensor([1], dtype=torch.int32)
        body_ids = torch.tensor([0], dtype=torch.int32)
        initial_pose = rigid_object.data.root_link_pose_w.torch.clone()
        target_pose = initial_pose[env_ids].clone()
        target_pose[:, :3] += torch.tensor([0.25, -0.1, 0.3])
        target_velocity = torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.1]])
        rigid_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
        rigid_object.write_root_link_velocity_to_sim_index(root_velocity=target_velocity, env_ids=env_ids)
        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[env_ids], target_pose)
        torch.testing.assert_close(rigid_object.data.root_link_vel_w.torch[env_ids], target_velocity)
        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[:1], initial_pose[:1])

        masses = torch.tensor([[3.0]])
        rigid_object.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(rigid_object.data.body_mass.torch[env_ids][:, body_ids], masses)

        coms = rigid_object.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
        coms[..., :3] = torch.tensor([[[0.03, -0.02, 0.01]]])
        rigid_object.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(rigid_object.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)

        inertias = rigid_object.data.body_inertia.torch[env_ids][:, body_ids].clone()
        inertias[..., 0] *= 1.2
        inertias[..., 4] *= 1.3
        inertias[..., 8] *= 1.4
        rigid_object.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(rigid_object.data.body_inertia.torch[env_ids][:, body_ids], inertias)

        rigid_object.write_root_link_velocity_to_sim_index(
            root_velocity=torch.zeros((2, 6)), env_ids=torch.tensor([0, 1], dtype=torch.int32)
        )
        initial_velocity = rigid_object.data.root_com_lin_vel_w.torch.clone()
        rigid_object.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.tensor([[[6.0, 0.0, 0.0]]]),
            torques=torch.zeros((1, 1, 3)),
            env_ids=torch.tensor([0], dtype=torch.int32),
            body_ids=body_ids,
        )
        rigid_object.write_data_to_sim()
        sim.step()
        rigid_object.update(sim.cfg.dt)

        assert rigid_object.data.root_com_lin_vel_w.torch[0, 0] > initial_velocity[0, 0]
        torch.testing.assert_close(
            rigid_object.data.root_com_lin_vel_w.torch[1], initial_velocity[1], atol=1e-5, rtol=0
        )
