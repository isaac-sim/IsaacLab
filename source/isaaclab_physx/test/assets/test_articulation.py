# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-PhysX integration coverage for articulations."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"


def _spawn_ordered_articulation() -> Articulation:
    """Spawn the cached local branching fixture with nonidentity public axes."""
    articulation = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
            actuators={},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    UsdPhysics.FixedJoint.Define(sim_utils.get_current_stage(), "/World/Robot/fixed_root").GetBody1Rel().SetTargets(
        ["/World/Robot/base"]
    )
    return articulation


def test_articulation_real_physx_seams() -> None:
    """Prove ordered joint state, model-property writes, Jacobian, and mass access."""
    with build_simulation_context(device="cpu", gravity_enabled=False) as sim:
        articulation = _spawn_ordered_articulation()
        sim.reset()

        assert articulation.is_initialized
        assert articulation.is_fixed_base
        assert articulation.joint_ordering is not None
        assert articulation.body_ordering is not None
        assert articulation.num_instances == 1
        assert articulation.num_joints >= 2
        assert articulation.num_bodies >= 3

        joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32)
        target_position = torch.tensor([[0.21, -0.13]])
        target_velocity = torch.tensor([[0.41, -0.23]])
        expected_position = articulation.data.joint_pos.torch.clone()
        expected_velocity = articulation.data.joint_vel.torch.clone()
        expected_position[:, joint_ids] = target_position
        expected_velocity[:, joint_ids] = target_velocity
        articulation.write_joint_state_to_sim_index(
            position=target_position, velocity=target_velocity, joint_ids=joint_ids
        )
        torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
        torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)
        joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
        torch.testing.assert_close(
            wp.to_torch(articulation.root_view.get_dof_positions()), expected_position[:, joint_backend_to_user]
        )

        body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32)
        masses = torch.tensor([[2.5, 3.5]])
        articulation.set_masses_index(masses=masses, body_ids=body_ids)
        torch.testing.assert_close(articulation.data.body_mass.torch[:, body_ids], masses)

        coms = articulation.data.body_com_pose_b.torch[:, body_ids].clone()
        coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03])
        coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02])
        articulation.set_coms_index(coms=coms, body_ids=body_ids)
        torch.testing.assert_close(articulation.data.body_com_pose_b.torch[:, body_ids], coms)

        inertias = articulation.data.body_inertia.torch[:, body_ids].clone()
        inertias[0, 0, 0] *= 1.2
        inertias[0, 1, 4] *= 1.3
        articulation.set_inertias_index(inertias=inertias, body_ids=body_ids)
        torch.testing.assert_close(articulation.data.body_inertia.torch[:, body_ids], inertias)

        body_backend_to_user = list(articulation.body_ordering.backend_to_user_indices)
        torch.testing.assert_close(
            wp.to_torch(articulation.root_view.get_masses()),
            articulation.data.body_mass.torch[:, body_backend_to_user],
        )
        torch.testing.assert_close(
            wp.to_torch(articulation.root_view.get_coms()),
            articulation.data.body_com_pose_b.torch[:, body_backend_to_user],
        )
        torch.testing.assert_close(
            wp.to_torch(articulation.root_view.get_inertias()),
            articulation.data.body_inertia.torch[:, body_backend_to_user],
        )

        sim.step()
        articulation.update(sim.cfg.dt)
        jacobian = articulation.data.body_link_jacobian_w.torch
        mass_matrix = articulation.data.mass_matrix.torch
        assert jacobian.shape == (
            1,
            articulation.num_bodies - 1,
            6,
            articulation.num_joints,
        )
        assert mass_matrix.shape == (1, articulation.num_joints, articulation.num_joints)
        assert torch.isfinite(jacobian).all()
        assert torch.isfinite(mass_matrix).all()
        torch.testing.assert_close(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_articulation_cuda_dynamics_access() -> None:
    """Smoke-test the distinct CUDA-backed Jacobian and mass-matrix path."""
    with build_simulation_context(device="cuda:0", gravity_enabled=False) as sim:
        articulation = _spawn_ordered_articulation()
        sim.reset()
        sim.step()
        articulation.update(sim.cfg.dt)

        assert articulation.data.body_link_jacobian_w.torch.device.type == "cuda"
        assert articulation.data.mass_matrix.torch.device.type == "cuda"
        assert torch.isfinite(articulation.data.body_link_jacobian_w.torch).all()
        assert torch.isfinite(articulation.data.mass_matrix.torch).all()
