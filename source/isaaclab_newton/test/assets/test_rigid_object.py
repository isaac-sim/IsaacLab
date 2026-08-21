# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-solver integration tests for Newton rigid objects."""

import pytest
import torch
import warp as wp
from isaaclab_newton.assets import RigidObject
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg, build_simulation_context

pytestmark = pytest.mark.integration

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
    ),
]


def _newton_sim_context(device: str):
    """Create a fresh kitless Newton simulation context."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            device=device,
            dt=1.0 / 60.0,
            gravity=(0.0, 0.0, 0.0),
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()),
        )
    )


def _spawn_cubes() -> RigidObject:
    """Author two local dynamic cuboids and return their Newton asset."""
    for env_index in range(2):
        sim_utils.create_prim(f"/World/Env_{env_index}", "Xform", translation=(2.0 * env_index, 0.0, 0.0))
    return RigidObject(
        RigidObjectCfg(
            prim_path="/World/Env_[^/]*/Object",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.2),
                rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionBaseCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
        )
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_rigid_object_real_newton_seams(device: str) -> None:
    """Exercise local initialization, partial state/property writes, gravity, and wrench delivery."""
    with _newton_sim_context(device) as sim:
        rigid_object = _spawn_cubes()
        sim.reset()

        assert rigid_object.is_initialized
        assert rigid_object.num_instances == 2
        assert rigid_object.body_names == ["Object"]
        assert rigid_object.data.body_mass.shape == (2, 1)
        assert rigid_object.data.body_com_pos_b.shape == (2, 1)
        assert rigid_object.data.body_inertia.shape == (2, 1, 9)

        initial_pose = rigid_object.data.root_link_pose_w.torch.clone()
        env_ids = torch.tensor([1], dtype=torch.int32, device=device)
        body_ids = torch.tensor([0], dtype=torch.int32, device=device)
        target_pose = initial_pose[env_ids].clone()
        target_pose[..., :3] += torch.tensor([0.5, -0.25, 0.75], device=device)
        rigid_object.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)

        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[env_ids], target_pose)
        torch.testing.assert_close(rigid_object.data.root_link_pose_w.torch[:1], initial_pose[:1])

        masses = torch.tensor([[4.0]], device=device)
        rigid_object.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
        torch.testing.assert_close(rigid_object.data.body_mass.torch[env_ids][:, body_ids], masses)
        torch.testing.assert_close(
            wp.to_torch(rigid_object.data._sim_bind_body_inv_mass)[env_ids][:, body_ids], masses.reciprocal()
        )

        model = SimulationManager.get_model()
        model_gravity = model.gravity[: model.world_count]
        new_gravity = torch.tensor([[0.0, 0.0, -2.0], [0.0, -3.0, -4.0]], device=device)
        wp.to_torch(model_gravity).copy_(new_gravity)
        SimulationManager.add_model_change(ModelFlags.MODEL_PROPERTIES)
        rigid_object.update(0.0)
        torch.testing.assert_close(rigid_object.data.GRAVITY_VEC_W.torch, new_gravity)
        torch.testing.assert_close(
            rigid_object.data.projected_gravity_b.torch,
            torch.nn.functional.normalize(new_gravity, dim=-1),
            atol=1e-6,
            rtol=1e-6,
        )

        initial_velocity = rigid_object.data.root_com_lin_vel_w.torch.clone()
        forces = torch.tensor([[[6.0, 0.0, 0.0]]], device=device)
        torques = torch.zeros_like(forces)
        rigid_object.permanent_wrench_composer.set_forces_and_torques_index(
            forces=forces,
            torques=torques,
            env_ids=torch.tensor([0], dtype=torch.int32, device=device),
            body_ids=body_ids,
        )
        rigid_object.write_data_to_sim()
        sim.step()
        rigid_object.update(sim.cfg.dt)

        assert rigid_object.data.root_com_lin_vel_w.torch[0, 0] > initial_velocity[0, 0]
