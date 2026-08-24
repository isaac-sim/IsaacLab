# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-Newton articulation coverage on one module-scoped scene."""

from dataclasses import dataclass

import pytest
import torch
from isaaclab_newton.assets import Articulation
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager
from newton import ModelFlags

from pxr import Gf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context

from source.isaaclab_newton.test.articulation_test_utils import author_fixed_spatial_chain

pytestmark = pytest.mark.integration


@dataclass
class _ArticulationScene:
    """Articulations that share one real Newton model and solver lifecycle."""

    sim: SimulationContext
    floating: Articulation
    fixed: Articulation
    device: str


def _newton_sim_context(*, device: str):
    """Create one kitless Newton simulation context."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            device=device,
            dt=1.0 / 120.0,
            gravity=(0.0, 0.0, 0.0),
            physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
        )
    )


def _author_two_link_articulations() -> Articulation:
    """Author two local one-DOF floating articulations."""
    link_cfg = sim_utils.CuboidCfg(
        size=(0.4, 0.1, 0.1),
        rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionBaseCfg(),
    )
    stage = sim_utils.get_current_stage()
    for env_index in range(2):
        env_path = f"/World/Env_{env_index}"
        robot_path = f"{env_path}/Robot"
        root_path = f"{robot_path}/Root"
        child_path = f"{robot_path}/Child"
        sim_utils.create_prim(env_path, "Xform", translation=(2.0 * env_index, 0.0, 0.0))
        sim_utils.create_prim(robot_path, "Xform")
        link_cfg.func(root_path, link_cfg, translation=(0.0, 0.0, 1.0))
        link_cfg.func(child_path, link_cfg, translation=(0.5, 0.0, 1.0))
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root_path))
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{robot_path}/Joint")
        joint.CreateBody0Rel().SetTargets([root_path])
        joint.CreateBody1Rel().SetTargets([child_path])
        joint.CreateAxisAttr().Set("Z")
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.25, 0.0, 0.0))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-0.25, 0.0, 0.0))
        joint.CreateLowerLimitAttr().Set(-90.0)
        joint.CreateUpperLimitAttr().Set(90.0)

    return Articulation(
        ArticulationCfg(
            prim_path="/World/Env_[^/]*/Robot",
            articulation_root_prim_path="/Root",
            actuators={
                "joint": ImplicitActuatorCfg(
                    joint_names_expr=["Joint"],
                    stiffness=20.0,
                    damping=2.0,
                )
            },
        )
    )


@pytest.fixture(scope="module")
def articulation_scene() -> _ArticulationScene:
    """Initialize every real Newton articulation once for this module."""
    device = "cpu"
    with _newton_sim_context(device=device) as sim:
        floating = _author_two_link_articulations()
        fixed = author_fixed_spatial_chain(
            prim_paths=("/World/Env_0/FixedRobot", "/World/Env_1/FixedRobot"),
            prim_path_expr="/World/Env_[^/]*/FixedRobot",
        )
        sim.reset()
        yield _ArticulationScene(sim=sim, floating=floating, fixed=fixed, device=device)


def test_articulation_initialization_and_partial_state(articulation_scene: _ArticulationScene) -> None:
    """Prove floating articulation discovery and isolated indexed state writes."""
    articulation = articulation_scene.floating
    device = articulation_scene.device
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.num_instances == 2
    assert articulation.num_bodies == 2
    assert articulation.num_joints == 1
    assert articulation.joint_names == ["Joint"]
    assert articulation.data.body_mass.shape == (2, 2)
    assert articulation.data.body_com_pos_b.shape == (2, 2)
    assert articulation.data.body_inertia.shape == (2, 2, 9)

    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([0], dtype=torch.int32, device=device)
    initial_root_pose = articulation.data.root_link_pose_w.torch.clone()
    initial_joint_pos = articulation.data.joint_pos.torch.clone()
    initial_joint_vel = articulation.data.joint_vel.torch.clone()
    target_root_pose = initial_root_pose[env_ids].clone()
    target_root_pose[:, :3] += torch.tensor([0.2, -0.1, 0.3], device=device)
    target_joint_pos = torch.tensor([[0.25]], device=device)
    target_joint_vel = torch.tensor([[-0.5]], device=device)
    articulation.write_root_link_pose_to_sim_index(root_pose=target_root_pose, env_ids=env_ids)
    articulation.write_joint_state_to_sim_index(
        position=target_joint_pos,
        velocity=target_joint_vel,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )

    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[env_ids], target_root_pose)
    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[:1], initial_root_pose[:1])
    torch.testing.assert_close(articulation.data.joint_pos.torch[env_ids], target_joint_pos)
    torch.testing.assert_close(articulation.data.joint_vel.torch[env_ids], target_joint_vel)
    torch.testing.assert_close(articulation.data.joint_pos.torch[:1], initial_joint_pos[:1])
    torch.testing.assert_close(articulation.data.joint_vel.torch[:1], initial_joint_vel[:1])


def test_articulation_model_properties_notify_newton(
    articulation_scene: _ArticulationScene, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prove partial inertial-property writes notify the live Newton model."""
    articulation = articulation_scene.floating
    device = articulation_scene.device
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    body_ids = torch.tensor([1], dtype=torch.int32, device=device)
    notifications = []
    add_model_change = SimulationManager.add_model_change

    def record_model_change(change: ModelFlags) -> None:
        notifications.append(change)
        add_model_change(change)

    monkeypatch.setattr(SimulationManager, "add_model_change", staticmethod(record_model_change))
    initial_mass = articulation.data.body_mass.torch.clone()
    masses = torch.tensor([[3.0]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_mass.torch[env_ids][:, body_ids], masses)
    torch.testing.assert_close(articulation.data.body_mass.torch[:1], initial_mass[:1])
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]

    notifications.clear()
    initial_com = articulation.data.body_com_pos_b.torch.clone()
    coms = torch.tensor([[[0.05, -0.02, 0.01]]], device=device)
    articulation.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch[env_ids][:, body_ids], coms)
    torch.testing.assert_close(articulation.data.body_com_pos_b.torch[:1], initial_com[:1])
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]

    notifications.clear()
    initial_inertia = articulation.data.body_inertia.torch.clone()
    inertias = torch.diag_embed(torch.tensor([[[2.0, 3.0, 4.0]]], device=device)).reshape(1, 1, 9)
    articulation.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_inertia.torch[env_ids][:, body_ids], inertias)
    torch.testing.assert_close(articulation.data.body_inertia.torch[:1], initial_inertia[:1])
    assert notifications == [ModelFlags.BODY_INERTIAL_PROPERTIES]


def test_articulation_dynamics_and_wrench_response(articulation_scene: _ArticulationScene) -> None:
    """Prove live floating-base dynamics data and isolated wrench delivery."""
    articulation = articulation_scene.floating
    device = articulation_scene.device
    jacobians = articulation.data.body_link_jacobian_w.torch
    mass_matrix = articulation.data.mass_matrix.torch
    assert jacobians.device.type == torch.device(device).type
    assert mass_matrix.device.type == torch.device(device).type
    assert jacobians.shape == (2, 2, 6, 7)
    assert mass_matrix.shape == (2, 7, 7)
    assert torch.isfinite(jacobians).all()
    assert torch.isfinite(mass_matrix).all()

    initial_velocity = articulation.data.root_com_lin_vel_w.torch.clone()
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=device),
        torques=torch.zeros((1, 1, 3), device=device),
        env_ids=torch.tensor([1], dtype=torch.int32, device=device),
        body_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    articulation.write_data_to_sim()
    articulation_scene.sim.step()
    articulation.update(articulation_scene.sim.cfg.dt)
    assert articulation.data.root_com_lin_vel_w.torch[1, 0] > initial_velocity[1, 0]
    torch.testing.assert_close(articulation.data.root_com_lin_vel_w.torch[0], initial_velocity[0], atol=1e-6, rtol=0)


def test_fixed_articulation_actuation_and_dynamics(articulation_scene: _ArticulationScene) -> None:
    """Prove fixed-root actuation, moving-link velocity, and dynamics data."""
    articulation = articulation_scene.fixed
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_instances == 2
    assert articulation.num_bodies == 7
    assert articulation.num_joints == 6
    initial_root_pose = articulation.data.root_link_pose_w.torch.clone()
    target = articulation.data.joint_pos.torch.clone()
    target[:, 0] = 0.05
    articulation.actuators.target_command.set_position_index(value=target)

    for _ in range(12):
        articulation.write_data_to_sim()
        articulation_scene.sim.step()
        articulation.update(articulation_scene.sim.cfg.dt)

    torch.testing.assert_close(articulation.data.root_link_pose_w.torch, initial_root_pose, atol=1e-6, rtol=0)
    assert torch.linalg.vector_norm(articulation.data.body_link_vel_w.torch[0, -1]) > 1e-3
    jacobians = articulation.data.body_link_jacobian_w.torch
    mass_matrix = articulation.data.mass_matrix.torch
    assert jacobians.shape == (2, 6, 6, 6)
    assert mass_matrix.shape == (2, 6, 6)
    assert torch.isfinite(jacobians).all()
    assert torch.isfinite(mass_matrix).all()
