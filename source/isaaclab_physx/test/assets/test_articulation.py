# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX articulation coverage on one module-scoped scene."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cpu").app

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import Articulation

from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext, build_simulation_context

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"
_NUM_ENVS = 2


@dataclass
class _ArticulationScene:
    """Articulations that share one real PhysX lifecycle."""

    sim: SimulationContext
    ordered: Articulation
    floating: Articulation
    tendon: Articulation
    device: str


def _spawn_articulation(name: str, *, y_offset: float, fixed_base: bool, spatial_tendon: bool = False) -> Articulation:
    """Spawn one two-environment branching-articulation island."""
    island_path = f"/World/{name}"
    sim_utils.create_prim(island_path, "Xform", translation=(0.0, y_offset, 0.0))
    for env_index in range(_NUM_ENVS):
        sim_utils.create_prim(f"{island_path}/Env_{env_index}", "Xform", translation=(3.0 * env_index, 0.0, 0.0))
    articulation = Articulation(
        ArticulationCfg(
            prim_path=f"{island_path}/Env_[^/]*/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
            actuators={
                "joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=5.0, damping=0.5),
            },
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    stage = sim_utils.get_current_stage()
    for env_index in range(_NUM_ENVS):
        prim_path = f"{island_path}/Env_{env_index}/Robot"
        for joint_name in ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"):
            drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{joint_name}"), "angular")
            drive.CreateStiffnessAttr(5.0)
            drive.CreateDampingAttr(0.5)
            drive.CreateMaxForceAttr(100.0)
        collision = UsdGeom.Cube.Define(stage, f"{prim_path}/base/collision")
        collision.CreateSizeAttr(0.1)
        UsdPhysics.CollisionAPI.Apply(collision.GetPrim())
        if fixed_base:
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed_root")
            fixed_joint.GetBody1Rel().SetTargets([f"{prim_path}/base"])
        if spatial_tendon:
            root_prim = stage.GetPrimAtPath(f"{prim_path}/base")
            root_attachment = PhysxSchema.PhysxTendonAttachmentAPI(root_prim, "root")
            root_attachment.CreateLocalPosAttr(Gf.Vec3f(0.0))
            root = PhysxSchema.PhysxTendonAttachmentRootAPI.Apply(root_prim, "root")
            root.CreateStiffnessAttr(5.0)
            root.CreateDampingAttr(0.5)
            root.CreateLimitStiffnessAttr(1.0)
            root.CreateOffsetAttr(0.0)

            leaf_prim = stage.GetPrimAtPath(f"{prim_path}/left_tip")
            leaf_attachment = PhysxSchema.PhysxTendonAttachmentAPI(leaf_prim, "leaf")
            leaf_attachment.CreateLocalPosAttr(Gf.Vec3f(0.0))
            leaf_attachment.CreateParentAttachmentAttr("root")
            leaf_attachment.CreateParentLinkRel().SetTargets([root_prim.GetPath()])
            leaf = PhysxSchema.PhysxTendonAttachmentLeafAPI.Apply(leaf_prim, "leaf")
            leaf.CreateRestLengthAttr(0.5)
            leaf.CreateLowerLimitAttr(0.0)
            leaf.CreateUpperLimitAttr(2.0)
    return articulation


@pytest.fixture(scope="module")
def articulation_scene() -> _ArticulationScene:
    """Initialize every real PhysX articulation once for this module."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    with build_simulation_context(device=device, gravity_enabled=False) as sim:
        ordered = _spawn_articulation("Ordered", y_offset=0.0, fixed_base=True)
        floating = _spawn_articulation("Floating", y_offset=3.0, fixed_base=False)
        tendon = _spawn_articulation("Tendon", y_offset=6.0, fixed_base=True, spatial_tendon=True)
        sim.reset()
        yield _ArticulationScene(sim=sim, ordered=ordered, floating=floating, tendon=tendon, device=device)


def test_articulation_initialization_and_partial_state(articulation_scene: _ArticulationScene) -> None:
    """Prove ordering and indexed state writes against the real PhysX view."""
    articulation = articulation_scene.ordered
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None
    assert articulation.num_instances == 2
    assert articulation.num_joints >= 2
    assert articulation.num_bodies >= 3

    env_ids = torch.tensor([1], dtype=torch.int32, device=articulation_scene.device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=articulation_scene.device)
    target_position = torch.tensor([[0.21, -0.13]], device=articulation_scene.device)
    target_velocity = torch.tensor([[0.41, -0.23]], device=articulation_scene.device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()
    expected_position[env_ids[:, None], joint_ids] = target_position
    expected_velocity[env_ids[:, None], joint_ids] = target_velocity
    articulation.write_joint_state_to_sim_index(
        position=target_position,
        velocity=target_velocity,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_positions()), expected_position[:, joint_backend_to_user]
    )


def test_articulation_joint_and_body_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove selected joint, body, and material properties round-trip through PhysX."""
    articulation = articulation_scene.ordered
    device = articulation_scene.device
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=device)
    backend_joint_ids = torch.as_tensor(articulation.joint_ordering.user_to_backend_indices)[joint_ids.cpu()]
    raw_friction_before = wp.to_torch(articulation.root_view.get_dof_friction_properties()).clone()
    static_friction = torch.tensor([[0.9, 0.7]], device=device)
    dynamic_friction = torch.tensor([[0.4, 0.3]], device=device)
    viscous_friction = torch.tensor([[0.11, 0.22]], device=device)
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static_friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    expected_raw_friction = raw_friction_before.clone()
    expected_raw_friction[1, backend_joint_ids, 0] = static_friction.cpu()
    expected_raw_friction[1, backend_joint_ids, 1] = dynamic_friction.cpu()
    expected_raw_friction[1, backend_joint_ids, 2] = viscous_friction.cpu()
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_friction_properties()),
        expected_raw_friction,
    )

    body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32, device=device)
    initial_mass = articulation.data.body_mass.torch.clone()
    masses = torch.tensor([[2.5, 3.5]], device=device)
    articulation.set_masses_index(masses=masses, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_mass.torch[env_ids][:, body_ids], masses)
    torch.testing.assert_close(articulation.data.body_mass.torch[:1], initial_mass[:1])

    initial_com = articulation.data.body_com_pose_b.torch.clone()
    coms = articulation.data.body_com_pose_b.torch[env_ids][:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03], device=device)
    coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02], device=device)
    articulation.set_coms_index(coms=coms, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch[env_ids][:, body_ids], coms)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch[:1], initial_com[:1])

    initial_inertia = articulation.data.body_inertia.torch.clone()
    inertias = articulation.data.body_inertia.torch[env_ids][:, body_ids].clone()
    inertias[0, 0, 0] *= 1.2
    inertias[0, 1, 4] *= 1.3
    articulation.set_inertias_index(inertias=inertias, env_ids=env_ids, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_inertia.torch[env_ids][:, body_ids], inertias)
    torch.testing.assert_close(articulation.data.body_inertia.torch[:1], initial_inertia[:1])

    body_backend_to_user = list(articulation.body_ordering.backend_to_user_indices)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_masses()).to(device),
        articulation.data.body_mass.torch[:, body_backend_to_user],
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_coms()).to(device),
        articulation.data.body_com_pose_b.torch[:, body_backend_to_user],
    )
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_inertias()).to(device),
        articulation.data.body_inertia.torch[:, body_backend_to_user],
    )

    materials_before = wp.to_torch(articulation.root_view.get_material_properties()).clone()
    materials = materials_before.clone()
    materials[1, :, 0] = 0.91
    materials[1, :, 1] = 0.17
    materials[1, :, 2] = 0.63
    articulation.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.array([1], dtype=wp.int32, device="cpu")
    )
    materials_after = wp.to_torch(articulation.root_view.get_material_properties())
    torch.testing.assert_close(materials_after[1:], materials[1:])
    torch.testing.assert_close(materials_after[:1], materials_before[:1])


def test_articulation_drive_and_dynamics(articulation_scene: _ArticulationScene) -> None:
    """Prove implicit drive delivery and live PhysX dynamics access."""
    articulation = articulation_scene.ordered
    device = articulation_scene.device
    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    joint_ids = torch.tensor([0], dtype=torch.int32, device=device)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(articulation.data.joint_vel.torch))
    initial_drive_position = articulation.data.joint_pos.torch[:, 0].clone()
    raw_target_before = wp.to_torch(articulation.root_view.get_dof_position_targets()).clone()
    drive_target = articulation.data.joint_pos.torch[env_ids][:, joint_ids].clone() + 0.4
    articulation.actuators.target_command.set_position_index(
        value=drive_target,
        env_ids=env_ids,
        joint_ids=joint_ids,
    )
    articulation.write_data_to_sim()
    raw_target_after = wp.to_torch(articulation.root_view.get_dof_position_targets())
    backend_joint_id = articulation.joint_ordering.user_to_backend_indices[0]
    torch.testing.assert_close(raw_target_after[1, backend_joint_id], drive_target[0, 0])
    torch.testing.assert_close(raw_target_after[0], raw_target_before[0])
    for _ in range(8):
        articulation.write_data_to_sim()
        articulation_scene.sim.step()
        articulation.update(articulation_scene.sim.cfg.dt)
    assert torch.abs(articulation.data.joint_pos.torch[1, 0] - initial_drive_position[1]) > 1e-6
    jacobian = articulation.data.body_link_jacobian_w.torch
    mass_matrix = articulation.data.mass_matrix.torch
    assert jacobian.shape == (_NUM_ENVS, articulation.num_bodies - 1, 6, articulation.num_joints)
    assert mass_matrix.shape == (_NUM_ENVS, articulation.num_joints, articulation.num_joints)
    assert jacobian.device.type == torch.device(articulation_scene.device).type
    assert mass_matrix.device.type == torch.device(articulation_scene.device).type
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(mass_matrix).all()
    torch.testing.assert_close(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-5, rtol=1e-5)


def test_floating_articulation_root_and_wrench_response(articulation_scene: _ArticulationScene) -> None:
    """Prove floating-root COM/link state and a real external-wrench response."""
    articulation = articulation_scene.floating
    device = articulation_scene.device
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.data.root_link_pose_w.torch.shape == (_NUM_ENVS, 7)
    assert articulation.data.root_com_pose_w.torch.shape == (_NUM_ENVS, 7)
    assert articulation.data.body_link_pose_w.torch.shape == (_NUM_ENVS, articulation.num_bodies, 7)
    assert articulation.data.body_com_pose_w.torch.shape == (_NUM_ENVS, articulation.num_bodies, 7)
    assert torch.isfinite(articulation.data.root_com_pose_w.torch).all()

    env_ids = torch.tensor([1], dtype=torch.int32, device=device)
    initial_pose = articulation.data.root_link_pose_w.torch.clone()
    target_pose = initial_pose[env_ids].clone()
    target_pose[:, :3] += torch.tensor([0.2, -0.1, 0.3], device=device)
    articulation.write_root_link_pose_to_sim_index(root_pose=target_pose, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[env_ids], target_pose)
    torch.testing.assert_close(articulation.data.root_link_pose_w.torch[:1], initial_pose[:1])

    initial_velocity = articulation.data.root_com_lin_vel_w.torch.clone()
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=device),
        torques=torch.zeros((1, 1, 3), device=device),
        env_ids=env_ids,
        body_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    articulation.write_data_to_sim()
    articulation_scene.sim.step()
    articulation.update(articulation_scene.sim.cfg.dt)
    assert articulation.data.root_com_lin_vel_w.torch[1, 0] > initial_velocity[1, 0]
    torch.testing.assert_close(articulation.data.root_com_lin_vel_w.torch[0], initial_velocity[0], atol=1e-6, rtol=0)


def test_spatial_tendon_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove a locally authored spatial tendon is discovered and writable."""
    articulation = articulation_scene.tendon
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_instances == _NUM_ENVS
    assert articulation.num_spatial_tendons == 1
    env_ids = torch.tensor([1], dtype=torch.int32, device=articulation_scene.device)
    initial_stiffness = articulation.data.spatial_tendon_stiffness.torch.clone()
    stiffness = torch.tensor([[12.0]], device=articulation_scene.device)
    damping = torch.tensor([[1.5]], device=articulation_scene.device)
    limit_stiffness = torch.tensor([[3.0]], device=articulation_scene.device)
    offset = torch.tensor([[0.1]], device=articulation_scene.device)
    articulation.set_spatial_tendon_stiffness_index(stiffness=stiffness, env_ids=env_ids)
    articulation.set_spatial_tendon_damping_index(damping=damping, env_ids=env_ids)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=limit_stiffness, env_ids=env_ids)
    articulation.set_spatial_tendon_offset_index(offset=offset, env_ids=env_ids)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch[env_ids], stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch[:1], initial_stiffness[:1])
    torch.testing.assert_close(articulation.data.spatial_tendon_damping.torch[env_ids], damping)
    torch.testing.assert_close(articulation.data.spatial_tendon_limit_stiffness.torch[env_ids], limit_stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_offset.torch[env_ids], offset)
