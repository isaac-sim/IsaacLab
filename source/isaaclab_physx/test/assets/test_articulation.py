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
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import SimulationContext, build_simulation_context

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"


@dataclass
class _ArticulationScene:
    """Articulations that share one real PhysX lifecycle."""

    sim: SimulationContext
    ordered: Articulation
    floating: Articulation
    tendon: Articulation
    device: str


def _spawn_articulation(prim_path: str, *, fixed_base: bool, spatial_tendon: bool = False) -> Articulation:
    """Spawn one local branching articulation island."""
    articulation = Articulation(
        ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
            actuators={},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    stage = sim_utils.get_current_stage()
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
        ordered = _spawn_articulation("/World/Ordered", fixed_base=True)
        floating = _spawn_articulation("/World/Floating", fixed_base=False)
        tendon = _spawn_articulation("/World/Tendon", fixed_base=True, spatial_tendon=True)
        sim.reset()
        yield _ArticulationScene(sim=sim, ordered=ordered, floating=floating, tendon=tendon, device=device)


def test_articulation_initialization_and_partial_state(articulation_scene: _ArticulationScene) -> None:
    """Prove ordering and indexed state writes against the real PhysX view."""
    articulation = articulation_scene.ordered
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None
    assert articulation.num_instances == 1
    assert articulation.num_joints >= 2
    assert articulation.num_bodies >= 3

    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=articulation_scene.device)
    target_position = torch.tensor([[0.21, -0.13]], device=articulation_scene.device)
    target_velocity = torch.tensor([[0.41, -0.23]], device=articulation_scene.device)
    expected_position = articulation.data.joint_pos.torch.clone()
    expected_velocity = articulation.data.joint_vel.torch.clone()
    expected_position[:, joint_ids] = target_position
    expected_velocity[:, joint_ids] = target_velocity
    articulation.write_joint_state_to_sim_index(
        position=target_position,
        velocity=target_velocity,
        joint_ids=joint_ids,
    )
    torch.testing.assert_close(articulation.data.joint_pos.torch, expected_position)
    torch.testing.assert_close(articulation.data.joint_vel.torch, expected_velocity)
    joint_backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_dof_positions()), expected_position[:, joint_backend_to_user]
    )


def test_articulation_model_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove selected body and material properties round-trip through PhysX."""
    articulation = articulation_scene.ordered
    device = articulation_scene.device
    body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32, device=device)
    masses = torch.tensor([[2.5, 3.5]], device=device)
    articulation.set_masses_index(masses=masses, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_mass.torch[:, body_ids], masses)

    coms = articulation.data.body_com_pose_b.torch[:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03], device=device)
    coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02], device=device)
    articulation.set_coms_index(coms=coms, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch[:, body_ids], coms)

    inertias = articulation.data.body_inertia.torch[:, body_ids].clone()
    inertias[0, 0, 0] *= 1.2
    inertias[0, 1, 4] *= 1.3
    articulation.set_inertias_index(inertias=inertias, body_ids=body_ids)
    torch.testing.assert_close(articulation.data.body_inertia.torch[:, body_ids], inertias)

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

    materials = torch.empty((1, articulation.root_view.max_shapes, 3))
    materials[..., 0] = 0.91
    materials[..., 1] = 0.17
    materials[..., 2] = 0.63
    articulation.root_view.set_material_properties(
        wp.from_torch(materials, dtype=wp.float32), wp.array([0], dtype=wp.int32, device="cpu")
    )
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_material_properties()), materials)


def test_articulation_dynamics_are_finite_and_fresh(articulation_scene: _ArticulationScene) -> None:
    """Prove live Jacobian and mass data remain valid after a simulation step."""
    articulation = articulation_scene.ordered
    articulation_scene.sim.step()
    articulation.update(articulation_scene.sim.cfg.dt)
    jacobian = articulation.data.body_link_jacobian_w.torch
    mass_matrix = articulation.data.mass_matrix.torch
    assert jacobian.shape == (1, articulation.num_bodies - 1, 6, articulation.num_joints)
    assert mass_matrix.shape == (1, articulation.num_joints, articulation.num_joints)
    assert jacobian.device.type == torch.device(articulation_scene.device).type
    assert mass_matrix.device.type == torch.device(articulation_scene.device).type
    assert torch.isfinite(jacobian).all()
    assert torch.isfinite(mass_matrix).all()
    torch.testing.assert_close(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-5, rtol=1e-5)


def test_floating_articulation_root_and_wrench_response(articulation_scene: _ArticulationScene) -> None:
    """Prove floating-root COM/link state and a real external-wrench response."""
    articulation = articulation_scene.floating
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.data.root_link_pose_w.torch.shape == (1, 7)
    assert articulation.data.root_com_pose_w.torch.shape == (1, 7)
    assert articulation.data.body_link_pose_w.torch.shape == (1, articulation.num_bodies, 7)
    assert articulation.data.body_com_pose_w.torch.shape == (1, articulation.num_bodies, 7)
    assert torch.isfinite(articulation.data.root_com_pose_w.torch).all()

    initial_velocity = articulation.data.root_com_lin_vel_w.torch.clone()
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=articulation_scene.device),
        torques=torch.zeros((1, 1, 3), device=articulation_scene.device),
        env_ids=torch.tensor([0], dtype=torch.int32, device=articulation_scene.device),
        body_ids=torch.tensor([0], dtype=torch.int32, device=articulation_scene.device),
    )
    articulation.write_data_to_sim()
    articulation_scene.sim.step()
    articulation.update(articulation_scene.sim.cfg.dt)
    assert articulation.data.root_com_lin_vel_w.torch[0, 0] > initial_velocity[0, 0]


def test_spatial_tendon_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove a locally authored spatial tendon is discovered and writable."""
    articulation = articulation_scene.tendon
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_spatial_tendons == 1
    stiffness = torch.tensor([[12.0]], device=articulation_scene.device)
    damping = torch.tensor([[1.5]], device=articulation_scene.device)
    limit_stiffness = torch.tensor([[3.0]], device=articulation_scene.device)
    offset = torch.tensor([[0.1]], device=articulation_scene.device)
    articulation.set_spatial_tendon_stiffness_index(stiffness=stiffness)
    articulation.set_spatial_tendon_damping_index(damping=damping)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=limit_stiffness)
    articulation.set_spatial_tendon_offset_index(offset=offset)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch, stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_damping.torch, damping)
    torch.testing.assert_close(articulation.data.spatial_tendon_limit_stiffness.torch, limit_stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_offset.torch, offset)
