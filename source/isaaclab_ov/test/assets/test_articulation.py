# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real OVPhysX articulation coverage on one module-scoped scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import warp as wp

from pxr import Gf, Sdf, UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.test.utils.articulation_ordering import (  # noqa: E402
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
)

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"


@dataclass
class _ArticulationScene:
    """Articulations that share one real OVPhysX lifecycle."""

    sim: SimulationContext
    ordered: Articulation
    floating: Articulation
    tendon: Articulation
    native: Articulation | None
    device: str


def _sim_context(device: str, *, use_newton_actuators: bool):
    """Build one local OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            physics=OvPhysxCfg(),
            device=device,
            gravity=(0.0, 0.0, 0.0),
            use_newton_actuators=use_newton_actuators,
        ),
        auto_add_lighting=False,
    )


def _apply_api_schema(prim, schema_name: str) -> None:
    """Author an applied-schema token without loading the Kit PhysX schema module."""
    schemas = list(prim.GetAppliedSchemas())
    schemas.append(schema_name)
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _spawn_articulation(
    prim_path: str, *, fixed_base: bool, native_actuator: bool = False, spatial_tendon: bool = False
) -> Articulation:
    """Spawn one cached local branching articulation island."""
    actuator_cfg = (
        IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=5.0,
            damping=0.5,
            actuator_effort_limit=100.0,
        )
        if native_actuator
        else ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=5.0, damping=0.5)
    )
    articulation = Articulation(
        ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
            actuators={"joints": actuator_cfg},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    stage = sim_utils.get_current_stage()
    for joint_name in ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow"):
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(f"{prim_path}/{joint_name}"), "angular")
        drive.CreateStiffnessAttr(5.0)
        drive.CreateDampingAttr(0.5)
        drive.CreateMaxForceAttr(100.0)
    if fixed_base:
        fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{prim_path}/fixed_root")
        fixed_joint.GetBody1Rel().SetTargets([f"{prim_path}/base"])
    if spatial_tendon:
        root_prim = stage.GetPrimAtPath(f"{prim_path}/base")
        _apply_api_schema(root_prim, "PhysxTendonAttachmentRootAPI:root")
        root_prim.CreateAttribute("physxTendon:root:localPos", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0))
        root_prim.CreateAttribute("physxTendon:root:stiffness", Sdf.ValueTypeNames.Float).Set(5.0)
        root_prim.CreateAttribute("physxTendon:root:damping", Sdf.ValueTypeNames.Float).Set(0.5)
        root_prim.CreateAttribute("physxTendon:root:limitStiffness", Sdf.ValueTypeNames.Float).Set(1.0)
        root_prim.CreateAttribute("physxTendon:root:offset", Sdf.ValueTypeNames.Float).Set(0.0)

        leaf_prim = stage.GetPrimAtPath(f"{prim_path}/left_tip")
        _apply_api_schema(leaf_prim, "PhysxTendonAttachmentLeafAPI:leaf")
        leaf_prim.CreateAttribute("physxTendon:leaf:localPos", Sdf.ValueTypeNames.Point3f).Set(Gf.Vec3f(0.0))
        leaf_prim.CreateAttribute("physxTendon:leaf:parentAttachment", Sdf.ValueTypeNames.Token).Set("root")
        leaf_prim.CreateRelationship("physxTendon:leaf:parentLink").SetTargets([root_prim.GetPath()])
        leaf_prim.CreateAttribute("physxTendon:leaf:restLength", Sdf.ValueTypeNames.Float).Set(0.5)
        leaf_prim.CreateAttribute("physxTendon:leaf:lowerLimit", Sdf.ValueTypeNames.Float).Set(0.0)
        leaf_prim.CreateAttribute("physxTendon:leaf:upperLimit", Sdf.ValueTypeNames.Float).Set(2.0)
    return articulation


@pytest.fixture(scope="module")
def articulation_scene() -> _ArticulationScene:
    """Initialize every real OVPhysX articulation once for this module."""
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    native_enabled = device.startswith("cuda")
    with _sim_context(device, use_newton_actuators=native_enabled) as sim:
        ordered = _spawn_articulation("/World/Ordered", fixed_base=True)
        floating = _spawn_articulation("/World/Floating", fixed_base=False)
        tendon = _spawn_articulation("/World/Tendon", fixed_base=True, spatial_tendon=True)
        native = _spawn_articulation("/World/Native", fixed_base=True, native_actuator=True) if native_enabled else None
        sim.reset()
        yield _ArticulationScene(
            sim=sim,
            ordered=ordered,
            floating=floating,
            tendon=tendon,
            native=native,
            device=device,
        )


def test_articulation_initialization_and_partial_state(articulation_scene: _ArticulationScene) -> None:
    """Prove ordering and indexed state writes against the real OVPhysX view."""
    articulation = articulation_scene.ordered
    device = articulation_scene.device
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
    assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES
    assert articulation.joint_ordering is not None
    assert articulation.body_ordering is not None

    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=device)
    target_position = torch.tensor([[0.21, -0.13]], device=device)
    target_velocity = torch.tensor([[0.41, -0.23]], device=device)
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


def test_articulation_joint_and_body_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove selected joint and body properties reach real OVPhysX bindings."""
    articulation = articulation_scene.ordered
    device = articulation_scene.device
    joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32, device=device)
    backend_friction_before = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_FRICTION_PROPERTIES)).clone()
    static_friction = torch.tensor([[0.9, 0.7]], device=device)
    dynamic_friction = torch.tensor([[0.4, 0.3]], device=device)
    viscous_friction = torch.tensor([[0.11, 0.22]], device=device)
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=static_friction,
        joint_dynamic_friction_coeff=dynamic_friction,
        joint_viscous_friction_coeff=viscous_friction,
        joint_ids=joint_ids,
    )
    backend_joint_ids = torch.as_tensor(articulation.joint_ordering.user_to_backend_indices)[joint_ids.cpu()]
    expected_backend_friction = backend_friction_before.clone()
    expected_backend_friction[:, backend_joint_ids, 0] = static_friction.cpu()
    expected_backend_friction[:, backend_joint_ids, 1] = dynamic_friction.cpu()
    expected_backend_friction[:, backend_joint_ids, 2] = viscous_friction.cpu()
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_FRICTION_PROPERTIES)),
        expected_backend_friction,
    )

    body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32, device=device)
    backend_body_ids = torch.as_tensor(articulation.body_ordering.user_to_backend_indices)[body_ids.cpu()]
    raw_mass_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_MASS)).clone()
    masses = torch.tensor([[2.5, 3.5]], device=device)
    articulation.set_masses_index(masses=masses, body_ids=body_ids)
    expected_raw_mass = raw_mass_before.clone()
    expected_raw_mass[:, backend_body_ids] = masses.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_MASS)), expected_raw_mass)

    raw_com_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_COM_POSE)).clone()
    coms = articulation.data.body_com_pose_b.torch[:, body_ids].clone()
    coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03], device=device)
    coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02], device=device)
    articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), body_ids=body_ids)
    expected_raw_com = raw_com_before.clone()
    expected_raw_com[:, backend_body_ids] = coms.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_COM_POSE)), expected_raw_com)

    raw_inertia_before = wp.to_torch(articulation.root_view.get_attribute(TT.BODY_INERTIA)).clone()
    inertias = articulation.data.body_inertia.torch[:, body_ids].clone()
    inertias[0, 0, 0] *= 1.2
    inertias[0, 1, 4] *= 1.3
    articulation.set_inertias_index(inertias=inertias, body_ids=body_ids)
    expected_raw_inertia = raw_inertia_before.clone()
    expected_raw_inertia[:, backend_body_ids] = inertias.cpu()
    torch.testing.assert_close(wp.to_torch(articulation.root_view.get_attribute(TT.BODY_INERTIA)), expected_raw_inertia)
    torch.testing.assert_close(articulation.data.body_mass.torch[:, body_ids], masses)
    torch.testing.assert_close(articulation.data.body_com_pose_b.torch[:, body_ids], coms)
    torch.testing.assert_close(articulation.data.body_inertia.torch[:, body_ids], inertias)


def test_articulation_drive_and_dynamics(articulation_scene: _ArticulationScene) -> None:
    """Prove implicit drive delivery and live OVPhysX dynamics access."""
    articulation = articulation_scene.ordered
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(articulation.data.joint_vel.torch))
    initial_drive_position = articulation.data.joint_pos.torch[:, 0].clone()
    drive_target = articulation.data.joint_pos.torch.clone()
    drive_target[:, 0] += 0.4
    articulation.actuators.target_command.set_position_index(value=drive_target, full_data=True)
    articulation.write_data_to_sim()
    backend_target = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION_TARGET))
    backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    torch.testing.assert_close(backend_target, drive_target[:, backend_to_user])

    for _ in range(8):
        articulation_scene.sim.step()
        articulation.update(articulation_scene.sim.cfg.dt)
        articulation.write_data_to_sim()
    assert torch.any(torch.abs(articulation.data.joint_pos.torch[:, 0] - initial_drive_position) > 1e-6)
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
    """Prove floating-root state and a real OVPhysX external-wrench response."""
    articulation = articulation_scene.floating
    device = articulation_scene.device
    assert articulation.is_initialized
    assert not articulation.is_fixed_base
    assert articulation.data.root_link_pose_w.torch.shape == (1, 7)
    assert articulation.data.root_com_pose_w.torch.shape == (1, 7)
    assert articulation.data.body_link_pose_w.torch.shape == (1, articulation.num_bodies, 7)
    assert articulation.data.body_com_pose_w.torch.shape == (1, articulation.num_bodies, 7)
    initial_velocity = articulation.data.root_com_lin_vel_w.torch.clone()
    articulation.permanent_wrench_composer.set_forces_and_torques_index(
        forces=torch.tensor([[[8.0, 0.0, 0.0]]], device=device),
        torques=torch.zeros((1, 1, 3), device=device),
        env_ids=torch.tensor([0], dtype=torch.int32, device=device),
        body_ids=torch.tensor([0], dtype=torch.int32, device=device),
    )
    articulation.write_data_to_sim()
    articulation_scene.sim.step()
    articulation.update(articulation_scene.sim.cfg.dt)
    assert articulation.data.root_com_lin_vel_w.torch[0, 0] > initial_velocity[0, 0]


def test_spatial_tendon_properties_round_trip(articulation_scene: _ArticulationScene) -> None:
    """Prove a locally authored spatial tendon is discovered and writable."""
    articulation = articulation_scene.tendon
    device = articulation_scene.device
    assert articulation.is_initialized
    assert articulation.is_fixed_base
    assert articulation.num_spatial_tendons == 1
    stiffness = torch.tensor([[12.0]], device=device)
    damping = torch.tensor([[1.5]], device=device)
    limit_stiffness = torch.tensor([[3.0]], device=device)
    offset = torch.tensor([[0.1]], device=device)
    articulation.set_spatial_tendon_stiffness_index(stiffness=stiffness)
    articulation.set_spatial_tendon_damping_index(damping=damping)
    articulation.set_spatial_tendon_limit_stiffness_index(limit_stiffness=limit_stiffness)
    articulation.set_spatial_tendon_offset_index(offset=offset)
    torch.testing.assert_close(articulation.data.spatial_tendon_stiffness.torch, stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_damping.torch, damping)
    torch.testing.assert_close(articulation.data.spatial_tendon_limit_stiffness.torch, limit_stiffness)
    torch.testing.assert_close(articulation.data.spatial_tendon_offset.torch, offset)


def test_native_actuator_submits_real_effort(articulation_scene: _ArticulationScene) -> None:
    """Prove the native controller reaches the real OVPhysX effort binding."""
    articulation = articulation_scene.native
    if articulation is None:
        pytest.skip("Native actuator wheel probe requires CUDA")
    assert articulation._actuator_control.native_actuator_path_active
    assert articulation.newton_actuator_adapter is not None
    target = articulation.data.joint_pos.torch.clone() + 0.2
    initial_position = articulation.data.joint_pos.torch.clone()
    articulation.actuators.target_command.set_position_index(value=target)
    articulation.write_data_to_sim()

    raw_effort = wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d).clone()
    backend_effort = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_ACTUATION_FORCE))
    backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
    assert torch.any(raw_effort != 0.0)
    torch.testing.assert_close(backend_effort, raw_effort[:, backend_to_user])

    for _ in range(8):
        articulation_scene.sim.step()
        articulation.update(articulation_scene.sim.cfg.dt)
        articulation.write_data_to_sim()
    assert torch.any(articulation.data.joint_pos.torch != initial_position)
    recomputed_effort = wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d)
    assert torch.any(recomputed_effort != raw_effort)
    torch.testing.assert_close(
        wp.to_torch(articulation.root_view.get_attribute(TT.DOF_ACTUATION_FORCE)),
        recomputed_effort[:, backend_to_user],
    )
