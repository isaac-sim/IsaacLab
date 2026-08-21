# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal real-OVPhysX integration coverage for articulations."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import warp as wp

from pxr import UsdPhysics

pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov import tensor_types as TT  # noqa: E402
from isaaclab_ov.assets import Articulation  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, build_simulation_context  # noqa: E402
from isaaclab.test.utils.articulation_ordering import (  # noqa: E402
    BRANCHING_MJWARP_BODY_NAMES,
    BRANCHING_MJWARP_JOINT_NAMES,
)

pytestmark = pytest.mark.integration

_FIXTURE = Path(__file__).parent / "data" / "articulation_ordering_branching.usda"


def _sim_context(device: str = "cpu", *, use_newton_actuators: bool = False):
    """Build a local CPU OVPhysX context from an in-memory USD stage."""
    return build_simulation_context(
        sim_cfg=SimulationCfg(
            physics=OvPhysxCfg(),
            device=device,
            gravity=(0.0, 0.0, 0.0),
            use_newton_actuators=use_newton_actuators,
        ),
        auto_add_lighting=False,
    )


def _spawn_ordered_articulation(*, native_actuator: bool = False) -> Articulation:
    """Spawn the cached local branching fixture in nonidentity public order."""
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
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(usd_path=str(_FIXTURE)),
            actuators={"joints": actuator_cfg},
            joint_ordering="mjwarp",
            body_ordering="mjwarp",
        )
    )
    fixed_joint = UsdPhysics.FixedJoint.Define(sim_utils.get_current_stage(), "/World/Robot/fixed_root")
    fixed_joint.GetBody1Rel().SetTargets(["/World/Robot/base"])
    return articulation


def test_articulation_real_ovphysx_seams() -> None:
    """Prove ordering, partial state/properties, drive delivery, and dynamics access."""
    with _sim_context() as sim:
        articulation = _spawn_ordered_articulation()
        sim.reset()

        assert articulation.is_initialized
        assert articulation.is_fixed_base
        assert tuple(articulation.joint_names) == BRANCHING_MJWARP_JOINT_NAMES
        assert tuple(articulation.body_names) == BRANCHING_MJWARP_BODY_NAMES
        assert articulation.joint_ordering is not None
        assert articulation.body_ordering is not None

        joint_ids = torch.tensor([articulation.num_joints - 1, 0], dtype=torch.int32)
        target_position = torch.tensor([[0.21, -0.13]])
        target_velocity = torch.tensor([[0.41, -0.23]])
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

        body_ids = torch.tensor([articulation.num_bodies - 1, 1], dtype=torch.int32)
        masses = torch.tensor([[2.5, 3.5]])
        articulation.set_masses_index(masses=masses, body_ids=body_ids)
        coms = articulation.data.body_com_pose_b.torch[:, body_ids].clone()
        coms[0, 0, :3] = torch.tensor([0.02, -0.01, 0.03])
        coms[0, 1, :3] = torch.tensor([-0.03, 0.01, 0.02])
        articulation.set_coms_index(coms=wp.from_torch(coms, dtype=wp.transformf), body_ids=body_ids)
        inertias = articulation.data.body_inertia.torch[:, body_ids].clone()
        inertias[0, 0, 0] *= 1.2
        inertias[0, 1, 4] *= 1.3
        articulation.set_inertias_index(inertias=inertias, body_ids=body_ids)
        torch.testing.assert_close(articulation.data.body_mass.torch[:, body_ids], masses)
        torch.testing.assert_close(articulation.data.body_com_pose_b.torch[:, body_ids], coms)
        torch.testing.assert_close(articulation.data.body_inertia.torch[:, body_ids], inertias)

        drive_target = articulation.data.joint_pos.torch.clone()
        drive_target[:, 0] += 0.15
        articulation.actuators.target_command.set_position_index(value=drive_target, full_data=True)
        articulation.write_data_to_sim()
        backend_target = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_POSITION_TARGET))
        backend_to_user = list(articulation.joint_ordering.backend_to_user_indices)
        torch.testing.assert_close(backend_target, drive_target[:, backend_to_user])

        sim.step()
        articulation.update(sim.cfg.dt)
        jacobian = articulation.data.body_link_jacobian_w.torch
        mass_matrix = articulation.data.mass_matrix.torch
        assert jacobian.shape == (1, articulation.num_bodies - 1, 6, articulation.num_joints)
        assert mass_matrix.shape == (1, articulation.num_joints, articulation.num_joints)
        assert torch.isfinite(jacobian).all()
        assert torch.isfinite(mass_matrix).all()
        torch.testing.assert_close(mass_matrix, mass_matrix.transpose(-1, -2), atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native actuator wheel probe requires CUDA")
def test_articulation_native_actuator_submits_real_ovphysx_effort() -> None:
    """Prove the local native controller reaches the real OVPhysX effort binding."""
    with _sim_context(device="cuda:0", use_newton_actuators=True) as sim:
        articulation = _spawn_ordered_articulation(native_actuator=True)
        sim.reset()

        assert articulation._actuator_control.native_actuator_path_active
        assert articulation.newton_actuator_adapter is not None
        target = articulation.data.joint_pos.torch.clone() + 0.2
        articulation.actuators.target_command.set_position_index(value=target)
        articulation.write_data_to_sim()

        raw_effort = wp.to_torch(articulation._physx_actuator_wrapper.joint_f_2d)
        backend_effort = wp.to_torch(articulation.root_view.get_attribute(TT.DOF_ACTUATION_FORCE))
        assert torch.any(raw_effort != 0.0)
        torch.testing.assert_close(backend_effort, raw_effort)
