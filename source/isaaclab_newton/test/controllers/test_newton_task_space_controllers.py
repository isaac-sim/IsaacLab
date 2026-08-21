# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless controller bridges for Newton articulation dynamics data."""

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
    OperationalSpaceController,
    OperationalSpaceControllerCfg,
)
from isaaclab.utils.math import subtract_frame_transforms
from source.isaaclab_newton.test.articulation_test_utils import author_fixed_spatial_chain, build_newton_context


def _end_effector_pose_b(articulation) -> torch.Tensor:
    """Return the last link pose in the fixed root frame."""
    root_pose_w = articulation.data.root_link_pose_w.torch
    ee_pose_w = articulation.data.body_link_pose_w.torch[:, -1]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, :3], root_pose_w[:, 3:7], ee_pose_w[:, :3], ee_pose_w[:, 3:7]
    )
    return torch.cat((ee_pos_b, ee_quat_b), dim=-1)


def _unpowered_actuators() -> dict:
    """Return passive joint configuration for effort-controlled bridges."""
    return {
        "joints": ImplicitActuatorCfg(
            joint_names_expr=["Joint_.*"],
            stiffness=0.0,
            damping=0.0,
        )
    }


def test_differential_ik_tracks_local_newton_chain() -> None:
    """A DifferentialIK command must move the real Newton end effector toward its target."""
    with build_newton_context() as sim:
        articulation = author_fixed_spatial_chain()
        sim.reset()

        initial_pose_b = _end_effector_pose_b(articulation)
        target_pose_b = initial_pose_b.clone()
        target_pose_b[:, 0] += 0.04
        jacobian = articulation.data.body_link_jacobian_w.torch[:, -1]
        controller = DifferentialIKController(
            DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            num_envs=1,
            device="cpu",
        )
        controller.set_command(target_pose_b, ee_quat=initial_pose_b[:, 3:7])
        joint_target = controller.compute(
            initial_pose_b[:, :3],
            initial_pose_b[:, 3:7],
            jacobian,
            articulation.data.joint_pos.torch,
        )
        assert torch.linalg.vector_norm(joint_target - articulation.data.joint_pos.torch) > 1e-3
        initial_error = torch.linalg.vector_norm(target_pose_b[:, :3] - initial_pose_b[:, :3])

        articulation.actuators.target_command.set_position_index(value=joint_target)
        for _ in range(20):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)

        final_pose_b = _end_effector_pose_b(articulation)
        final_error = torch.linalg.vector_norm(target_pose_b[:, :3] - final_pose_b[:, :3])
        assert final_error < 0.5 * initial_error


def test_operational_space_consumes_newton_jacobian_mass_and_gravity() -> None:
    """OSC must turn live Newton dynamics into a finite, mass-dependent motion response."""
    with build_newton_context(gravity=(0.0, -9.81, 0.0)) as sim:
        articulation = author_fixed_spatial_chain(actuators=_unpowered_actuators())
        sim.reset()

        ee_pose_b = _end_effector_pose_b(articulation)
        target_pose_b = ee_pose_b.clone()
        target_pose_b[:, 0] += 0.02
        jacobian = articulation.data.body_link_jacobian_w.torch[:, -1]
        mass_matrix = articulation.data.mass_matrix.torch
        gravity = articulation.data.gravity_compensation_forces.torch
        assert torch.linalg.matrix_rank(jacobian).item() == 6
        assert torch.all(torch.linalg.eigvalsh(mass_matrix) > 0.0)
        assert gravity[0, 1] > 1.0

        cfg = OperationalSpaceControllerCfg(
            target_types=["pose_abs"],
            inertial_dynamics_decoupling=True,
            gravity_compensation=True,
            motion_stiffness_task=20.0,
            motion_damping_ratio_task=1.0,
        )
        controller = OperationalSpaceController(cfg, num_envs=1, device="cpu")
        controller.set_command(target_pose_b)
        effort = controller.compute(
            jacobian,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=articulation.data.body_link_vel_w.torch[:, -1],
            mass_matrix=mass_matrix,
            gravity=gravity,
        )

        noninertial_cfg = cfg.replace(inertial_dynamics_decoupling=False)
        noninertial_controller = OperationalSpaceController(noninertial_cfg, num_envs=1, device="cpu")
        noninertial_controller.set_command(target_pose_b)
        noninertial_effort = noninertial_controller.compute(
            jacobian,
            current_ee_pose_b=ee_pose_b,
            current_ee_vel_b=articulation.data.body_link_vel_w.torch[:, -1],
            gravity=gravity,
        )
        assert torch.isfinite(effort).all()
        assert torch.linalg.vector_norm(effort) > 1.0
        assert not torch.allclose(effort, noninertial_effort)
        initial_error = torch.linalg.vector_norm(target_pose_b[:, :3] - ee_pose_b[:, :3])

        articulation.actuators.target_command.set_effort_index(value=effort)
        for _ in range(8):
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)

        final_error = torch.linalg.vector_norm(target_pose_b[:, :3] - _end_effector_pose_b(articulation)[:, :3])
        assert final_error < initial_error


def _gravity_drift(*, compensate: bool) -> torch.Tensor:
    """Measure passive joint drift in a new context, optionally applying OSC gravity effort."""
    with build_newton_context(gravity=(0.0, -9.81, 0.0)) as sim:
        articulation = author_fixed_spatial_chain(actuators=_unpowered_actuators())
        sim.reset()
        initial_joint_pos = articulation.data.joint_pos.torch.clone()
        controller = OperationalSpaceController(
            OperationalSpaceControllerCfg(
                target_types=["wrench_abs"],
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                gravity_compensation=True,
            ),
            num_envs=1,
            device="cpu",
        )
        controller.set_command(torch.zeros((1, 6)))

        for _ in range(20):
            effort = torch.zeros_like(articulation.data.joint_pos.torch)
            if compensate:
                effort = controller.compute(
                    articulation.data.body_link_jacobian_w.torch[:, -1],
                    gravity=articulation.data.gravity_compensation_forces.torch,
                )
            articulation.actuators.target_command.set_effort_index(value=effort)
            articulation.write_data_to_sim()
            sim.step()
            articulation.update(sim.cfg.dt)

        return torch.linalg.vector_norm(articulation.data.joint_pos.torch - initial_joint_pos)


def test_operational_space_gravity_compensation_holds_static_chain() -> None:
    """Live Newton gravity effort must hold the chain materially closer to its initial state."""
    uncompensated_drift = _gravity_drift(compensate=False)
    compensated_drift = _gravity_drift(compensate=True)

    assert uncompensated_drift > 1e-2
    assert compensated_drift < 0.1 * uncompensated_drift
