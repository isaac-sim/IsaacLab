# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parity tests for the Newton-backed operational-space controller."""

import pytest
import torch
import warp as wp

from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.utils.math import compute_pose_error, matrix_from_quat

pytestmark = pytest.mark.integration

_NUM_ENVS = 4
_NUM_DOF = 7
_DEVICE = "cpu"


def _random_quat(generator: torch.Generator) -> torch.Tensor:
    """Random unit quaternions in ``(x, y, z, w)`` order, shape (``_NUM_ENVS``, 4)."""
    quat = torch.randn(_NUM_ENVS, 4, generator=generator, device=_DEVICE)
    return quat / quat.norm(dim=-1, keepdim=True)


def _reference_efforts(
    controller: OperationalSpaceController,
    jacobian_b: torch.Tensor,
    ee_pose_b: torch.Tensor,
    ee_vel_b: torch.Tensor,
    ee_force_b: torch.Tensor,
    mass_matrix: torch.Tensor,
    gravity: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    nullspace_joint_pos_target: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the operational-space law independently in Torch.

    Gains and selection axes are rotated from the task frame into the root frame here, the way the
    controller used to do it, so the oracle is independent of the Newton backend's own frame
    handling.
    """
    cfg = controller.cfg
    num_envs, _, num_dof = jacobian_b.shape
    joint_efforts = torch.zeros(num_envs, num_dof, device=_DEVICE)

    rot_task_b = matrix_from_quat(controller._task_frame_pose_b[:, 3:])
    rot_b_task = rot_task_b.mT

    def to_root_frame(axis_values: torch.Tensor) -> torch.Tensor:
        """Block-rotate a per-axis task-frame diagonal into a root-frame 6x6 matrix."""
        task = axis_values
        root = torch.zeros_like(task)
        root[:, 0:3, 0:3] = rot_task_b @ task[:, 0:3, 0:3] @ rot_b_task
        root[:, 3:6, 3:6] = rot_task_b @ task[:, 3:6, 3:6] @ rot_b_task
        return root

    os_mass_matrix_b = torch.zeros(num_envs, 6, 6, device=_DEVICE)
    mass_matrix_inv = None

    if controller.desired_ee_pose_b is not None:
        pose_error_b = torch.cat(
            compute_pose_error(
                ee_pose_b[:, :3],
                ee_pose_b[:, 3:],
                controller.desired_ee_pose_b[:, :3],
                controller.desired_ee_pose_b[:, 3:],
                rot_error_type="axis_angle",
            ),
            dim=-1,
        )
        des_ee_acc_b = to_root_frame(controller._motion_p_gains_task) @ pose_error_b.unsqueeze(-1) + to_root_frame(
            controller._motion_d_gains_task
        ) @ (-ee_vel_b).unsqueeze(-1)
        selection_motion_b = to_root_frame(controller._selection_matrix_motion_task)
        if cfg.implementation == "newton":
            des_ee_acc_b = selection_motion_b @ des_ee_acc_b
        if cfg.inertial_dynamics_decoupling:
            mass_matrix_inv = torch.inverse(mass_matrix)
            if cfg.partial_inertial_dynamics_decoupling:
                os_mass_matrix_b[:, 0:3, 0:3] = torch.inverse(
                    jacobian_b[:, 0:3] @ mass_matrix_inv @ jacobian_b[:, 0:3].mT
                )
                os_mass_matrix_b[:, 3:6, 3:6] = torch.inverse(
                    jacobian_b[:, 3:6] @ mass_matrix_inv @ jacobian_b[:, 3:6].mT
                )
            else:
                os_mass_matrix_b[:] = torch.inverse(jacobian_b @ mass_matrix_inv @ jacobian_b.mT)
            os_command_forces_b = os_mass_matrix_b @ des_ee_acc_b
        else:
            os_command_forces_b = des_ee_acc_b
        if cfg.implementation != "newton":
            os_command_forces_b = selection_motion_b @ os_command_forces_b
        joint_efforts += (jacobian_b.mT @ os_command_forces_b).squeeze(-1)

    if controller.desired_ee_wrench_b is not None:
        if cfg.contact_wrench_stiffness_task is not None:
            measured_wrench_b = torch.zeros(num_envs, 6, device=_DEVICE)
            measured_wrench_b[:, 0:3] = ee_force_b
            measured_wrench_b[:, 3:6] = controller.desired_ee_wrench_b[:, 3:6]
            wrench_command_b = controller.desired_ee_wrench_b.unsqueeze(-1) + to_root_frame(
                controller._contact_wrench_p_gains_task
            ) @ (controller.desired_ee_wrench_b - measured_wrench_b).unsqueeze(-1)
        else:
            wrench_command_b = controller.desired_ee_wrench_b.unsqueeze(-1)
        selection_force_b = to_root_frame(controller._selection_matrix_force_task)
        joint_efforts += (jacobian_b.mT @ selection_force_b @ wrench_command_b).squeeze(-1)

    if cfg.gravity_compensation:
        joint_efforts += gravity

    if cfg.nullspace_control == "position":
        if cfg.inertial_dynamics_decoupling and not cfg.partial_inertial_dynamics_decoupling:
            jacobian_pinv_transpose = os_mass_matrix_b @ jacobian_b @ mass_matrix_inv
        else:
            jacobian_pinv_transpose = torch.pinverse(jacobian_b).mT
        nullspace_jacobian_transpose = torch.eye(n=num_dof, device=_DEVICE) - jacobian_b.mT @ jacobian_pinv_transpose
        joint_acc_nullspace = (
            controller._nullspace_p_gain * (nullspace_joint_pos_target - joint_pos)
            + controller._nullspace_d_gain * (-joint_vel)
        ).unsqueeze(-1)
        posture_force = (
            mass_matrix @ joint_acc_nullspace
            if cfg.implementation != "newton" or cfg.inertial_dynamics_decoupling
            else joint_acc_nullspace
        )
        joint_efforts += (nullspace_jacobian_transpose @ posture_force).squeeze(-1)

    return joint_efforts


_SCENARIOS = {
    "pose_abs": dict(target_types=["pose_abs"]),
    "pose_rel": dict(target_types=["pose_rel"]),
    "pose_abs_task_frame": dict(target_types=["pose_abs"], task_frame=True),
    "pose_abs_decoupled": dict(target_types=["pose_abs"], inertial_dynamics_decoupling=True, task_frame=True),
    "pose_abs_partial_decoupled": dict(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
    "pose_abs_gravity": dict(target_types=["pose_abs"], gravity_compensation=True, task_frame=True),
    "pose_abs_nullspace": dict(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        nullspace_control="position",
        task_frame=True,
    ),
    "pose_abs_nullspace_without_inertia": dict(target_types=["pose_abs"], nullspace_control="position"),
    "pose_abs_nullspace_partial_inertia": dict(
        target_types=["pose_abs"],
        nullspace_control="position",
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=True,
    ),
    "wrench_open_loop": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        task_frame=True,
    ),
    "wrench_closed_loop": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        task_frame=True,
    ),
    "wrench_closed_loop_decoupled": dict(
        target_types=["pose_abs", "wrench_abs"],
        contact_wrench_stiffness_task=0.5,
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        inertial_dynamics_decoupling=True,
        gravity_compensation=True,
        nullspace_control="position",
        task_frame=True,
    ),
    "wrench_decoupled_partial_axes": dict(
        target_types=["pose_abs", "wrench_abs"],
        motion_control_axes_task=(1, 1, 0, 1, 1, 1),
        contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
        inertial_dynamics_decoupling=True,
        task_frame=True,
    ),
    "variable_kp": dict(target_types=["pose_abs"], impedance_mode="variable_kp", task_frame=True),
    "variable": dict(target_types=["pose_abs"], impedance_mode="variable", task_frame=True),
}


def _build(scenario: dict, implementation: str = "newton") -> tuple[OperationalSpaceController, bool]:
    """Instantiate a controller from a scenario, returning it with its task-frame flag."""
    scenario = dict(scenario)
    task_frame = scenario.pop("task_frame", False)
    cfg = OperationalSpaceControllerCfg(
        implementation=implementation,
        motion_stiffness_task=(120.0, 130.0, 140.0, 15.0, 16.0, 17.0),
        motion_damping_ratio_task=(1.0, 1.1, 0.9, 1.0, 1.2, 0.8),
        **scenario,
    )
    return OperationalSpaceController(cfg, _NUM_ENVS, _DEVICE), task_frame


@pytest.mark.parametrize("implementation", ["native", "newton"])
@pytest.mark.parametrize("scenario_name", list(_SCENARIOS))
def test_backend_matches_operational_space_law(scenario_name: str, implementation: str) -> None:
    """The Newton-backed controller matches an independent operational-space reference."""
    generator = torch.Generator(device=_DEVICE).manual_seed(0)
    controller, task_frame = _build(_SCENARIOS[scenario_name], implementation)

    ee_pose_b = torch.cat([0.4 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
    ee_vel_b = 0.2 * torch.randn(_NUM_ENVS, 6, generator=generator)
    ee_force_b = 3.0 * torch.randn(_NUM_ENVS, 3, generator=generator)
    jacobian_b = torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator)
    factor = torch.randn(_NUM_ENVS, _NUM_DOF, _NUM_DOF, generator=generator)
    mass_matrix = factor @ factor.mT + 3.0 * torch.eye(_NUM_DOF)  # SPD
    gravity = 0.5 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    joint_pos = 0.3 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    joint_vel = 0.2 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    nullspace_target = 0.1 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    task_frame_pose_b = (
        torch.cat([0.2 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
        if task_frame
        else None
    )

    command = []
    for target_type in controller.cfg.target_types:
        if target_type == "pose_abs":
            command.append(
                torch.cat([0.3 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
            )
        elif target_type == "pose_rel":
            command.append(0.1 * torch.randn(_NUM_ENVS, 6, generator=generator))
        else:
            command.append(5.0 * torch.randn(_NUM_ENVS, 6, generator=generator))
    if controller.cfg.impedance_mode in ("variable_kp", "variable"):
        command.append(torch.rand(_NUM_ENVS, 6, generator=generator) * 150.0 + 50.0)
    if controller.cfg.impedance_mode == "variable":
        command.append(torch.rand(_NUM_ENVS, 6, generator=generator) * 2.0)
    controller.set_command(
        torch.cat(command, dim=-1), current_ee_pose_b=ee_pose_b, current_task_frame_pose_b=task_frame_pose_b
    )

    efforts = controller.compute(
        jacobian_b=jacobian_b,
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=ee_vel_b,
        current_ee_force_b=ee_force_b,
        mass_matrix=mass_matrix,
        gravity=gravity,
        current_joint_pos=joint_pos,
        current_joint_vel=joint_vel,
        nullspace_joint_pos_target=nullspace_target,
    ).clone()
    reference = _reference_efforts(
        controller,
        jacobian_b,
        ee_pose_b,
        ee_vel_b,
        ee_force_b,
        mass_matrix,
        gravity,
        joint_pos,
        joint_vel,
        nullspace_target,
    )
    torch.testing.assert_close(efforts, reference, atol=1e-3, rtol=1e-3)


def test_reset_clears_the_task_space_targets() -> None:
    """After a reset no target is commanded, so only gravity compensation remains."""
    generator = torch.Generator(device=_DEVICE).manual_seed(0)
    controller, _ = _build(
        dict(
            target_types=["pose_abs", "wrench_abs"],
            gravity_compensation=True,
            # closed-loop force control, so a stale measured wrench would surface as torque
            contact_wrench_stiffness_task=(0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            contact_wrench_control_axes_task=(0, 0, 1, 0, 0, 0),
        )
    )

    ee_pose_b = torch.cat([0.4 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1)
    gravity = 0.5 * torch.randn(_NUM_ENVS, _NUM_DOF, generator=generator)
    command = torch.cat(
        [
            torch.cat([0.3 * torch.randn(_NUM_ENVS, 3, generator=generator), _random_quat(generator)], dim=-1),
            5.0 * torch.randn(_NUM_ENVS, 6, generator=generator),
        ],
        dim=-1,
    )
    controller.set_command(command, current_ee_pose_b=ee_pose_b)
    # step once while commanded to populate the measured-wrench port;
    # only then does the reset have stale state to clear
    controller.compute(
        jacobian_b=torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator),
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=0.2 * torch.randn(_NUM_ENVS, 6, generator=generator),
        current_ee_force_b=3.0 * torch.randn(_NUM_ENVS, 3, generator=generator),
        gravity=gravity,
    )
    controller.reset()

    efforts = controller.compute(
        jacobian_b=torch.randn(_NUM_ENVS, 6, _NUM_DOF, generator=generator),
        current_ee_pose_b=ee_pose_b,
        current_ee_vel_b=0.2 * torch.randn(_NUM_ENVS, 6, generator=generator),
        current_ee_force_b=3.0 * torch.randn(_NUM_ENVS, 3, generator=generator),
        gravity=gravity,
    )
    torch.testing.assert_close(efforts, gravity, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_joints", [4, 6])
def test_inertial_decoupling_requires_six_controlled_joints(num_joints: int) -> None:
    """Newton rejects under-actuated decoupling but accepts the six-joint boundary."""
    controller = OperationalSpaceController(
        OperationalSpaceControllerCfg(
            implementation="newton", target_types=["pose_abs"], inertial_dynamics_decoupling=True
        ),
        1,
        "cpu",
    )
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    controller.set_command(pose)
    kwargs = dict(
        jacobian_b=torch.eye(6, num_joints).unsqueeze(0),
        current_ee_pose_b=pose,
        current_ee_vel_b=torch.zeros(1, 6),
        mass_matrix=torch.eye(num_joints).unsqueeze(0),
    )
    if num_joints < 6:
        with pytest.raises(ValueError, match="at least 6 controlled DOFs"):
            controller.compute(**kwargs)
    else:
        torch.testing.assert_close(controller.compute(**kwargs), torch.zeros(1, num_joints))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA")
def test_captured_compute_tracks_commands_and_recaptures_after_reset() -> None:
    """Captured compute observes updated targets and gains; reset starts a new capture lifetime."""
    device = "cuda:0"
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        controller = OperationalSpaceController(
            OperationalSpaceControllerCfg(
                implementation="newton", target_types=["pose_abs"], impedance_mode="variable_kp"
            ),
            1,
            device,
        )
        pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=device)
        jacobian = torch.eye(6, 7, device=device).unsqueeze(0)
        velocity = torch.zeros(1, 6, device=device)
        command = torch.cat((pose.clone(), torch.full((1, 6), 100.0, device=device)), dim=1)
        command[:, 0] = 0.1

        def compute():
            return controller.compute(jacobian, current_ee_pose_b=pose, current_ee_vel_b=velocity)

        for _ in range(2):
            controller.set_command(command)
            compute()
            with wp.ScopedCapture() as capture:
                efforts = compute()
            command[:, 0] *= 2
            command[:, 7:] *= 2
            controller.set_command(command)
            wp.capture_launch(capture.graph)
            expected = torch.zeros(1, 7, device=device)
            expected[:, 0] = command[:, 0] * command[:, 7]
            torch.testing.assert_close(efforts, expected)
            controller.reset()
            torch.testing.assert_close(compute(), torch.zeros_like(expected))


@pytest.mark.parametrize("implementation", ["native", "newton"])
def test_joint_count_inference_and_runtime_gravity_toggle(implementation):
    """Both solvers infer dimensions and respect enabling/disabling gravity between calls."""
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], implementation=implementation)
    controller = OperationalSpaceController(cfg, 1, "cpu")
    pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    controller.set_command(pose)
    for count in (7, 6, 8):
        gravity = torch.arange(count, dtype=torch.float32).unsqueeze(0)
        for enabled in (False, True, False):
            cfg.gravity_compensation = enabled
            actual = controller.compute(
                jacobian_b=torch.eye(6, count).unsqueeze(0),
                current_ee_pose_b=pose,
                current_ee_vel_b=torch.zeros(1, 6),
                gravity=gravity if enabled else None,
            )
            torch.testing.assert_close(actual, gravity if enabled else torch.zeros_like(gravity))
