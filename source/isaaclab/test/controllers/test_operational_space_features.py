# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
import torch

from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("partial_inertial_decoupling", [False, True])
def test_inertial_decoupling_handles_singular_task_inertia(partial_inertial_decoupling: bool):
    """Inertial decoupling produces finite efforts for rank-deficient Jacobians in a mixed batch."""
    num_envs = 3
    num_joints = 7
    cfg = OperationalSpaceControllerCfg(
        target_types=["pose_abs"],
        inertial_dynamics_decoupling=True,
        partial_inertial_dynamics_decoupling=partial_inertial_decoupling,
    )
    controller = OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")

    target_pose = torch.tensor([[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    controller.set_command(target_pose)

    jacobian = torch.zeros(num_envs, 6, num_joints)
    jacobian[:, :6, :6] = torch.eye(6)
    jacobian[1, 1] = 0.0  # singular translational task-space inertia
    jacobian[2, 5] = 0.0  # singular rotational task-space inertia

    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)

    joint_efforts = controller.compute(
        jacobian_b=jacobian,
        current_ee_pose_b=current_pose,
        current_ee_vel_b=torch.zeros(num_envs, 6),
        mass_matrix=torch.eye(num_joints).repeat(num_envs, 1, 1),
    )

    assert torch.isfinite(joint_efforts).all()
    torch.testing.assert_close(
        joint_efforts[:, 0],
        torch.full((num_envs,), 10.0),
    )
    torch.testing.assert_close(
        joint_efforts[:, 1:],
        torch.zeros(num_envs, num_joints - 1),
    )


def _pose_abs_controller(num_envs: int) -> OperationalSpaceController:
    cfg = OperationalSpaceControllerCfg(target_types=["pose_abs"], inertial_dynamics_decoupling=False)
    return OperationalSpaceController(cfg, num_envs=num_envs, device="cpu")


def test_pose_abs_target_quaternion_is_normalized():
    """Scaling an absolute pose quaternion must not change the target orientation or the commanded efforts."""
    num_envs = 2
    unit_quat = torch.tensor([[0.0, 0.0, 0.3826834, 0.9238795]]).repeat(num_envs, 1)  # 45 deg about z
    unit_target = torch.cat([torch.tensor([[0.1, 0.0, 0.0]]).repeat(num_envs, 1), unit_quat], dim=-1)
    scaled_target = unit_target.clone()
    scaled_target[0, 3:7] *= 3.0
    scaled_target[1, 3:7] *= -0.25  # sign flip encodes the same rotation

    current_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).repeat(num_envs, 1)
    jacobian = torch.zeros(num_envs, 6, 7)
    jacobian[:, :6, :6] = torch.eye(6)

    efforts = []
    for target in (unit_target, scaled_target):
        controller = _pose_abs_controller(num_envs)
        controller.set_command(target, current_ee_pose_b=current_pose)
        torch.testing.assert_close(
            torch.linalg.norm(controller.desired_ee_pose_task[:, 3:7], dim=-1), torch.ones(num_envs)
        )
        efforts.append(
            controller.compute(
                jacobian_b=jacobian, current_ee_pose_b=current_pose, current_ee_vel_b=torch.zeros(num_envs, 6)
            )
        )

    torch.testing.assert_close(efforts[0], efforts[1])
    assert torch.isfinite(efforts[1]).all()


def test_pose_abs_degenerate_quaternion_falls_back_to_current_orientation():
    """Zero and non-finite quaternions keep the current orientation, or identity without a current pose."""
    num_envs = 3
    current_quat = torch.tensor([[0.0, 0.7071068, 0.0, 0.7071068]]).repeat(num_envs, 1)  # 90 deg about y
    current_pose = torch.cat([torch.zeros(num_envs, 3), current_quat], dim=-1)
    target = torch.zeros(num_envs, 7)
    target[0, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    target[1, 3:7] = 0.0
    target[2, 3:7] = torch.tensor([float("nan"), 0.0, 0.0, 1.0])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target, current_ee_pose_b=current_pose)
    torch.testing.assert_close(controller.desired_ee_pose_task[0, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(controller.desired_ee_pose_task[1:, 3:7], current_quat[1:])

    controller = _pose_abs_controller(num_envs)
    controller.set_command(target)
    torch.testing.assert_close(
        controller.desired_ee_pose_task[1:, 3:7], torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(num_envs - 1, 1)
    )
