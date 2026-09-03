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
