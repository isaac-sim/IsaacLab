# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from unittest.mock import Mock

import torch

from isaaclab_tasks.core.locomotion.locomotion_direct_env import LocomotionDirectEnv


def test_action_target_is_staged_once_per_environment_step():
    """Test that effort targets are staged once and persist across decimation substeps."""
    env = object.__new__(LocomotionDirectEnv)
    env._is_closed = True
    env.action_scale = 0.5
    env.joint_gears = torch.tensor([2.0, 3.0, 4.0])
    env._joint_dof_idx = torch.arange(3)
    env.robot = Mock()
    actions = torch.tensor([[-2.0, -0.5, 1.5], [0.25, 2.0, -1.25]])

    env._pre_physics_step(actions)
    for _ in range(4):
        env._apply_action()

    env.robot.set_joint_effort_target_index.assert_called_once()
    expected_target = 0.5 * env.joint_gears * torch.clamp(actions, -1.0, 1.0)
    torch.testing.assert_close(env.robot.set_joint_effort_target_index.call_args.kwargs["target"], expected_target)
    torch.testing.assert_close(env.actions, actions)
    assert env.actions.data_ptr() != actions.data_ptr()
