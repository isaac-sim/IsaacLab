# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MDP termination terms."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.terminations import joint_vel_out_of_manual_limit
from isaaclab.managers import SceneEntityCfg


def test_joint_velocity_limit_terminates_non_finite_values() -> None:
    """Terminate environments with non-finite joint velocities before they reach observations."""
    joint_vel = torch.tensor([[1.0, 2.0], [float("nan"), 2.0], [float("inf"), 2.0], [101.0, 2.0]])
    robot = SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=joint_vel)))
    env = SimpleNamespace(scene={"robot": robot})

    result = joint_vel_out_of_manual_limit(env, max_velocity=100.0, asset_cfg=SceneEntityCfg("robot"))

    torch.testing.assert_close(result, torch.tensor([False, True, True, True]))
