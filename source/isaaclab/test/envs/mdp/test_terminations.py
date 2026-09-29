# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for MDP termination terms."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg


def test_joint_effort_limit_terminates_only_environments_with_clipped_selected_joints():
    """Terminate exactly the environments whose applied effort differs from the computed effort on selected joints."""
    computed = torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
    # Environment 0 clips the selected joint, environment 1 clips nothing, environment 2 clips the selected joint
    # downward, and environment 3 clips only joints that are not selected.
    applied = torch.tensor([[10.0, 19.0, 30.0], [10.0, 20.0, 30.0], [10.0, 17.0, 30.0], [9.0, 20.0, 29.0]])
    robot = SimpleNamespace(
        actuators=SimpleNamespace(
            computed_effort=SimpleNamespace(torch=computed), applied_effort=SimpleNamespace(torch=applied)
        )
    )
    env = SimpleNamespace(scene={"robot": robot})

    result = joint_effort_out_of_limit(env, SceneEntityCfg("robot", joint_ids=[1]))

    torch.testing.assert_close(result, torch.tensor([True, False, True, False]))
