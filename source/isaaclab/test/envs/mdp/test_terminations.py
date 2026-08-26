# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused unit tests for MDP termination terms."""

from types import SimpleNamespace

import torch

from isaaclab.envs.mdp.terminations import joint_effort_out_of_limit
from isaaclab.managers import SceneEntityCfg


def test_joint_effort_limit_terminates_only_environments_with_clipped_selected_joints() -> None:
    """Compare computed and applied torque over the selected joints for each environment."""
    robot = SimpleNamespace(
        actuators=SimpleNamespace(
            computed_effort=SimpleNamespace(
                torch=torch.tensor([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]])
            ),
            applied_effort=SimpleNamespace(
                torch=torch.tensor([[10.0, 19.0, 30.0], [10.0, 20.0, 30.0], [10.0, 17.0, 30.0]])
            ),
        )
    )
    env = SimpleNamespace(scene={"robot": robot})
    asset_cfg = SceneEntityCfg("robot", joint_ids=[1])

    result = joint_effort_out_of_limit(env, asset_cfg)

    torch.testing.assert_close(result, torch.tensor([True, False, True]))
