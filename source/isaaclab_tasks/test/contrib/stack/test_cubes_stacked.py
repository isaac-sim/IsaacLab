# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch

from isaaclab_tasks.contrib.stack.config.franka.stack_rl_env_cfg import FrankaCubeStackRLEnvCfg
from isaaclab_tasks.contrib.stack.mdp import cubes_stacked

pytestmark = pytest.mark.kitless


def test_stacked_cubes_can_require_the_expected_vertical_separation() -> None:
    # Each row holds the center heights of one scene, bottom to top.
    heights = torch.tensor(
        [
            [0.0, 0.0477, 0.0954],
            [0.0, 0.0020, 0.0497],
            [0.0, 0.0477, 0.0497],
        ]
    )
    num_envs = len(heights)
    scene = {
        "robot": SimpleNamespace(
            data=SimpleNamespace(joint_pos=SimpleNamespace(torch=torch.full((num_envs, 2), 0.04))),
            find_joints=lambda names: ([0, 1], names),
        )
    }
    for index, center_z in enumerate(heights.T, start=1):
        positions = torch.zeros((num_envs, 3))
        positions[:, 2] = center_z
        scene[f"cube_{index}"] = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=positions),
                root_lin_vel_w=SimpleNamespace(torch=torch.zeros_like(positions)),
            )
        )
    cfg = FrankaCubeStackRLEnvCfg()
    success = cfg.terminations.success
    env = SimpleNamespace(
        scene=scene,
        cfg=cfg,
        device="cpu",
    )

    assert cubes_stacked(env).tolist() == [True, True, True]
    assert success.func(env, **success.params).tolist() == [True, False, False]
    assert cubes_stacked(env, cube_3_cfg=None, symmetric_height_tolerance=True).tolist() == [True, False, True]
