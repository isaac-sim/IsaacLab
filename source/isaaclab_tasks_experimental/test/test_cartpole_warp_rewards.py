# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Cartpole Warp reward launch replay."""

from types import SimpleNamespace

import numpy as np
import warp as wp
from isaaclab_tasks_experimental.manager_based.classic.cartpole.mdp.rewards import joint_pos_target_l2

from isaaclab.utils.warp import WarpLaunchCache


def test_joint_pos_target_l2_replay_selects_target_site():
    """A changed reward target should use a command recorded for that static value."""
    joint_pos = wp.array([[0.0], [2.0]], dtype=wp.float32, device="cpu")
    joint_mask = wp.array([True], dtype=wp.bool, device="cpu")
    output = wp.zeros(2, dtype=wp.float32, device="cpu")
    asset = SimpleNamespace(data=SimpleNamespace(joint_pos=SimpleNamespace(warp=joint_pos)))
    env = SimpleNamespace(scene={"robot": asset}, num_envs=2, device="cpu", _warp_launch=WarpLaunchCache(device="cpu"))
    asset_cfg = SimpleNamespace(name="robot", joint_mask=joint_mask)

    joint_pos_target_l2(env, output, target=0.0, asset_cfg=asset_cfg)
    np.testing.assert_allclose(output.numpy(), np.array([0.0, 4.0], dtype=np.float32), atol=2.0e-6)

    joint_pos_target_l2(env, output, target=1.0, asset_cfg=asset_cfg)
    np.testing.assert_allclose(output.numpy(), np.array([1.0, 1.0], dtype=np.float32), atol=2.0e-6)
