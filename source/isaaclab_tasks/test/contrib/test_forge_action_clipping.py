# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Check FORGE orientation targets without constructing a simulation scene."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_tasks.contrib.forge.forge_env import ForgeEnv

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("current", "target_yaw", "expected", "expected_delta_yaw"),
    [
        ((179.0, 0.0, 0.0), 0.0, (180.0, 0.0, 0.0), 0.0),
        ((181.0, 0.0, 0.0), 0.0, (180.0, 0.0, 0.0), 0.0),
        ((200.0, 0.0, 0.0), 0.0, (195.0, 0.0, 0.0), 0.0),
        ((179.0, -10.0, 0.0), 0.0, (180.0, -5.0, 0.0), 0.0),
        ((179.0, 10.0, 0.0), 0.0, (180.0, 5.0, 0.0), 0.0),
        ((179.0, 0.0, -170.0), 170.0, (180.0, 0.0, 185.0), -20.0),
        ((179.0, 0.0, 230.0), -90.0, (180.0, 0.0, 225.0), -320.0),
    ],
    ids=[
        "roll-below-pi",
        "roll-above-pi",
        "roll-capped",
        "pitch-negative",
        "pitch-positive",
        "yaw-across-pi",
        "yaw-avoids-joint-limit",
    ],
)
def test_apply_action_clips_orientation(current, target_yaw, expected, expected_delta_yaw):
    """Correct tilt toward upright and follow the permitted yaw interval with a five-degree cap."""
    angles = torch.deg2rad(torch.tensor([current]))
    actions = torch.zeros((1, 7))
    # The 180-degree roll reverses the bolt-frame yaw in the end-effector frame.
    actions[:, 5] = 2.0 * (-target_yaw + 180.0) / 270.0 - 1.0
    env = SimpleNamespace(
        cfg=SimpleNamespace(ctrl=SimpleNamespace(pos_action_bounds=[1.0] * 3, rot_action_bounds=[1.0] * 3)),
        num_envs=1,
        device="cpu",
        last_update_timestamp=0.0,
        _robot=SimpleNamespace(_data=SimpleNamespace(_sim_timestamp=0.0)),
        actions=actions,
        fixed_pos_obs_frame=torch.zeros((1, 3)),
        init_fixed_pos_obs_noise=torch.zeros((1, 3)),
        fingertip_midpoint_pos=torch.zeros((1, 3)),
        fingertip_midpoint_quat=quat_from_euler_xyz(*angles.unbind(dim=1)),
        pos_threshold=torch.ones((1, 3)),
        rot_threshold=torch.deg2rad(torch.full((1, 3), 5.0)),
        generate_ctrl_signals=Mock(),
    )

    ForgeEnv._apply_action(env)

    env.generate_ctrl_signals.assert_called_once()
    actual = env.generate_ctrl_signals.call_args.kwargs["ctrl_target_fingertip_midpoint_quat"]
    expected_angles = torch.deg2rad(torch.tensor([expected]))
    expected_quat = quat_from_euler_xyz(*expected_angles.unbind(dim=1))
    # Quaternion sign does not change the represented orientation.
    sign = torch.sign((actual * expected_quat).sum(dim=-1, keepdim=True))
    torch.testing.assert_close(actual * sign, expected_quat, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(env.delta_yaw, torch.deg2rad(torch.tensor([expected_delta_yaw])), atol=1e-6, rtol=0.0)
