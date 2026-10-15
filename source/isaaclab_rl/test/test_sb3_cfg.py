# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the Stable-Baselines3 configuration parser."""

import pytest

from isaaclab_rl.sb3 import process_sb3_cfg

pytestmark = pytest.mark.unit


def test_linear_schedules_keep_their_own_initial_values() -> None:
    cfg = process_sb3_cfg(
        {"learning_rate": "lin_3e-4", "clip_range": "lin_0.2", "clip_range_vf": "lin_0.5", "n_timesteps": 10},
        num_envs=1,
    )

    for key, initial_value in (("learning_rate", 3e-4), ("clip_range", 0.2), ("clip_range_vf", 0.5)):
        assert cfg[key](1.0) == initial_value
        assert cfg[key](0.5) == 0.5 * initial_value
