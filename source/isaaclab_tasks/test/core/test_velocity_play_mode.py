# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from isaaclab_tasks.core.velocity.config.cassie.rough_env_cfg import CassieRoughEnvCfg
from isaaclab_tasks.core.velocity.config.g1.rough_env_cfg import G1RoughEnvCfg
from isaaclab_tasks.core.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg


@pytest.mark.parametrize(
    "cfg_type,lin_vel_x,episode_length_s",
    [
        (CassieRoughEnvCfg, (0.7, 1.0), 20.0),
        (G1RoughEnvCfg, (1.0, 1.0), 40.0),
        (H1RoughEnvCfg, (1.0, 1.0), 40.0),
    ],
)
def test_velocity_play_mode_applies_task_overrides(cfg_type, lin_vel_x, episode_length_s):
    cfg = cfg_type()

    cfg.play_mode()

    assert cfg.scene.num_envs == 50
    assert cfg.observations.policy.enable_corruption is False
    assert cfg.episode_length_s == episode_length_s
    assert cfg.commands.base_velocity.ranges.lin_vel_x == lin_vel_x
    assert cfg.commands.base_velocity.ranges.lin_vel_y == (0.0, 0.0)
    assert cfg.commands.base_velocity.ranges.heading == (0.0, 0.0)
