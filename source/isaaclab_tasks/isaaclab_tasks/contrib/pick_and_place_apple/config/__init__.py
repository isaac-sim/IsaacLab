# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the pick-and-place apple task."""

import gymnasium as gym

# Gymnasium registrations.
gym.register(
    id="IsaacContrib-Pick-And-Place-Apple-H2-Sharpa",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.pick_and_place_apple.config.env_config:H2PnpAppleRLEnvCfg"
        ),
    },
)

gym.register(
    id="IsaacContrib-Pick-And-Place-Apple-H2-Sharpa-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.pick_and_place_apple.config.env_config:H2PnpAppleRLEnvCfg"
        ),
    },
)
