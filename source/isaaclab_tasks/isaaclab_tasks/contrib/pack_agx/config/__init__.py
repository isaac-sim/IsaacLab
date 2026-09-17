# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the AGX Orin packing task."""

import gymnasium as gym

# Gymnasium registrations.
gym.register(
    id="IsaacContrib-Pack-AGX-Orin-H2-Sharpa",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.pack_agx.config.env_config:H2PackAgxOrinRLEnvCfg",
    },
)

gym.register(
    id="IsaacContrib-Pack-AGX-Orin-H2-Sharpa-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.pack_agx.config.env_config:H2PackAgxOrinRLEnvCfg",
    },
)
