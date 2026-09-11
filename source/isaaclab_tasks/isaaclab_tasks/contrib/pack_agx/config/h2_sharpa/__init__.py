# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium registrations for the H2 + Sharpa AGX Orin packing environments."""

import gymnasium as gym

for task_id in ("IsaacContrib-Pack-AGX-Orin-H2-Sharpa", "IsaacContrib-Pack-AGX-Orin-H2-Sharpa-Eval"):
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "isaaclab_tasks.contrib.pack_agx.h2_sharpa_env_cfg:H2PackAgxOrinRLEnvCfg",
        },
    )
