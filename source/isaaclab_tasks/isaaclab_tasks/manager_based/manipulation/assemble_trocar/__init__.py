# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Install Trocar task with G129 + Dex3 robot.

This module registers the Install Trocar task in IsaacLab's gymnasium registry,
allowing it to be discovered and used through IsaacLab's standard task interfaces.
"""

import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g129_dex3_env_cfg:G1AssembleTrocarEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g129_dex3_env_cfg:G1AssembleTrocarEvalEnvCfg",
    },
    disable_env_checker=True,
)
