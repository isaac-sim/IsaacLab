# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Install Trocar task with G129 + Dex3 robot.

This module registers the Install Trocar task in IsaacLab's gymnasium registry,
allowing it to be discovered and used through IsaacLab's standard task interfaces.

The environment configuration is imported from RLinf's task definitions to avoid
code duplication while enabling IsaacLab-native task discovery.
"""

import gymnasium as gym

from . import g129_dex3_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-RLinf-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g129_dex3_env_cfg.G1AssembleTrocarEnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-RLinf-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": g129_dex3_env_cfg.G1AssembleTrocarEvalEnvCfg,
    },
    disable_env_checker=True,
)
