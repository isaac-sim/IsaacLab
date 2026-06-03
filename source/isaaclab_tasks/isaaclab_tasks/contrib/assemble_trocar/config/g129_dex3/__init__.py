# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium registrations for the G1 (29-DoF body + Dex3 hands) assemble-trocar environments."""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg:G1AssembleTrocarEnvCfg",
    },
)

gym.register(
    id="Isaac-Assemble-Trocar-G129-Dex3-Eval-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.contrib.assemble_trocar.g129_dex3_env_cfg:G1AssembleTrocarEvalEnvCfg",
    },
)
