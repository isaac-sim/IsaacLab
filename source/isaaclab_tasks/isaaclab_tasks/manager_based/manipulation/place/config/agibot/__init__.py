# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

##
# Register Gym environments.
##

##
# Agibot Right Arm: place toy2box task, with RmpFlow
##
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_toy2box_rmp_rel_env_cfg:RmpFlowAgibotPlaceToy2BoxEnvCfg",
    },
    disable_env_checker=True,
)

##
# Agibot Left Arm: place upright mug task, with RmpFlow
##
gym.register(
    id="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_upright_mug_rmp_rel_env_cfg:RmpFlowAgibotPlaceUprightMugEnvCfg",
    },
    disable_env_checker=True,
)
