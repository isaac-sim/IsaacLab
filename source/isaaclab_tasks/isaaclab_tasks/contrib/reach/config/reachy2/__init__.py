# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

##
# Joint Position Control — bimanual (both arms, independent end-effector targets)
##

gym.register(
    id="IsaacContrib-Reach-Reachy2-Bimanual",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_joint_pos_env_cfg:Reachy2BimanualReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Reachy2BimanualReachPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Reach-Reachy2-Bimanual-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bimanual_joint_pos_env_cfg:Reachy2BimanualReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Reachy2BimanualReachPPORunnerCfg",
    },
)
