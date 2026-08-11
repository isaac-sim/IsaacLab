# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-agent inverted double-pendulum balancing environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Pendulum-MARL-Direct",
    entry_point=f"{__name__}.pendulum_marl_env:PendulumMARLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pendulum_marl_env_cfg:PendulumMARLEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_marl_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_marl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_marl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_marl_mappo_cfg.yaml",
    },
)
