# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Manager-based ANYmal-C example using the contributed trail terrains."""

import gymnasium as gym

from isaaclab_tasks.contrib.velocity.config.anymal_c import agents

from .anymal_trail_env_cfg import AnymalCTrailEnvCfg

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Trail-AnymalC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.anymal_trail_env_cfg:AnymalCTrailEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

__all__ = ["AnymalCTrailEnvCfg"]
