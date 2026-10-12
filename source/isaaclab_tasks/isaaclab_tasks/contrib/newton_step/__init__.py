# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A reach task whose operational-space controller runs inside the captured Newton step."""

import gymnasium as gym

from isaaclab_tasks.core.reach.config.franka import agents as franka_reach_agents

gym.register(
    id="IsaacContrib-NewtonStep-Reach-Franka-NewtonOSC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_newton_osc_env_cfg:FrankaReachNewtonOSCEnvCfg",
        "rsl_rl_cfg_entry_point": f"{franka_reach_agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
