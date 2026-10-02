# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OneRobotics A1 bimanual reach environment registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Reach-OneRobotics-A1-Bimanual",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:OneRoboticsA1BimanualReachEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:OneRoboticsA1BimanualReachPPORunnerCfg"
        ),
        "default_agent": "rsl_rl",
    },
)
