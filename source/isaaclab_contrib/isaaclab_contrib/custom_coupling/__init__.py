# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in example of a custom Newton coupling manager."""

import gymnasium as gym

from isaaclab.utils.module import lazy_export

gym.register(
    id="IsaacContrib-Lift-Soft-Franka-Custom-Coupling",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.franka_soft_env_cfg:FrankaSoftCustomCouplingEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.core.lift.config.franka_soft.agents.rsl_rl_ppo_cfg:FrankaDeformablePPORunnerCfg"
        ),
    },
)

lazy_export()
