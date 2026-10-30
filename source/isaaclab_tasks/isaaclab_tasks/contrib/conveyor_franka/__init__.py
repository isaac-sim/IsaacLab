# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Four-cube racetrack transfer with Newton."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Conveyor-Racetrack-Transfer-PhysX-CPU-v0",
    entry_point=f"{__name__}.conveyor_franka_physx_env:ConveyorFrankaPhysxEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conveyor_franka_physx_env_cfg:ConveyorFrankaPhysxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ConveyorFrankaPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Conveyor-Racetrack-Transfer-v0",
    entry_point=f"{__name__}.conveyor_franka_env:ConveyorFrankaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conveyor_franka_env_cfg:ConveyorFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ConveyorFrankaPPORunnerCfg",
    },
)
