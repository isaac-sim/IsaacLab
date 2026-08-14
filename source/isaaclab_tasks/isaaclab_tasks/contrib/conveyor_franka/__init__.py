# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-selectable conveyor scene with a Franka robot."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-Conveyor-Franka-Newton-v0",
    entry_point=f"{__name__}.conveyor_franka_env:ConveyorFrankaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conveyor_franka_env_cfg:ConveyorFrankaEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ConveyorFrankaPPORunnerCfg",
    },
)

gym.register(
    # The native PhysxSurfaceVelocityAPI path is intentionally CPU-only. Keep
    # that execution contract visible in the public task ID so a CUDA launch is
    # never mistaken for a supported configuration.
    id="IsaacContrib-Conveyor-Franka-PhysX-CPU-v0",
    entry_point=f"{__name__}.conveyor_franka_env:ConveyorFrankaEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.conveyor_franka_physx_env_cfg:ConveyorFrankaPhysxEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ConveyorFrankaPPORunnerCfg",
    },
)
