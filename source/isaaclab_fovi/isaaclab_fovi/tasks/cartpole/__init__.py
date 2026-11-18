# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cartpole with a foveated camera."""

import gymnasium as gym

gym.register(
    id="Isaac-Cartpole-Camera-Fovi",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:CartpoleFoviEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents:CartpoleFoviPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
