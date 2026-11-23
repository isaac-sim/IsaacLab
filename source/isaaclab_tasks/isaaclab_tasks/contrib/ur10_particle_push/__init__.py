# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 granular-media pushing with declarative Newton MPM coupling."""

import gymnasium as gym

from . import agents

gym.register(
    id="IsaacContrib-UR10-Particle-Push",
    entry_point=f"{__name__}.ur10_particle_push_env:UR10ParticlePushEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_particle_push_env_cfg:UR10ParticlePushEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ParticlePushPPORunnerCfg",
    },
)
