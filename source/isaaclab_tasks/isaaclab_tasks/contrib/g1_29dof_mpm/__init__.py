# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""G1 29-DoF locomotion over granular media simulated with a coupled MJWarp/MPM solver."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="IsaacContrib-Velocity-Sand-G1-29dof-MPM",
    entry_point=f"{__name__}.g1_mpm_env:G1MPMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_mpm_env_cfg:G1MPMEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MPMPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Sand-G1-29dof-MPM-Play",
    entry_point=f"{__name__}.g1_mpm_env:G1MPMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_mpm_env_cfg:G1MPMEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MPMPPORunnerCfg",
    },
)

for suffix, cfg_name in (("", "G1RoughMPMEnvCfg"), ("-Play", "G1RoughMPMEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Velocity-Sand-Rough-G1-29dof-MPM{suffix}",
        entry_point=f"{__name__}.g1_mpm_env:G1MPMEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_mpm_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MPMPPORunnerCfg",
        },
    )

for suffix, cfg_name in (("", "G1TiltingMPMEnvCfg"), ("-Play", "G1TiltingMPMEnvCfg_PLAY")):
    gym.register(
        id=f"IsaacContrib-Balance-Sand-G1-29dof-MPM-Tilting{suffix}",
        entry_point=f"{__name__}.g1_mpm_env:G1TiltingMPMEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.g1_mpm_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1MPMPPORunnerCfg",
        },
    )
