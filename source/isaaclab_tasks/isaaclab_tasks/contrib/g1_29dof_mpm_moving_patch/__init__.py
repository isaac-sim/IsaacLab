# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental G1 moving-patch tasks; registration has no core-manager side effects."""

import gymnasium as gym

for suffix, config in (("", "G1MovingPatchEnvCfg"), ("-Play", "G1MovingPatchEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-Sand-G1-29dof-MPM-MovingPatch" + suffix,
        entry_point=f"{__name__}.mpm_env:G1MovingPatchEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mpm_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1MovingPatchPPORunnerCfg",
        },
    )


for suffix, config in (("", "G1MixedTerrainEnvCfg"), ("-Play", "G1MixedTerrainEnvCfg_PLAY")):
    gym.register(
        id="IsaacContrib-Velocity-G1-29dof-MPM-MovingPatch-MixedTerrain" + suffix,
        entry_point=f"{__name__}.mixed_env:G1MixedTerrainEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.mixed_env_cfg:{config}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:G1MixedTerrainPPORunnerCfg",
        },
    )
