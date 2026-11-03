# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck walking and contributed Pollen skill recipes.

The original flat/rough walking registrations retain their behavior. Separate ``-Pollen`` walking
recipes and the recovery, seated, skating, kicking and distilled VelStand tasks are discovered by
Isaac Lab's standard task importer. Select supported gearbox-play variants with ``presets=backlash``.
See ``UPSTREAM.md`` and the MicroDuck training guides for source provenance and validation limits.
"""

import gymnasium as gym

gym.register(
    id="IsaacContrib-Velocity-Flat-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MicroDuckVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckSymmetricPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Velocity-Rough-MicroDuck",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:MicroDuckVelocityRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckRoughSymmetricPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:MicroDuckRoughSymmetricPPORunnerCfg",
    },
)
