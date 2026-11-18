# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kuka-Allegro lift and reorientation with a foveated base camera."""

import gymnasium as gym

for task, cfg in (("Lift", "KukaAllegroLiftFoviEnvCfg"), ("Reorient", "KukaAllegroReorientFoviEnvCfg")):
    gym.register(
        id=f"Isaac-{task}-KukaAllegro-Camera-Fovi",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.env_cfg:{cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.agents:KukaAllegroFoviPPORunnerCfg",
            "default_agent": "rsl_rl",
        },
    )
