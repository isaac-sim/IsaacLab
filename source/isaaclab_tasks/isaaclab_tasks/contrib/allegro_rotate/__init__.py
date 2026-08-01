# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro in-hand free-cylinder rotation and rolling tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Inhand-Rotate-Allegro-v0",
    entry_point=f"{__name__}.allegro_rotate_env:AllegroRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_rotate_env_cfg:AllegroRotateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroRotatePPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Inhand-Roll-Allegro-v0",
    entry_point=f"{__name__}.allegro_rotate_env:AllegroRotateEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_rotate_env_cfg:AllegroRotateEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroRollPPORunnerCfg",
    },
)

gym.register(
    id="IsaacContrib-Inhand-Rotate-Grasp-Allegro-v0",
    entry_point=f"{__name__}.allegro_rotate_grasp_env:AllegroRotateGraspEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_rotate_grasp_env_cfg:AllegroRotateGraspEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroRollPPORunnerCfg",
    },
)
