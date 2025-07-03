# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import gymnasium as gym

from .pick_place_locomanipulation.fixed_base_upper_body_ik_g1_env_cfg import FixedBaseUpperBodyIKG1EnvCfg
from .pick_place_locomanipulation.fixed_base_upper_body_ik_gr1t2_env_cfg import FixedBaseUpperBodyIKGR1T2EnvCfg
from .pick_place_locomanipulation.locomanipulation_g1_env_cfg import LocomanipulationG1EnvCfg

gym.register(
    id="Isaac-PickPlace-FixedBaseUpperBodyIK-GR1T2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FixedBaseUpperBodyIKGR1T2EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": LocomanipulationG1EnvCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FixedBaseUpperBodyIKG1EnvCfg,
    },
    disable_env_checker=True,
)
