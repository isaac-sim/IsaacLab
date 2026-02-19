# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""This sub-module contains the functions that are specific to the locomanipulation environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-PickPlace-Locomanipulation-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.locomanipulation_g1_env_cfg:LocomanipulationG1EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-PickPlace-FixedBaseUpperBodyIK-G1-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.fixed_base_upper_body_ik_g1_env_cfg:FixedBaseUpperBodyIKG1EnvCfg",
    },
    disable_env_checker=True,
)
