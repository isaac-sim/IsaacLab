# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium registration for the G1 (29-DoF body + Dex3 hands) apple pick-and-place environment."""

import gymnasium as gym

gym.register(
    id="IsaacContrib-PickPlace-Apple-G1-Dex3",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apple_pick_place_g1_dex3_env_cfg:G1ApplePickPlaceEnvPresetCfg",
    },
)
