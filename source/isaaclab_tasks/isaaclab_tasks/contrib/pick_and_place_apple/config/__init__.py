# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task assets and H2 + Sharpa embodiment exports for the pick-and-place apple task."""

from __future__ import annotations

import gymnasium as gym

from isaaclab_tasks.contrib.rlinf_assets import NUREC_ASSET_ROOT, PROP_ASSET_ROOT
from .camera_config import CameraBaseCfg, CameraPresets
from .metadata import H2_DEFAULT_JOINT_POS, H2_PNP_APPLE_CUSTOM_JOINT_POS
from .robot_config import H2RobotPresets, make_h2_sharpa_cfg

TABLE_USD = f"{PROP_ASSET_ROOT}/Assets/Table256/Table256_cloth.usd"
APPLE_USD = f"{PROP_ASSET_ROOT}/Assets/Apple033/Apple033.usd"
PLATE_USD = f"{PROP_ASSET_ROOT}/Assets/SimReady_Furniture/plate_large/plate_large_rigid.usd"
BACKGROUND_USD = f"{NUREC_ASSET_ROOT}/IMG_6246_nurec_aligned_scaled.usdz"

# Task-specific start pose for the apple task.
H2_PNP_APPLE_INIT_POS: tuple[float, float, float] = (-0.95, 0.0, 1.05)
H2_PNP_APPLE_INIT_ROT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

__all__ = [
    "APPLE_USD",
    "BACKGROUND_USD",
    "CameraBaseCfg",
    "CameraPresets",
    "H2_DEFAULT_JOINT_POS",
    "H2_PNP_APPLE_CUSTOM_JOINT_POS",
    "H2_PNP_APPLE_INIT_POS",
    "H2_PNP_APPLE_INIT_ROT",
    "H2RobotPresets",
    "PLATE_USD",
    "TABLE_USD",
    "make_h2_sharpa_cfg",
]


# Gymnasium registrations.
gym.register(
    id="IsaacContrib-Pick-And-Place-Apple-H2-Sharpa",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.pick_and_place_apple.config.env_config:H2PnpAppleRLEnvCfg"
        ),
    },
)

gym.register(
    id="IsaacContrib-Pick-And-Place-Apple-H2-Sharpa-Eval",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.contrib.pick_and_place_apple.config.env_config:H2PnpAppleRLEnvCfg"
        ),
    },
)
