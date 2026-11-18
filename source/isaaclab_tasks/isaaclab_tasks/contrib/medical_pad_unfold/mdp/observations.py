# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms of the medical-pad task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def camera_rgb(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera")) -> torch.Tensor:
    """The camera's latest RGB image as uint8 ``(N, H, W, 3)``."""
    return env.scene[sensor_cfg.name].data.output["rgb"].torch
