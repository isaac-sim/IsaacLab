# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Separated berry layout and summed two-way gripper reactions."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..pick_berries_env_cfg import BerryPickEnvCfg
    from .runtime import BerryRuntime

BERRY_NAMES = ("raspberry", "blackberry", "blueberry", "strawberry")


def berry_configs(cfg: BerryPickEnvCfg) -> list[BerryPickEnvCfg]:
    """Create per-berry configurations with separated offsets [m] about the plate center."""
    if cfg.berry != "all":
        return [cfg]
    if cfg.berry_asset_path:
        raise ValueError("A custom --asset can only be used with a single berry")
    if cfg.target_berry not in BERRY_NAMES:
        raise ValueError(f"Unknown target berry: {cfg.target_berry}")
    configs = []
    for name, (dx, dy) in zip(BERRY_NAMES, ((-0.025, -0.025), (0.025, -0.025), (-0.025, 0.025), (0.025, 0.025))):
        item = cfg.copy()
        item.berry = name
        item.berry_position = (cfg.berry_position[0] + dx, cfg.berry_position[1] + dy, cfg.berry_position[2])
        configs.append(item)
    return configs


def advance_berries(berries: Iterable[BerryRuntime], robot: Articulation) -> None:
    """Advance a nonempty tissue collection and apply its summed finger force [N] once."""
    total = np.zeros((2, 3), np.float32)
    for berry in berries:
        berry.advance(robot, apply_force=False)
        total += berry.last_force
        fingers = berry.fingers
    force = torch.as_tensor(total[None], dtype=torch.float32, device=robot.device)
    robot.set_external_force_and_torque(force, torch.zeros_like(force), body_ids=fingers, is_global=True)
