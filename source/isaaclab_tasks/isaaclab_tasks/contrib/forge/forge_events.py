# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv


def randomize_dead_zone(env: DirectRLEnv, env_ids: torch.Tensor | None):
    env.dead_zone_thresholds = (
        torch.rand((env.num_envs, 6), dtype=torch.float32, device=env.device) * env.default_dead_zone
    )
