# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event (reset) terms for the two-cup pour task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def reset_pour_scene(env: FrankaPourEnv, env_ids: torch.Tensor) -> None:
    """Reset the arm to home and refill the source cup with media for ``env_ids``."""
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(list(env_ids), device=env.device, dtype=torch.long)
    env.reset_pour_scene(env_ids.long())
