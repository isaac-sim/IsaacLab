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


def reset_pour_scene(env: FrankaPourEnv, env_ids: torch.Tensor | slice) -> None:
    """Reset the arm to home and refill the source cup with media for ``env_ids``."""
    env.reset_pour_scene(env_ids)
