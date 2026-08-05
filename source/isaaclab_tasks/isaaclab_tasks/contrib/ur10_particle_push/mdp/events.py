# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for UR10 particle pushing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..ur10_particle_push_env import UR10ParticlePushEnv


def reset_push_scene(env: UR10ParticlePushEnv, env_ids: torch.Tensor) -> None:
    """Reset the robot and fixed granular payload for selected environments."""
    env.reset_push_scene(env_ids)
