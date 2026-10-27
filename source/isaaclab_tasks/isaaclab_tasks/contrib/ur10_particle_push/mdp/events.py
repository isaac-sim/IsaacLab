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


def randomize_push_scene(env: UR10ParticlePushEnv, env_ids: torch.Tensor) -> None:
    """Randomize the robot and single-pile state for selected environments."""
    env.randomize_push_scene(env_ids)
