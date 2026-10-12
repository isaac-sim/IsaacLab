# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-object data in the policy's four fixed cube identities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .reset_events import CUBE_COUNT

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_values(env: ManagerBasedRLEnv, attribute: str, *, all_cubes: bool = False) -> torch.Tensor:
    """Gather cube data in identity order, with shape (num_envs, four, attribute_size)."""
    return torch.stack([getattr(env.scene[f"cube_{i}"].data, attribute).torch for i in range(CUBE_COUNT)], dim=1)
