# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

# Sentinel marking an unset ``env_pose`
_UNSET_ENV_POSE: Any = object()


@dataclass
class ClonePlan:
    """Flat cloning source of truth.

    Produced by scene planning after representative source prims are assigned. The
    three fields are the same flat replication contract consumed by USD, physics,
    and downstream scene-data providers: each source path maps to the destination
    template at the same index, and :attr:`clone_mask` selects the environments
    populated from that source.
    """

    sources: tuple[str, ...]
    """Source prim paths used for replication."""

    destinations: tuple[str, ...]
    """Destination path templates, one per source path."""

    clone_mask: torch.Tensor
    """Boolean tensor of shape ``[len(sources), num_envs]``;
    ``clone_mask[i, j]`` is ``True`` if env ``j`` was populated from
    :attr:`sources` ``[i]``."""

    env_pose: torch.Tensor = field(default=_UNSET_ENV_POSE)
    """Environment pose tensor of shape ``[num_envs, 7]``;
    ``env_pose[j, :3]`` is the position [m] of env ``j`` and ``env_pose[j, 3:]``
    is its quaternion in xyzw. Defaults to identity at the origin (allocated on
    :attr:`clone_mask`'s device) when omitted at construction."""

    def __post_init__(self) -> None:
        if self.env_pose is _UNSET_ENV_POSE:
            num_envs = int(self.clone_mask.shape[1])
            pose = torch.zeros((num_envs, 7), dtype=torch.float32, device=self.clone_mask.device)
            pose[:, 6] = 1.0  # identity quaternion (xyzw)
            self.env_pose = pose
