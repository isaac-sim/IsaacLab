# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
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
