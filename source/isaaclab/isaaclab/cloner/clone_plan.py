# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ClonePlan:
    """Per-group mapping from prototype prims to per-environment clones.

    Produced by :func:`~isaaclab.cloner.clone_from_template` for each prototype group it
    discovers under the template root. Lets downstream consumers (e.g. mesh samplers,
    ray-cast sensors) read prototype geometry once and scatter to environments via
    :attr:`clone_mask` instead of walking per-env USD paths.

    Attributes are population-time invariants and the dataclass is frozen. Hash and
    equality operate on :attr:`dest_template` only (the natural identity — it is the key
    in :attr:`SimulationContext.get_clone_plans`); the mutable list/tensor fields are
    excluded since ``torch.Tensor`` is not hashable and structural equality is rarely the
    semantics consumers want.
    """

    dest_template: str
    """Destination path template for this group, e.g. ``"/World/envs/env_{}/Object"``."""

    prototype_paths: list[str] = field(hash=False, compare=False)
    """Prototype prim paths in this group, e.g.
    ``["/World/template/Object/proto_asset_0", "/World/template/Object/proto_asset_1"]``."""

    clone_mask: torch.Tensor = field(hash=False, compare=False)
    """Boolean tensor of shape ``[num_prototypes_in_group, num_envs]``;
    ``clone_mask[i, j]`` is ``True`` iff env ``j`` was populated from
    :attr:`prototype_paths` ``[i]``. Each column sums to exactly one."""
