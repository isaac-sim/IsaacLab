# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable physical parcel identities behind the checkpoint's four observation slots."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from .mdp.reset_events import CUBE_COUNT

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def cube_values(env: ManagerBasedRLEnv, attribute: str, *, all_cubes: bool = False) -> torch.Tensor:
    """Gather a rigid-object data attribute in policy-slot or complete physical-inventory order.

    Values retain the attribute's units and world frame. The result has shape
    ``(num_envs, num_slots_or_parcels, attribute_size)``. Tasks without a parcel pool
    retain the original four fixed cube identities.
    """
    pool = getattr(env, "conveyor_cube_pool", None)
    assets = pool.assets if pool is not None else tuple(env.scene[f"cube_{i}"] for i in range(CUBE_COUNT))
    values = torch.stack([getattr(asset.data, attribute).torch for asset in assets], dim=1)
    if pool is not None and not all_cubes:
        values = values.gather(1, pool.slot_ids[..., None].expand(-1, -1, values.shape[-1]))
    return values


class ConveyorCubePool:
    """Bind four policy slots to a larger pool without moving or copying physical bodies.

    Assignments are independent per environment. Local parcels keep their slots;
    the active grasp is additionally pinned even if it moves outside the workcell.
    """

    def __init__(self, assets: tuple[RigidObject, ...], num_envs: int, device: str) -> None:
        if len(assets) < CUBE_COUNT:
            raise ValueError("The conveyor policy requires at least four physical parcels.")
        self.assets = assets
        self.slot_ids = torch.arange(CUBE_COUNT, device=device).repeat(num_envs, 1)
        self.assignment_counts = torch.zeros((num_envs, len(assets)), device=device, dtype=torch.long)
        self.assignment_counts[:, :CUBE_COUNT] = 1
        self.transfer_counts = torch.zeros_like(self.assignment_counts)

    def reset(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        """Restore reset-recipe slot identities for selected environments."""
        self.slot_ids[env_ids] = torch.arange(CUBE_COUNT, device=self.slot_ids.device)

    def refresh(
        self,
        positions: torch.Tensor,
        local: torch.Tensor,
        candidates: torch.Tensor,
        target_slots: torch.Tensor,
        pinned: torch.Tensor,
    ) -> torch.Tensor:
        """Assign arriving parcels to remote slots, preserving all local and pinned identities.

        Args:
            positions: Physical parcel positions in the workspace [m], shape ``(N, P, 3)``.
            local: Parcels still in the manipulation region, shape ``(N, P)``.
            candidates: Parcels eligible for pickup, shape ``(N, P)``.
            target_slots: Current command's policy slot, shape ``(N,)``.
            pinned: Whether the active parcel must retain its slot, shape ``(N,)``.

        Returns:
            Environments whose slot assignments changed, shape ``(N,)``.
        """
        mapped = torch.zeros_like(candidates).scatter_(1, self.slot_ids, True)
        available = candidates & ~mapped
        reusable = ~local.gather(1, self.slot_ids)
        reusable.scatter_(1, target_slots[:, None], ~pinned[:, None] & reusable.gather(1, target_slots[:, None]))
        changed = torch.zeros_like(pinned)
        # Prefer parcels assigned less often, then the arrival with more belt travel remaining.
        priority = positions[..., 0] - 10.0 * self.assignment_counts
        for slot in range(CUBE_COUNT):
            eligible = reusable[:, slot] & available.any(dim=1)
            rows = eligible.nonzero(as_tuple=False).flatten()
            if not rows.numel():
                continue
            choices = torch.where(available, priority, -torch.inf).argmax(dim=1)[rows]
            self.slot_ids[rows, slot] = choices
            self.assignment_counts[rows, choices] += 1
            available[rows, choices] = False
            changed[rows] = True
        return changed

    def record_transfers(self, env_ids: torch.Tensor, target_slots: torch.Tensor) -> None:
        """Credit stable placements to physical parcels before the command selects its next slot."""
        physical_ids = self.slot_ids[env_ids, target_slots]
        self.transfer_counts[env_ids, physical_ids] += 1
