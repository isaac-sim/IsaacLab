# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-object data in policy-slot or physical inventory order."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .reset_events import CUBE_COUNT

if TYPE_CHECKING:
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
