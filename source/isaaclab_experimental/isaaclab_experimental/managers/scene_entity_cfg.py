# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental fork of :class:`isaaclab.managers.SceneEntityCfg`.

This adds Warp-only cached selections (e.g. a joint mask) while keeping compatibility
with the stable manager stack (which type-checks against the stable SceneEntityCfg).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.assets import ArticulationCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveScene


class SceneEntityCfg(_SceneEntityCfg):
    """Scene entity configuration with an optional Warp joint mask.

    Notes:
    - `joint_mask` is intended for Warp kernels only.
    """

    joint_mask: wp.array | None = None

    """Integer indices of selected joints — used for subset-sized gathers where a boolean mask
    cannot provide the mapping from output index k to joint index."""
    joint_ids_wp: wp.array | None = None

    """Integer indices of selected bodies — used for subset-sized body gathers."""
    body_ids_wp: wp.array | None = None

    @classmethod
    def from_stable(cls, stable: _SceneEntityCfg) -> SceneEntityCfg:
        """Build a warp scene-entity cfg from a stable one.

        Copies every field declared on the stable cfg; the warp-specific fields
        stay ``None`` and are filled by :meth:`resolve` at scene build time.
        """
        return cls(**{name: getattr(stable, name) for name in _SceneEntityCfg.__dataclass_fields__})

    def resolve(self, scene: InteractiveScene):
        # run the stable resolution first (fills joint_ids/body_ids from names/regex)
        super().resolve(scene)

        entity = scene[self.name]

        # Populate Warp selections from device tensors without a host round trip.
        if isinstance(entity.cfg, ArticulationCfg):
            joint_ids = (
                torch.arange(entity.num_joints, device=scene.device)[self.joint_ids]
                if isinstance(self.joint_ids, slice)
                else self.joint_ids.remainder(entity.num_joints)
            )
            mask = torch.zeros(entity.num_joints, dtype=torch.bool, device=scene.device)
            mask.index_fill_(0, joint_ids, True)
            self.joint_mask = wp.from_torch(mask)
            self.joint_ids_wp = wp.from_torch(joint_ids.to(torch.int32))

        if hasattr(entity, "num_bodies"):
            body_ids = (
                torch.arange(entity.num_bodies, device=scene.device)[self.body_ids]
                if isinstance(self.body_ids, slice)
                else self.body_ids.remainder(entity.num_bodies)
            )
            self.body_ids_wp = wp.from_torch(body_ids.to(torch.int32))
        elif isinstance(self.body_ids, torch.Tensor):
            self.body_ids_wp = wp.from_torch(self.body_ids.to(torch.int32))
