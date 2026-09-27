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

        Copies every constructor field declared on the stable cfg; the warp-specific fields
        stay ``None`` and are filled by :meth:`resolve` at scene build time.
        """
        return cls(
            **{
                name: getattr(stable, name)
                for name, field in _SceneEntityCfg.__dataclass_fields__.items()
                if field.init
            }
        )

    def resolve(self, scene: InteractiveScene):
        # run the stable resolution first (fills joint_ids/body_ids from names/regex)
        super().resolve(scene)

        entity = scene[self.name]

        # -- Warp joint mask / ids for articulations
        if isinstance(entity.cfg, ArticulationCfg):
            joint_ids_list = (
                list(range(entity.num_joints)[self.joint_ids])
                if isinstance(self.joint_ids, slice)
                else [index % entity.num_joints for index in self.joint_ids]
            )
            mask_list = [False] * entity.num_joints
            for idx in joint_ids_list:
                mask_list[idx] = True
            self.joint_mask = wp.array(mask_list, dtype=wp.bool, device=scene.device)
            self.joint_ids_wp = wp.array(joint_ids_list, dtype=wp.int32, device=scene.device)

        # -- Warp body ids
        if self.body_ids != slice(None) or hasattr(entity, "num_bodies"):
            num_bodies = len(entity.body_names)
            body_ids_list = (
                list(range(num_bodies)[self.body_ids])
                if isinstance(self.body_ids, slice)
                else [index % num_bodies for index in self.body_ids]
            )
            self.body_ids_wp = wp.array(body_ids_list, dtype=wp.int32, device=scene.device)
