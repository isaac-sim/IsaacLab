# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Experimental fork of :class:`isaaclab.managers.SceneEntityCfg`.

This adds Warp-only cached selections (e.g. a joint mask) while keeping compatibility
with the stable manager stack (which type-checks against the stable SceneEntityCfg).
"""

from __future__ import annotations

import warp as wp

from isaaclab.assets.articulation.base_articulation import BaseArticulation
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _SceneEntityCfg
from isaaclab.scene import InteractiveScene


class SceneEntityCfg(_SceneEntityCfg):
    """Scene entity configuration with an optional Warp joint mask.

    Notes:
    - `joint_mask` is intended for Warp kernels only.
    """

    # TODO(jichuanh): review the necessity of these two attributes.
    joint_mask: wp.array | None = None
    joint_ids_wp: wp.array | None = (
        None  # Needed for subset-sized outputs/gathers (len(selected)); mask can't map k→joint/order.
    )

    def resolve(self, scene: InteractiveScene):
        # run the stable resolution first (fills joint_ids/body_ids from names/regex)
        super().resolve(scene)

        # Build a Warp joint mask for articulations only.
        entity = scene[self.name]
        if not isinstance(entity, BaseArticulation):
            return

        # Pre-allocate a full-length mask (all True for default selection).
        if self.joint_ids == slice(None):
            joint_ids_list = list(range(entity.num_joints))
            mask_list = [True] * entity.num_joints
        else:
            joint_ids_list = list(self.joint_ids)
            mask_list = [False] * entity.num_joints
            for idx in joint_ids_list:
                mask_list[idx] = True
        self.joint_mask = wp.array(mask_list, dtype=wp.bool, device=scene.device)
        self.joint_ids_wp = wp.array(joint_ids_list, dtype=wp.int32, device=scene.device)
