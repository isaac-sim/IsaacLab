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

from isaaclab.assets import Articulation
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg as _SceneEntityCfg
from isaaclab.scene import InteractiveScene


class SceneEntityCfg(_SceneEntityCfg):
    """Scene entity configuration with an optional Warp joint mask.

    Notes:
    - `joint_mask` is intended for Warp kernels only.
    """

    joint_mask: wp.array | None = None

    def resolve(self, scene: InteractiveScene):
        # run the stable resolution first (fills joint_ids/body_ids from names/regex)
        super().resolve(scene)

        # Build a Warp joint mask for articulations.
        entity: Articulation = scene[self.name]

        # Pre-allocate a full-length mask (all True for default selection).
        if self.joint_ids == slice(None):
            mask_list = [True] * entity.num_joints
        else:
            mask_list = [False] * entity.num_joints
            for idx in self.joint_ids:
                mask_list[idx] = True
        self.joint_mask = wp.array(mask_list, dtype=wp.bool, device=scene.device)
