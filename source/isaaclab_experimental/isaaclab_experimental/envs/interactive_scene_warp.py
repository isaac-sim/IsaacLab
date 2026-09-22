# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native interactive scene with env_mask support for reset."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from isaaclab.scene import InteractiveScene


class InteractiveSceneWarp(InteractiveScene):
    """Interactive scene with warp-native env_mask support for reset.

    Extends :class:`InteractiveScene` to accept a boolean warp mask for selective resets,
    avoiding the need to convert between env_ids and masks.
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        """Reset scene entities using either env_ids or a boolean env_mask.

        Args:
            env_ids: The indices of the environments to reset. Defaults to None (all instances).
            env_mask: Boolean warp mask of shape (num_envs,). Defaults to None.
        """
        # -- assets (support env_mask)
        for articulation in self._articulations.values():
            articulation.reset(env_ids, env_mask=env_mask)
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_ids)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_ids, env_mask=env_mask)
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_ids, env_mask=env_mask)
        # -- sensors (no env_mask support)
        for sensor in self._sensors.values():
            sensor.reset(env_ids)
