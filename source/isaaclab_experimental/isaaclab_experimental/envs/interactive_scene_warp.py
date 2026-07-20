# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native interactive scene with env_mask support for reset."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


class InteractiveSceneWarp(InteractiveScene):
    """Interactive scene with warp-native env_mask support for reset.

    Extends :class:`InteractiveScene` to accept a boolean warp mask for selective resets,
    avoiding the need to convert between env_ids and masks.
    """

    def __init__(self, cfg: InteractiveSceneCfg):
        """Initialize the Warp scene and cache non-terrain environment origins.

        Args:
            cfg: Configuration for the interactive scene.
        """
        super().__init__(cfg)
        if self.terrain is None:
            self._env_origins_wp = wp.from_torch(self.env_origins, dtype=wp.vec3f)

    @property
    def env_origins_wp(self) -> wp.array(dtype=wp.vec3f):
        """Cached zero-copy Warp view of environment origins [m], shape ``(num_envs,)``."""
        if self.terrain is not None:
            return self.terrain.env_origins_wp
        return self._env_origins_wp

    def reset(
        self,
        env_ids: Sequence[int] | None = None,
        env_mask: wp.array(dtype=wp.bool) | None = None,
    ) -> None:
        """Reset scene entities using environment IDs or a boolean mask.

        Mask-based calls stay on the Warp data path for every entity that supports
        masks. Surface grippers remain an explicit ID-based host boundary and are
        reset only when :paramref:`env_ids` is also provided. Calls that only pass
        IDs preserve the behavior of :meth:`InteractiveScene.reset`.

        Args:
            env_ids: The indices of the environments to reset. Defaults to None (all instances).
            env_mask: Boolean warp mask of shape (num_envs,). Defaults to None.
        """
        if env_mask is None:
            super().reset(env_ids)
            return
        if env_mask.dtype != wp.bool or env_mask.ndim != 1:
            raise TypeError(f"env_mask must be a one-dimensional Warp boolean array, got {env_mask}.")
        if env_mask.shape[0] != self.cfg.num_envs:
            raise ValueError(f"env_mask must have shape ({self.cfg.num_envs},), received {env_mask.shape}.")
        if env_mask.device != wp.get_device(self.sim.device):
            raise ValueError(f"env_mask must be on device {self.sim.device}, received {env_mask.device}.")

        # -- Warp-capable assets
        for articulation in self._articulations.values():
            articulation.reset(env_mask=env_mask)
        for deformable_object in self._deformable_objects.values():
            deformable_object.reset(env_mask=env_mask)
        for rigid_object in self._rigid_objects.values():
            rigid_object.reset(env_mask=env_mask)
        for rigid_object_collection in self._rigid_object_collections.values():
            rigid_object_collection.reset(env_mask=env_mask)
        # -- Warp-capable sensors
        for sensor in self._sensors.values():
            sensor.reset(env_mask=env_mask)

        if env_ids is not None:
            self.reset_host(env_ids)

    def reset_host(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset ID-based scene entities at an explicit host boundary.

        Args:
            env_ids: The indices of the environments to reset. Defaults to all instances.
        """
        for surface_gripper in self._surface_grippers.values():
            surface_gripper.reset(env_ids)
