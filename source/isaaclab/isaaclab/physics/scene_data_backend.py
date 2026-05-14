# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend interface and data formats for the scene data provider.

These types live in :mod:`isaaclab.physics` rather than
:mod:`isaaclab.scene.scene_data_provider` so that physics backends
(``isaaclab_physx``, ``isaaclab_newton``) can subclass
:class:`SceneDataBackend` without pulling :mod:`isaaclab.scene` into the
``AppLauncher`` pre-launch import chain. ``AppLauncher._create_app`` pops
``*lab*`` modules from ``sys.modules`` during Kit init and any submodule
imported during that window ends up orphaned from its parent's
``__dict__`` after restoration.
"""

from __future__ import annotations

import warp as wp


class SceneDataFormat:
    @wp.struct
    class Vec3_Quat:
        positions: wp.array(dtype=wp.vec3f) = None
        orientations: wp.array(dtype=wp.quatf) = None

    @wp.struct
    class Vec3_Matrix33:
        positions: wp.array(dtype=wp.vec3f) = None
        orientations: wp.array(dtype=wp.mat33f) = None

    @wp.struct
    class Transform:
        transforms: wp.array(dtype=wp.transformf) = None

    @wp.struct
    class Matrix44:
        matrices: wp.array(dtype=wp.mat44f) = None


class SceneDataBackend:
    @property
    def transforms(
        self,
    ) -> (
        SceneDataFormat.Vec3_Quat | SceneDataFormat.Transform | SceneDataFormat.Matrix44 | SceneDataFormat.Vec3_Matrix33
    ):
        """Return the sim backends transforms as one of the SceneDataFormat structs."""
        raise NotImplementedError

    @property
    def transform_count(self) -> int:
        """Return the number of transforms in the sim backend."""
        raise NotImplementedError

    @property
    def transform_paths(self) -> list[str]:
        """Return the paths for each transform."""
        raise NotImplementedError
