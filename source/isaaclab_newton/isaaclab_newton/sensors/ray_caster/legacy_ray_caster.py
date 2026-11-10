# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy Newton adapters for Warp-mesh ray-caster implementations."""

from __future__ import annotations

# pyright: reportInvalidTypeForm=none, reportPrivateUsage=none
import warnings

from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster import BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster.base_multi_mesh_ray_caster_camera import BaseMultiMeshRayCasterCamera
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.base_ray_caster_camera import BaseRayCasterCamera

from .newton_raycast_sensor import _NewtonRayCasterPoseMixin


class LegacyRayCaster(_NewtonRayCasterPoseMixin, BaseRayCaster):
    """Legacy Newton ray caster that queries one configured Warp mesh."""


class LegacyRayCasterCamera(_NewtonRayCasterPoseMixin, BaseRayCasterCamera):
    """Legacy Newton ray-caster camera backed by configured Warp meshes."""


class LegacyMultiMeshRayCaster(_NewtonRayCasterPoseMixin, BaseMultiMeshRayCaster):
    """Legacy Newton ray caster for configured static and dynamic Warp meshes."""


class LegacyMultiMeshRayCasterCamera(_NewtonRayCasterPoseMixin, BaseMultiMeshRayCasterCamera):
    """Legacy Newton ray-caster camera for configured Warp meshes."""


def _warn_legacy_alias(old_name: str, new_name: str) -> None:
    """Warn when a pre-rename Newton backend class is constructed directly."""
    warnings.warn(
        f"isaaclab_newton.sensors.{old_name} is deprecated; use isaaclab_newton.sensors.{new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class RayCasterCamera(LegacyRayCasterCamera):
    """Deprecated alias for :class:`LegacyRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("RayCasterCamera", "LegacyRayCasterCamera")
        super().__init__(cfg)


class MultiMeshRayCaster(LegacyMultiMeshRayCaster):
    """Deprecated alias for :class:`LegacyMultiMeshRayCaster`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCaster", "LegacyMultiMeshRayCaster")
        super().__init__(cfg)


class MultiMeshRayCasterCamera(LegacyMultiMeshRayCasterCamera):
    """Deprecated alias for :class:`LegacyMultiMeshRayCasterCamera`."""

    def __init__(self, cfg):
        _warn_legacy_alias("MultiMeshRayCasterCamera", "LegacyMultiMeshRayCasterCamera")
        super().__init__(cfg)
