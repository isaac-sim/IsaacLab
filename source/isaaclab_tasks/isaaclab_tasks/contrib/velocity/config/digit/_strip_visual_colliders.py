# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Spawner that drops the collision role from Digit's RealSense decoration meshes.

``digit_v4.usd`` applies ``UsdPhysics.CollisionAPI`` to 32 prims, every one of them under a
``/Visual/`` scope of a RealSense camera mount -- ``Glass``, ``USB_C``, ``Case_front``,
``Case_back``, ``Mount``, ``Camera_module``, ``Front_mask``, ``camera_mask`` on each of the four
mounts. They become 32 convex-mesh colliders, 57% of the robot's shapes, while the arms, hips and
rods carry none, and on generated terrain they produced contact forces of 3e7 N on bodies 1.4 m
above the ground. Colliders on a camera's glass and USB connector are an authoring error.

Only the collision role is removed; the meshes still render.

The removal has to happen before the physics backend reads the stage, and it has to stay scoped to
the prims this asset spawned, so it runs as the asset's own spawner rather than as a hook on the
scene builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import spawn_from_usd

if TYPE_CHECKING:
    from pxr import Usd

    from isaaclab.sim.spawners.from_files import UsdFileCfg

_VISUAL_SCOPE = "/Visual/"
_MOUNT_SCOPE = "camera_mount"


def spawn_digit(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn Digit, then clear ``CollisionAPI`` from its camera decoration meshes.

    Args:
        prim_path: Prim path (or expression) to spawn the asset at.
        cfg: Spawner configuration.
        translation: Translation [m] to apply to the spawned prim.
        orientation: Orientation ``(w, x, y, z)`` to apply to the spawned prim.
        **kwargs: Forwarded to :func:`~isaaclab.sim.spawners.from_files.spawn_from_usd`.

    Returns:
        The spawned source prim.
    """
    from pxr import UsdPhysics

    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for child in prim.GetStage().Traverse():
        path = child.GetPath().pathString
        if not path.startswith(prim.GetPath().pathString):
            continue
        if _VISUAL_SCOPE in path and _MOUNT_SCOPE in path and child.HasAPI(UsdPhysics.CollisionAPI):
            child.RemoveAPI(UsdPhysics.CollisionAPI)
    return prim
