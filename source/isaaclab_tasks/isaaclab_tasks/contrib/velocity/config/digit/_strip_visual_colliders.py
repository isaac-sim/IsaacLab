# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Remove CollisionAPI from Digit's RealSense *visual* decoration meshes before the model is built.

digit_v4.usd authors UsdPhysics.CollisionAPI on 32 prims, every one of them under a `/Visual`
scope of a RealSense camera mount (Glass, USB_C, Case_front, Case_back, Mount, Camera_module,
Front_mask, camera_mask, x4 mounts). They become 32 CONVEX_MESH colliders -- 57% of the robot's 56
shapes -- while the arms, hips and rods carry none. Colliders on a camera's glass and USB
connector are an authoring error, not a modelling choice.

The strip has to happen before NewtonManager.instantiate_builder_from_stage() reads the stage;
clearing shape_flags afterwards is too late because MuJoCo has already compiled the geoms.
"""
from __future__ import annotations

_INSTALLED = False
PATTERN = "/Visual/"


def install(pattern: str = PATTERN, require: str = "camera_mount"):
    global _INSTALLED
    if _INSTALLED:
        return
    # Multi-env scenes build the Newton model through the cloner, not
    # NewtonManager.instantiate_builder_from_stage, so hook the function that actually reads the
    # stage into a ModelBuilder.
    from isaaclab_newton.cloner import replicate as _rep

    original = _rep._build_newton_builder_from_mapping

    def patched(stage, *args, **kwargs):
        from pxr import UsdPhysics

        removed = []
        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            if pattern in path and require in path and prim.HasAPI(UsdPhysics.CollisionAPI):
                prim.RemoveAPI(UsdPhysics.CollisionAPI)
                removed.append(path)
        return original(stage, *args, **kwargs)

    _rep._build_newton_builder_from_mapping = patched
    _INSTALLED = True
