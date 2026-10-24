# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared collision preparation for the contributed steering-wheel asset."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

# These are the graspable wheel surfaces. Their decomposition preserves the
# rim opening and spoke gaps that a single hull would fill.
_STEERING_WHEEL_DECOMPOSED_MESHES = frozenset(
    {
        "sm_steeringwheel_a01_wheel_rim_01",
        "sm_steeringwheel_a01_wheel_spokes_01",
    }
)


def condition_newton_steering_wheel_detail_collisions(object_cfg: RigidObjectCfg) -> None:
    """Hull only non-graspable wheel details that MuJoCo cannot compile as solids.

    The authored wheel uses convex decomposition throughout. Tiny hub and
    quick-release decomposition pieces can have near-zero volume, which MuJoCo
    rejects for solid inertia. The rim and spokes remain decomposed so the
    wheel remains graspable; only detail meshes are switched to convex hulls.
    """
    if object_cfg.spawn is None or getattr(object_cfg.spawn, "_kitless_newton_wheel_collision", False):
        return
    original_spawn = object_cfg.spawn.func

    def _spawn_with_selective_collision(prim_path, cfg, *args, **kwargs):
        from pxr import Usd, UsdGeom

        prim = original_spawn(prim_path, cfg, *args, **kwargs)
        for root_path in sim_utils.find_matching_prim_paths(prim_path):
            root = prim.GetStage().GetPrimAtPath(root_path)
            if not root.IsValid():
                continue
            for mesh_prim in Usd.PrimRange(root):
                if not mesh_prim.IsA(UsdGeom.Mesh) or mesh_prim.GetName() in _STEERING_WHEEL_DECOMPOSED_MESHES:
                    continue
                approximation = mesh_prim.GetAttribute("physics:approximation")
                if approximation and approximation.Get() is not None:
                    approximation.Set("convexHull")
        return prim

    object_cfg.spawn.func = _spawn_with_selective_collision
    object_cfg.spawn._kitless_newton_wheel_collision = True
