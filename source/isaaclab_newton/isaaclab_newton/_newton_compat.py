# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility aliases for supported Newton releases."""

import functools
from importlib import metadata
from typing import Any

import newton
import warp as wp

try:
    from newton import ModelFlags
except ImportError:
    from newton.solvers import SolverNotifyFlags as ModelFlags


@wp.kernel
def _clamp_mujoco_warp_collision_count(collision_count: wp.array(dtype=wp.int32), capacity: int):
    """Clamp the broadphase count to the capacity of its output buffers."""
    if collision_count[0] > capacity:
        collision_count[0] = capacity


def patch_mujoco_warp_collision_count() -> None:
    """Prevent MuJoCo Warp 3.11 from reading beyond its broadphase buffers.

    MuJoCo Warp's broadphase counts every candidate but stores only the first
    ``naconmax`` entries. Version 3.11's convex CCD loop consumes the unbounded
    count and can therefore read beyond those buffers. Version 3.12 bounds the
    loop; clamp the shared count before the 3.11 narrowphase for equivalent
    behavior.
    """
    try:
        mujoco_warp_version = metadata.version("mujoco-warp")
    except metadata.PackageNotFoundError:
        return
    if not mujoco_warp_version.startswith("3.11."):
        return

    from mujoco_warp._src import collision_driver

    if getattr(collision_driver._narrowphase, "_isaaclab_bounds_collision_count", False):
        return

    original_narrowphase = collision_driver._narrowphase

    @functools.wraps(original_narrowphase)
    def bounded_narrowphase(model, data, context):
        wp.launch(
            _clamp_mujoco_warp_collision_count,
            dim=1,
            inputs=[data.ncollision, data.naconmax],
            device=data.ncollision.device,
        )
        original_narrowphase(model, data, context)

    bounded_narrowphase._isaaclab_bounds_collision_count = True
    collision_driver._narrowphase = bounded_narrowphase


def refit_shape_bvh(model: Any, state: Any) -> None:
    """Refit a model's shape BVH using the API available in the installed Newton release."""
    if hasattr(model, "bvh_refit_shapes"):
        model.bvh_refit_shapes(state)
    else:
        newton.geometry.refit_bvh_shape(model, state)


__all__ = ["ModelFlags", "patch_mujoco_warp_collision_count", "refit_shape_bvh"]
