# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def create_mpm_particle_visualization(
    prim_path: str,
    positions: np.ndarray,
    widths: np.ndarray,
    color: Sequence[float],
) -> list[str]:
    """Create one ``UsdGeom.Points`` prim per environment for Kit MPM particle rendering.

    The created prims are static USD containers: per-frame position updates are
    handled by :meth:`isaaclab_newton.physics.NewtonManager.sync_particles_to_usd`
    for prims registered via
    :meth:`isaaclab_newton.physics.NewtonManager.register_particle_visual_prim`.

    Args:
        prim_path: Base prim path; one ``Points`` prim is created per environment
            at ``{prim_path}/env_{idx}``.
        positions: Initial world-frame particle positions [m], shape
            ``(num_envs, particles_per_env, 3)``.
        widths: Particle display widths (diameters) [m], one per particle.
        color: RGB display color of the particles.

    Returns:
        The created ``Points`` prim paths, one per environment.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    prim_paths = [f"{prim_path}/env_{env_idx}" for env_idx in range(positions.shape[0])]
    points_prims = [UsdGeom.Points.Define(stage, path) for path in prim_paths]

    widths_vt = Vt.FloatArray.FromNumpy(np.ascontiguousarray(widths, dtype=np.float32))
    color_vt = Vt.Vec3fArray([Gf.Vec3f(*(float(value) for value in color))])
    with Sdf.ChangeBlock():
        for env_idx, points in enumerate(points_prims):
            points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions[env_idx]))
            points.CreateWidthsAttr(widths_vt)
            points.CreateDisplayColorAttr(color_vt)

    return prim_paths
