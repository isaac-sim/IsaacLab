# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg


def create_mpm_particle_visualization(
    prim_paths: Sequence[str],
    positions: np.ndarray,
    widths: np.ndarray,
    color: Sequence[float],
    visual_material: VisualMaterialCfg | None = None,
) -> list[str]:
    """Create one ``UsdGeom.Points`` prim per environment for USD-stage MPM particle rendering.

    The created prims are static USD containers: per-frame position updates are
    handled by :meth:`isaaclab_newton.physics.NewtonManager.sync_particles_to_usd`
    for prims registered via
    :meth:`isaaclab_newton.physics.NewtonManager.register_particle_visual_prim`.

    Args:
        prim_paths: ``Points`` prim paths, one per environment.
        positions: Initial world-frame particle positions [m], shape
            ``(num_envs, particles_per_env, 3)``.
        widths: Particle display widths (diameters) [m], one per particle.
        color: Fallback RGB display color of the particles.
        visual_material: Optional visual material bound to every particle cloud.

    Returns:
        The created ``Points`` prim paths, one per environment.
    """
    from pxr import Gf, Sdf, UsdGeom, Vt  # noqa: PLC0415

    import isaaclab.sim as sim_utils

    stage = sim_utils.get_current_stage()
    prim_paths = list(prim_paths)
    if len(prim_paths) != positions.shape[0]:
        raise ValueError(
            "Expected one particle visualization prim path per environment, "
            f"got {len(prim_paths)} paths for {positions.shape[0]} environments."
        )
    points_prims = [UsdGeom.Points.Define(stage, path) for path in prim_paths]

    widths_vt = Vt.FloatArray.FromNumpy(np.ascontiguousarray(widths, dtype=np.float32))
    color_vt = Vt.Vec3fArray([Gf.Vec3f(*(float(value) for value in color))])
    with Sdf.ChangeBlock():
        for env_idx, points in enumerate(points_prims):
            points.SetResetXformStack(True)
            points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(positions[env_idx]))
            points.CreateWidthsAttr(widths_vt)
            points.CreateDisplayColorAttr(color_vt)

    if visual_material is not None:
        for path in prim_paths:
            material_scope_path = Sdf.Path(path).GetParentPath().AppendChild("Looks")
            UsdGeom.Scope.Define(stage, material_scope_path)
            material_path = material_scope_path.AppendChild("visualMaterial")
            visual_material.func(str(material_path), visual_material)
            if stage.GetPrimAtPath(material_path).IsValid():
                sim_utils.bind_visual_material(path, str(material_path), stage=stage)

    return prim_paths
