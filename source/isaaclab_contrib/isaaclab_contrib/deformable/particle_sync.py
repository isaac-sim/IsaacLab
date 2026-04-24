# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle sync utilities for writing Newton particle positions to USD/Fabric for Kit rendering."""

from __future__ import annotations

import logging
import re

import numpy as np
import warp as wp

from isaaclab.physics import PhysicsManager

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _sync_particle_points(
    fabric_points: wp.fabricarrayarray(dtype=wp.vec3f),
    fabric_world_matrices: wp.fabricarray(dtype=wp.mat44d),
    offsets: wp.fabricarray(dtype=wp.uint32),
    particle_q: wp.array(dtype=wp.vec3f),
    num_points: int,
):
    """Write Newton particle positions into Fabric mesh point arrays as local-frame points.

    Newton stores particle positions in world space in ``state.particle_q``. The Fabric
    ``points`` attribute on a ``UsdGeom.Mesh`` is local-space -- Kit multiplies by the
    mesh prim's resolved ``omni:fabric:worldMatrix`` at render time.

    This kernel inverts the mesh prim's world matrix to convert each world-space particle
    position into local-space before writing.
    """
    i = wp.tid()
    offset = int(offsets[i])

    # Un-transpose Fabric's stored matrix to get the standard homogeneous form
    world_matrix = wp.transpose(wp.mat44f(fabric_world_matrices[i]))
    inv_world_matrix = wp.inverse(world_matrix)

    for j in range(num_points):
        wp_in = particle_q[offset + j]
        # Apply inverse transform to homogeneous point (w=1).
        fabric_points[i][j] = wp.vec3f(
            inv_world_matrix[0, 0] * wp_in[0]
            + inv_world_matrix[0, 1] * wp_in[1]
            + inv_world_matrix[0, 2] * wp_in[2]
            + inv_world_matrix[0, 3],
            inv_world_matrix[1, 0] * wp_in[0]
            + inv_world_matrix[1, 1] * wp_in[1]
            + inv_world_matrix[1, 2] * wp_in[2]
            + inv_world_matrix[1, 3],
            inv_world_matrix[2, 0] * wp_in[0]
            + inv_world_matrix[2, 1] * wp_in[1]
            + inv_world_matrix[2, 2] * wp_in[2]
            + inv_world_matrix[2, 3],
        )


def sync_particles_to_usd() -> None:
    """Write Newton particle_q to Fabric mesh point arrays for Kit viewport rendering.

    For each deformable body whose mesh prim carries a ``newton:particleOffset``
    attribute, this function copies the corresponding slice of ``state_0.particle_q``
    into the Fabric ``points`` array so the Kit viewport reflects the current
    deformation.

    No-op when there is no ``_usdrt_stage``, no simulation state, or no
    deformable bodies registered.
    """
    from isaaclab_newton.physics import NewtonManager

    if (
        NewtonManager._usdrt_stage is None
        or NewtonManager._state_0 is None
        or not NewtonManager._deformable_registry
        or NewtonManager._state_0.particle_q is None
    ):
        return
    if not NewtonManager._particles_dirty:
        return
    pq = NewtonManager._state_0.particle_q
    try:
        import usdrt

        selection = NewtonManager._usdrt_stage.SelectPrims(
            require_attrs=[
                (usdrt.Sdf.ValueTypeNames.Point3fArray, "points", usdrt.Usd.Access.ReadWrite),
                (usdrt.Sdf.ValueTypeNames.UInt, NewtonManager._newton_particle_offset_attr, usdrt.Usd.Access.Read),
                (usdrt.Sdf.ValueTypeNames.Matrix4d, "omni:fabric:worldMatrix", usdrt.Usd.Access.Read),
            ],
            device=str(PhysicsManager._device),
        )
        if selection.GetCount() == 0:
            return
        fabric_points = wp.fabricarrayarray(data=selection, attrib="points", dtype=wp.vec3f)
        fabric_offsets = wp.fabricarray(data=selection, attrib=NewtonManager._newton_particle_offset_attr)
        fabric_world_matrices = wp.fabricarray(data=selection, attrib="omni:fabric:worldMatrix")
        num_points = NewtonManager._deformable_registry[0].particles_per_body
        wp.launch(
            _sync_particle_points,
            dim=selection.GetCount(),
            inputs=[fabric_points, fabric_world_matrices, fabric_offsets, pq, num_points],
            device=PhysicsManager._device,
        )
        NewtonManager._particles_dirty = False
    except Exception as exc:
        logger.debug("[sync_particles_to_usd] %s", exc)


def setup_fabric_particle_sync() -> None:
    """Set up Fabric attributes for deformable particle sync after simulation starts.

    For each deformable registry entry, this function:

    1. Resolves regex prim paths to concrete per-instance paths.
    2. Overwrites the visual mesh topology to match the simulation mesh so
       Fabric particle sync writes the correct number of points.
    3. Creates a per-instance ``newton:particleOffset`` Fabric attribute so
       :func:`sync_particles_to_usd` can find the right slice of ``particle_q``.
    4. Triggers an initial particle sync.
    """
    import usdrt
    from pxr import UsdGeom

    from isaaclab.sim.utils.stage import get_current_stage
    from isaaclab_newton.physics import NewtonManager

    if NewtonManager._usdrt_stage is None:
        NewtonManager._usdrt_stage = get_current_stage(fabric=True)

    stage = get_current_stage()
    for entry in NewtonManager._deformable_registry:
        for inst_idx, offset in enumerate(entry.particle_offsets):
            # Resolve regex pattern to concrete instance path
            resolved_sim = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), entry.sim_mesh_prim_path)
            resolved_sim = re.sub(r"\.\*", str(inst_idx), resolved_sim)
            mesh_prim = stage.GetPrimAtPath(resolved_sim)

            resolved_vis = re.sub(r"(?<=[Ee]nv_)\.\*", str(inst_idx), entry.vis_mesh_prim_path)
            resolved_vis = re.sub(r"\.\*", str(inst_idx), resolved_vis)
            vis_prim = stage.GetPrimAtPath(resolved_vis)
            vis_mesh = UsdGeom.Mesh(vis_prim)

            if not mesh_prim or not mesh_prim.IsValid():
                logger.warning("[setup_fabric_particle_sync] prim not found at %s", resolved_sim)
                continue

            # Overwrite visual mesh topology to match sim mesh so Fabric
            # particle sync can write the correct number of points.
            if mesh_prim.GetTypeName() == "TetMesh":
                tet_mesh = UsdGeom.TetMesh(mesh_prim)
                surface_indices = tet_mesh.GetSurfaceFaceVertexIndicesAttr().Get()
                if surface_indices is None or len(surface_indices) == 0:
                    raise ValueError(
                        "Deformable body has no surface indices on its TetMesh prim; "
                        "cannot sync to visual mesh."
                    )
                vis_mesh.GetPointsAttr().Set(tet_mesh.GetPointsAttr().Get())
                vis_mesh.GetFaceVertexIndicesAttr().Set(np.asarray(surface_indices).flatten())
                vis_mesh.GetFaceVertexCountsAttr().Set([3] * len(surface_indices))
            else:
                sim_mesh = UsdGeom.Mesh(mesh_prim)
                vis_mesh.GetFaceVertexIndicesAttr().Set(sim_mesh.GetFaceVertexIndicesAttr().Get())
                vis_mesh.GetFaceVertexCountsAttr().Set(sim_mesh.GetFaceVertexCountsAttr().Get())

            # Per-instance particle offset so the Fabric sync kernel
            # can find the right slice of particle_q.
            fab_prim = NewtonManager._usdrt_stage.GetPrimAtPath(vis_prim.GetPath().pathString)
            fab_prim.CreateAttribute(NewtonManager._newton_particle_offset_attr, usdrt.Sdf.ValueTypeNames.UInt, True)
            fab_prim.GetAttribute(NewtonManager._newton_particle_offset_attr).Set(offset)

    NewtonManager._mark_particles_dirty()
    if NewtonManager._particle_sync_fn is not None:
        NewtonManager._particle_sync_fn()
