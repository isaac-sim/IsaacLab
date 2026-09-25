# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Build Newton rendering geometry from imported deformable prototypes."""

from __future__ import annotations

import warp as wp
from newton import ModelBuilder

from isaaclab.scene_data.deformable_discovery import DeformableStageEntry


def add_shadow_deformables_to_builder(builder: ModelBuilder, entries: list[DeformableStageEntry]) -> dict[str, int]:
    """Build declared visual geometry and return its native particle destinations.

    Args:
        builder: Render-only Newton model under construction.
        entries: Imported geometry expanded to this representation's destination instances.

    Returns:
        Exact visual mesh paths mapped to their first render-particle indices.
        SDP supplies the corresponding world-space points after physics initialization.
    """
    offsets = {}
    for entry in entries:
        offsets[entry.vis_mesh_path] = builder.particle_count
        if entry.deformable_type == "surface" or entry.vis_mesh_path != entry.sim_mesh_path:
            builder.add_cloth_mesh(
                pos=wp.vec3(*entry.init_pos),
                rot=wp.quat(*entry.init_rot),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=entry.vis_vertices,
                indices=entry.vis_indices,
                density=1.0,
                tri_ke=1e4,
                tri_ka=1e4,
                tri_kd=1.5e-6,
                edge_ke=5.0,
                edge_kd=1e-2,
                particle_radius=0.008,
            )
        else:
            builder.add_soft_mesh(
                pos=wp.vec3(*entry.init_pos),
                rot=wp.quat(*entry.init_rot),
                scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=entry.vertices,
                indices=entry.indices,
                density=1000.0,
                k_mu=1e5,
                k_lambda=1e5,
                k_damp=0.0,
            )
    return offsets
