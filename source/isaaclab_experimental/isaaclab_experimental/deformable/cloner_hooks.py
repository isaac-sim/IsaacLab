# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloner hooks for registering deformable bodies during Newton world replication."""

from __future__ import annotations

import logging

import warp as wp
from newton import ModelBuilder

logger = logging.getLogger(__name__)


def add_deformable_entry_to_builder(
    builder: ModelBuilder,
    entry,
    env_idx: int,
    env_position: list[float],
) -> None:
    """Add a single deformable registry entry to the builder for one environment.

    Args:
        builder: The Newton model builder.
        entry: A :class:`DeformableRegistryEntry` with mesh data and config.
        env_idx: The environment index.
        env_position: World position [x, y, z] for this environment.
    """
    before_count = getattr(builder, "particle_count", 0)

    body_pos = wp.vec3(
        entry.init_pos[0] + env_position[0],
        entry.init_pos[1] + env_position[1],
        entry.init_pos[2] + env_position[2],
    )
    body_rot = wp.quat(entry.init_rot[0], entry.init_rot[1], entry.init_rot[2], entry.init_rot[3])

    if entry.deformable_type == "volume":
        builder.add_soft_mesh(
            pos=body_pos,
            rot=body_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=entry.vertices,
            indices=entry.indices,
            density=entry.density,
            k_mu=entry.k_mu,
            k_lambda=entry.k_lambda,
            k_damp=entry.k_damp,
            particle_radius=entry.particle_radius,
        )
    elif entry.deformable_type == "surface":
        builder.add_cloth_mesh(
            pos=body_pos,
            rot=body_rot,
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=entry.vertices,
            indices=entry.indices,
            density=entry.density,
            tri_ke=entry.tri_ke,
            tri_ka=entry.tri_ka,
            tri_kd=entry.tri_kd,
            edge_ke=entry.edge_ke,
            edge_kd=entry.edge_kd,
            particle_radius=entry.particle_radius,
        )
    else:
        raise ValueError(
            f"Invalid deformable type '{entry.deformable_type}' for registry entry"
            f" with prim path '{entry.prim_path}'"
        )

    after_count = getattr(builder, "particle_count", 0)
    delta = after_count - before_count

    entry.particle_offsets.append(before_count)
    if env_idx == 0:
        entry.particles_per_body = delta


def per_world_deformable_hook(builder: ModelBuilder, world_idx: int, env_position: list[float]) -> None:
    """Per-world builder hook: add all deformable bodies from the registry.

    Args:
        builder: The Newton model builder.
        world_idx: The world/environment index.
        env_position: World position [x, y, z] for this environment.
    """
    from isaaclab_newton.physics import NewtonManager

    for entry in NewtonManager._deformable_registry:
        add_deformable_entry_to_builder(builder, entry, world_idx, env_position)


def post_replicate_deformable_hook(builder: ModelBuilder) -> None:
    """Post-replicate hook: call ``builder.color()`` if deformable bodies are present.

    Required by the VBD solver for parallel vertex colouring.

    Args:
        builder: The Newton model builder.
    """
    from isaaclab_newton.physics import NewtonManager

    if NewtonManager._deformable_registry:
        builder.color()
