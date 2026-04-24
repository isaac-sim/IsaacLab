# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cloner hooks for registering deformable bodies during Newton world replication."""

from __future__ import annotations

from newton import ModelBuilder

from .deformable_object import add_deformable_entry_to_builder


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
