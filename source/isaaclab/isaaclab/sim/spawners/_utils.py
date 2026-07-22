# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by the spawner implementations."""

from __future__ import annotations


def props_expr(prim_path: str, pattern: str) -> str:
    """Join an anchor prim path with a cfg-relative target pattern.

    Implements the key convention of the fragment mapping spawner configuration
    fields (e.g. :attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props`):
    an empty string selects the anchor prim itself, and any other pattern is
    grafted under the anchor prim.

    Args:
        prim_path: The absolute path of the anchor prim.
        pattern: The cfg-relative target pattern.

    Returns:
        The absolute prim path expression to pass to a fragment family writer.
    """
    if not pattern:
        return prim_path
    return f"{prim_path}/{pattern}"


def resolve_deformable_slot(cfg) -> tuple[str, dict] | None:
    """Pick the active deformable slot on a spawner cfg.

    Returns ``("volume" | "surface", mapping)`` for the new dict slots, or None when neither is
    set. Raises when more than one of the new slots and the legacy ``deformable_props`` is set.

    Args:
        cfg: The spawner configuration to inspect.

    Returns:
        The active slot as a ``(kind, mapping)`` pair, or None when no dict slot is set.

    Raises:
        ValueError: If more than one deformable slot (including the legacy ``deformable_props``) is set.
    """
    slots = [
        ("volume", getattr(cfg, "volume_deformable_props", None)),
        ("surface", getattr(cfg, "surface_deformable_props", None)),
    ]
    active = [(kind, mapping) for kind, mapping in slots if mapping is not None]
    legacy = getattr(cfg, "deformable_props", None)
    if len(active) + (legacy is not None) > 1:
        raise ValueError(
            "A spawner configuration may set at most one deformable slot: 'volume_deformable_props',"
            " 'surface_deformable_props', or the legacy 'deformable_props'."
        )
    return active[0] if active else None
