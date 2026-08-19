# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by the spawner implementations."""

from __future__ import annotations


def props_expr(prim_path: str, pattern: str) -> str:
    """Append a cfg-relative target pattern to an anchor prim path.

    Implements the key convention of the fragment mapping spawner configuration fields
    (e.g. :attr:`~isaaclab.sim.spawners.RigidObjectSpawnerCfg.rigid_props`): the key is a
    regular-expression suffix appended to the anchor prim path, so it carries its own leading
    ``/`` when it targets descendants. An empty key selects the anchor prim itself.

    The result is a plain regular expression matched against whole prim paths by
    :func:`~isaaclab.sim.utils.find_matching_prims`, so ``"/[^/]+"`` selects the anchor's direct
    children and ``"/.*"`` everything beneath it. Use ``"(/.*)?"`` on the rare occasion the anchor
    itself is also a valid family target.

    Args:
        prim_path: The absolute path of the anchor prim.
        pattern: The cfg-relative target pattern.

    Returns:
        The absolute prim path expression to pass to a fragment family writer.
    """
    return f"{prim_path}{pattern}"


def fragment_mapping(value) -> dict | None:
    """Normalize a fragment spawner-configuration value to a target-pattern mapping.

    The mapping form (``{pattern: [fragment, ...]}``) is the general spelling. As a convenience,
    a bare fragment or a sequence of fragments is accepted for the common case of authoring on
    the anchor prim itself, and is read as ``{"": [...]}``. Legacy dataclass configurations are
    reported as ``None`` so callers route them to the legacy writers.

    Args:
        value: The value of a fragment spawner-configuration field.

    Returns:
        The equivalent target-pattern mapping, or None when the value is a legacy configuration.
    """
    from isaaclab.sim.schemas.schemas_cfg import SchemaFragment  # noqa: PLC0415

    if isinstance(value, dict):
        return value
    if isinstance(value, SchemaFragment):
        return {"": [value]}
    if isinstance(value, (list, tuple)) and all(isinstance(item, SchemaFragment) for item in value):
        return {"": list(value)}
    return None


def resolve_deformable_slot(cfg) -> tuple[str, dict] | None:
    """Pick the active deformable slot on a spawner cfg.

    Returns ``("volume" | "surface", mapping)`` for the new slots, normalized through
    :func:`fragment_mapping`, or None when neither is set. Raises when more than one of the new
    slots and the legacy ``deformable_props`` is set.

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
    active = [(kind, fragment_mapping(value)) for kind, value in slots if value is not None]
    legacy = getattr(cfg, "deformable_props", None)
    if len(active) + (legacy is not None) > 1:
        raise ValueError(
            "A spawner configuration may set at most one deformable slot: 'volume_deformable_props',"
            " 'surface_deformable_props', or the legacy 'deformable_props'."
        )
    return active[0] if active else None
