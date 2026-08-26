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


def fragment_mapping(value, default_pattern: str = "") -> dict | None:
    """Normalize a fragment spawner-configuration value to a target-pattern mapping.

    The mapping form (``{pattern: [fragment, ...]}``) is the general spelling. As a convenience, a
    bare fragment or a sequence of fragments is accepted and read as ``{default_pattern: [...]}``.
    The caller picks that default so the convenience form keeps the reach the legacy writers had:
    the file spawners tune a prim together with its subtree, while the shape, mesh, and converter
    spawners author the one prim they just created. Legacy dataclass configurations are reported
    as ``None`` so callers route them to the legacy writers.

    Args:
        value: The value of a fragment spawner-configuration field.
        default_pattern: The target pattern to use for the bare fragment (or sequence) form.

    Returns:
        The equivalent target-pattern mapping, or None when the value is a legacy configuration.
    """
    from isaaclab.sim.schemas.schemas_cfg import SchemaFragment  # noqa: PLC0415

    if isinstance(value, dict):
        return value
    if isinstance(value, SchemaFragment):
        return {default_pattern: [value]}
    if isinstance(value, (list, tuple)) and all(isinstance(item, SchemaFragment) for item in value):
        # an empty sequence carries no fragments and no targeting intent, so it maps to an empty
        # mapping rather than a targeted entry with nothing to author
        return {default_pattern: list(value)} if value else {}
    return None


def bare_fragments(value) -> bool:
    """Report whether a fragment spawner-configuration value uses the convenience form.

    The convenience form is a bare fragment or a sequence of fragments, i.e. everything
    :func:`fragment_mapping` normalizes onto its default pattern. It carries no targeting
    intent of its own, so callers may widen or narrow the target set on the user's behalf.

    Args:
        value: The value of a fragment spawner-configuration field.

    Returns:
        True when the value is a bare fragment or a sequence of fragments.
    """
    from isaaclab.sim.schemas.schemas_cfg import SchemaFragment  # noqa: PLC0415

    if isinstance(value, SchemaFragment):
        return True
    return isinstance(value, (list, tuple)) and all(isinstance(item, SchemaFragment) for item in value)


def subtree_carries_api(prim_path: str, api_type, stage) -> bool:
    """Report whether a prim or any of its descendants carries a USD API schema.

    Args:
        prim_path: The absolute path of the prim rooted at the searched subtree.
        api_type: The USD API schema type to look for (e.g. ``UsdPhysics.RigidBodyAPI``).
        stage: The stage containing the prim.

    Returns:
        True when the prim itself or a prim beneath it carries the API schema.
    """
    from pxr import Usd  # noqa: PLC0415

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return False
    for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        if candidate.HasAPI(api_type):
            return True
    return False
