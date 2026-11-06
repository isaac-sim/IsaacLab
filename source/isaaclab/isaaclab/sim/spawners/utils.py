# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private helpers shared by the spawner implementations."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pxr import Usd

logger = logging.getLogger(__name__)


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
    from ..schemas.schemas_cfg import SchemaFragment  # noqa: PLC0415

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
    from ..schemas.schemas_cfg import SchemaFragment  # noqa: PLC0415

    if isinstance(value, SchemaFragment):
        return True
    return isinstance(value, (list, tuple)) and all(isinstance(item, SchemaFragment) for item in value)


def apply_schema_props(
    value, anchor_path: str, apply_func: Callable, define_func: Callable, stage: Usd.Stage | None
) -> None:
    """Author a schema family from a spawner-configuration value onto a freshly spawned prim.

    A fragment mapping applies one ``apply_func`` call per entry, in insertion order, with the
    pattern anchored at ``anchor_path`` (so ``""`` targets the anchor itself) and the API created
    when missing. A legacy dataclass configuration routes to ``define_func``.

    Args:
        value: The value of the spawner-configuration field.
        anchor_path: The absolute path of the prim the target patterns anchor on.
        apply_func: The fragment family writer, e.g. ``schemas.apply_mass_properties``.
        define_func: The legacy writer, e.g. ``schemas.define_mass_properties``.
        stage: The stage containing the prim.
    """
    mapping = fragment_mapping(value)
    if mapping is None:
        define_func(anchor_path, value, stage=stage)
        return
    for pattern, fragments in mapping.items():
        apply_func(props_expr(anchor_path, pattern), fragments, create_if_missing=True, stage=stage)


def apply_mesh_collision_props(value, anchor_path: str, default_pattern: str, stage: Usd.Stage) -> None:
    """Author the mesh-collision family from a spawner-configuration value onto colliders.

    Mesh-collision settings describe how a collider is cooked, so the family targets the matched
    prims that carry ``UsdPhysics.CollisionAPI``; other matches are ignored. Each collider receives
    the fragments through :func:`~isaaclab.sim.schemas.apply_mesh_collision_properties`, or a
    legacy configuration through :func:`~isaaclab.sim.schemas.define_mesh_collision_properties`.
    Colliders inside instances cannot be authored on and are skipped. A pattern that matches no
    writable collider logs a warning and authors nothing.

    Args:
        value: The value of the ``mesh_collision_props`` spawner-configuration field.
        anchor_path: The absolute path of the prim the target patterns anchor on.
        default_pattern: The target pattern for the bare fragment (or sequence) and legacy forms.
        stage: The stage containing the prims.
    """
    from pxr import UsdPhysics  # noqa: PLC0415

    from .. import schemas  # noqa: PLC0415
    from ..utils import find_matching_prims  # noqa: PLC0415

    mapping = fragment_mapping(value, default_pattern)
    for pattern, fragments in ({default_pattern: value} if mapping is None else mapping).items():
        if mapping is not None and not fragments:
            continue
        expr = props_expr(anchor_path, pattern)
        colliders = [
            prim
            for prim in find_matching_prims(expr, stage)
            if prim.HasAPI(UsdPhysics.CollisionAPI) and not (prim.IsInstance() or prim.IsInstanceProxy())
        ]
        if not colliders:
            logger.warning("No mesh-collision targets (colliders) matched expression '%s'; nothing was authored.", expr)
            continue
        for collider in colliders:
            collider_path = collider.GetPath().pathString
            if mapping is None:
                schemas.define_mesh_collision_properties(collider_path, fragments, stage=stage)
            else:
                schemas.apply_mesh_collision_properties(collider_path, fragments, stage=stage)


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


def resolve_deformable_slot(cfg) -> tuple[str, dict] | None:
    """Select one deformable family; convenience forms target only the spawn prim.

    Unlike rigid-body tuning, deformable creation must not expand to every mesh in a file.
    Setting an empty slot still requests a deformable body with default properties.
    """
    active = [
        (kind, value)
        for kind, value in (("volume", cfg.volume_deformable_props), ("surface", cfg.surface_deformable_props))
        if value is not None
    ]
    if len(active) + (cfg.deformable_props is not None) > 1:
        raise ValueError(
            "Set only one deformable slot: volume_deformable_props, surface_deformable_props, or deformable_props."
        )
    if not active:
        return None
    kind, value = active[0]
    mapping = fragment_mapping(value)
    if mapping is None:
        raise TypeError(f"{kind}_deformable_props requires a fragment, fragment sequence, or target mapping.")
    return kind, mapping or {"": []}
