# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queries over the prototype/clone relation a :class:`~isaaclab.cloner.ClonePlan` describes.

Each plan row pairs a prototype path that exists once on the stage with a destination
template and the environments it populates, so the relation is partial in both directions: a
prototype reaches only the environments its row covers, and an environment holds only the
assets whose rows cover it. :func:`path_to_clone`, :func:`path_env_ids` and
:func:`path_to_source` are the three ways to walk it; :func:`iter_sources` is
:func:`path_to_source` for callers that need every variant behind one template.

Environment ids are not mask columns. Column ``j`` stands for
:attr:`~isaaclab.cloner.ClonePlan.env_ids`\\ ``[j]``, which is the number
:func:`~isaaclab.cloner.replicate` formats into the template. These functions take and
return environment ids throughout.

A path belongs to the nearest row containing it: the deepest prototype root, or the template
leaving the shortest suffix. Rows tying there are one asset's variants, and the environment
picks between them. ``test/cloner/test_clone_plan_algebra.py`` pins that down.

The path primitives are aliased ``pth`` because ``path`` is a parameter name here.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from . import path as pth

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _row_env_ids(plan: ClonePlan, row: int) -> tuple[int, ...]:
    """Env ids populated from a plan row: the plan's env ids at the row's ``True`` columns."""
    columns = plan.clone_mask[row].nonzero(as_tuple=False).flatten().tolist()
    if plan.env_ids is None:
        return tuple(int(column) for column in columns)
    return tuple(int(plan.env_ids[column]) for column in columns)


def _column_for_env_id(plan: ClonePlan, env_id: int) -> int | None:
    """Mask column standing for ``env_id``, or ``None`` when the plan does not target it.

    Guards against out-of-range and negative ids, which plain indexing would raise on or
    silently wrap around.
    """
    if plan.env_ids is None:
        return env_id if 0 <= env_id < plan.clone_mask.shape[1] else None
    columns = (plan.env_ids == env_id).nonzero(as_tuple=False).flatten().tolist()
    return int(columns[0]) if columns else None


def _source_rows(plan: ClonePlan, path: str) -> list[int]:
    """Rows whose prototype subtree owns ``path``, nearest owner only, in row order."""
    rows = [
        row for row, source in enumerate(plan.sources) if "{}" in plan.destinations[row] and pth.under(path, source)
    ]
    if not rows:
        return []
    nearest = max(len(plan.sources[row].rstrip("/")) for row in rows)
    return [row for row in rows if len(plan.sources[row].rstrip("/")) == nearest]


def _clone_rows(plan: ClonePlan, path_expr: str, *, populated_only: bool) -> list[tuple[str, pth.TemplateMatch, int]]:
    """Collect ``(template, match, row)`` for the nearest destination template owning ``path_expr``.

    A shorter suffix below the template means a longer matched prefix, i.e. a nearer owner.
    The suffix does not depend on how many digits a row's env ids have, so a variant is never
    ranked out by the width of its env id.

    ``populated_only`` is the active-row policy: :func:`iter_sources` ranks only rows that
    populate an env, so a nearer but empty template cannot hide a populated ancestor, while
    :func:`path_to_source` ranks every row and filters afterwards, so an empty nearest owner
    resolves to ``None`` and its caller falls back to direct stage resolution.
    """
    candidates: list[tuple[str, pth.TemplateMatch, int]] = []
    for row, template in enumerate(plan.destinations):
        if "{}" not in template:
            continue
        if populated_only and not _row_env_ids(plan, row):
            continue
        matched = pth.match(path_expr, template)
        if matched is None:
            continue
        candidates.append((template, matched, row))
    if not candidates:
        return []
    nearest = min(len(matched.suffix) for _, matched, _ in candidates)
    return [candidate for candidate in candidates if len(candidate[1].suffix) == nearest]


def _owning_template(plan: ClonePlan, path_expr: str) -> tuple[str, list[int], pth.TemplateMatch] | None:
    """Resolve the single destination template owning ``path_expr``.

    Returns:
        ``(template, rows, match)`` where ``rows`` are all rows sharing the winning template,
        in row order, or ``None`` when no template owns ``path_expr``.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally near templates
            (a genuine ambiguity). Nested templates do not conflict: the nearest one wins.
    """
    candidates = _clone_rows(plan, path_expr, populated_only=False)
    if not candidates:
        return None
    owning_templates = {template for template, _, _ in candidates}
    if len(owning_templates) > 1:
        raise ValueError(f"path_expr {path_expr!r}: matches multiple destination templates {sorted(owning_templates)}.")
    template, matched, _ = candidates[0]
    return template, [row for _, _, row in candidates], matched


def path_env_ids(plan: ClonePlan, path: str) -> tuple[int, ...]:
    """Return the environments a prototype ``path`` is replicated to.

    Args:
        plan: Active clone plan.
        path: Prototype path.

    Returns:
        The ascending env ids populated from ``path``'s owning rows, empty when the plan does
        not own ``path``.
    """
    env_ids: set[int] = set()
    for row in _source_rows(plan, path):
        env_ids.update(_row_env_ids(plan, row))
    return tuple(sorted(env_ids))


def path_to_clone(plan: ClonePlan, path: str, env_id: int) -> str | None:
    """Return the clone of a prototype ``path`` in one environment.

    Only the prototype root is swapped; everything below it is carried through unchanged.

    Args:
        plan: Active clone plan.
        path: Prototype path.
        env_id: Target environment id.

    Returns:
        The clone path in ``env_id``, or ``None`` when ``path`` is unowned, ``env_id`` is not
        targeted by the plan, or no owning row populates it. Where several variants share the
        prototype subtree, the variant populating ``env_id`` is used.
    """
    column = _column_for_env_id(plan, env_id)
    if column is None:
        return None
    for row in _source_rows(plan, path):
        if bool(plan.clone_mask[row][column]):
            return pth.rebase(path, plan.sources[row], plan.destinations[row].format(env_id))
    return None


def path_to_source(plan: ClonePlan, path_expr: str, env_id: int | None = None) -> tuple[str, str, str] | None:
    """Resolve a clone-side expression to the prototype it was cloned from.

    A *concrete* clone path names its environment in the template's clone slot, and that
    environment selects which variant to report — which is what lets this undo
    :func:`path_to_clone` for a heterogeneous asset. A *wildcard* expression
    (``.../env_.*/...``) names no environment and stands for all of them, so it resolves to
    the first populated variant unless ``env_id`` says which one to take.

    Args:
        plan: Active clone plan.
        path_expr: Clone-side path expression (e.g. a sensor's ``prim_path``, with ``.*`` env
            wildcard) or a concrete clone path.
        env_id: Environment whose variant to resolve. Defaults to the one ``path_expr`` names
            when it is concrete, and to no particular environment otherwise.

    Returns:
        A ``(source_path, destination_glob, asset_suffix)`` tuple, where ``asset_suffix`` is
        the part of ``path_expr`` below the owning template. ``None`` when ``path_expr``
        matches no row, or no matching row populates the requested environment, letting
        callers fall back to direct stage resolution.

        Partial-env coverage is supported: when the matching rows cover only a subset of envs
        (an asset present in some envs but not others, as in heterogeneous scenes), the
        returned glob resolves to just those envs.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally near templates.
    """
    owner = _owning_template(plan, path_expr)
    if owner is None:
        return None
    template, rows, matched = owner
    if env_id is None and matched.instance.isdigit():
        env_id = int(matched.instance)
    # Resolution must walk a prototype that exists on stage, so rows populating no env at all
    # are skipped rather than reported.
    if env_id is None:
        rows = [row for row in rows if plan.clone_mask[row].any()]
    else:
        column = _column_for_env_id(plan, env_id)
        if column is None:
            return None
        rows = [row for row in rows if bool(plan.clone_mask[row][column])]
    if not rows:
        return None
    return plan.sources[rows[0]], template.replace("{}", "*"), matched.suffix


def iter_sources(plan: ClonePlan, path_expr: str) -> Iterator[tuple[str, str, str, tuple[int, ...]]]:
    """Yield every populated plan row whose destination owns a path expression.

    Where :func:`path_to_source` names one variant, this yields them all, for callers that
    must visit each prototype behind a destination template (loading one mesh per variant).

    Example:
        For a row with prototype root ``"/World/source/Robot"``, destination template
        ``"/World/scenes/{}/Robot"`` and env ids ``(0, 2)``, querying
        ``"/World/scenes/.*/Robot/base"`` yields ``("/World/source/Robot",
        "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 2))``.

    Args:
        plan: Clone plan to query.
        path_expr: Clone-side prim path or path expression.

    Yields:
        ``(source_root, destination_template, source_path, env_ids)`` per row of the nearest
        owning template, in row order. Rows populating no env are skipped.
    """
    for template, matched, row in _clone_rows(plan, path_expr, populated_only=True):
        template_norm = template.rstrip("/") or "/"
        source_root = plan.sources[row].rstrip("/") or "/"
        source_path = source_root + matched.suffix if source_root != "/" else matched.suffix or "/"
        yield source_root, template_norm, source_path, _row_env_ids(plan, row)
