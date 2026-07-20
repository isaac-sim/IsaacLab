# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clone-plan queries and the source<->clone maps.

A :class:`~isaaclab.cloner.ClonePlan` is a partial map between two prim-path spaces:
the *source* space (prototype paths that exist once) and the *clone* space (per-env
destination paths and their globs), related by the plan's incidence mask. This module
provides the relation queries and the push/pull maps over that structure.

Access these through the package, e.g.::

    import isaaclab.cloner as cloner

    cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", env_id=2)
    cloner.query.path_to_source(plan, "/World/envs/env_.*/Robot/base")

All functions take ``plan`` first. The module imports the path primitives aliased as
``pth`` because ``path`` is used here as a parameter name.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from . import path as pth

if TYPE_CHECKING:
    from ..clone_plan import ClonePlan


def _source_rows(plan: ClonePlan, path: str) -> list[int]:
    """Row indices whose source subtree owns ``path`` (source space), in row order."""
    return [
        row for row, source in enumerate(plan.sources) if "{}" in plan.destinations[row] and pth.under(path, source)
    ]


def _row_env_ids(plan: ClonePlan, row: int) -> tuple[int, ...]:
    """Env ids populated from a plan row (the ``True`` columns of its mask)."""
    return tuple(int(i) for i in plan.clone_mask[row].nonzero(as_tuple=False).flatten().tolist())


def _clone_candidates(plan: ClonePlan, path_expr: str) -> list[tuple[str, str, int]]:
    """Collect ``(template, suffix, row)`` for every row whose template owns ``path_expr``."""
    candidates: list[tuple[str, str, int]] = []
    for row, template in enumerate(plan.destinations):
        if "{}" not in template:
            continue
        suffix = pth.relativize(path_expr, template)
        if suffix is None:
            continue
        candidates.append((template, suffix, row))
    return candidates


def source_idx(plan: ClonePlan, path: str, *, space: str) -> int | None:
    """Return the source/row index whose chart owns ``path``, or ``None``.

    The returned index co-indexes :attr:`~isaaclab.cloner.ClonePlan.sources`,
    :attr:`~isaaclab.cloner.ClonePlan.destinations`, and the rows of
    :attr:`~isaaclab.cloner.ClonePlan.clone_mask`. It is a prototype index, independent
    of the number of clones.

    Args:
        plan: Active clone plan.
        path: Path to resolve. In ``"source"`` space it is a prototype path matched
            against the plan's sources; in ``"clone"`` space it is a destination
            expression matched against the plan's templates.
        space: Either ``"source"`` or ``"clone"``.

    Returns:
        The owning row index (the nearest/most-specific owner in ``"clone"`` space), or
        ``None`` when no row owns ``path``.

    Raises:
        ValueError: If ``space`` is not ``"source"`` or ``"clone"``, or (in ``"clone"``
            space) if ``path`` is owned by multiple distinct, equally specific templates.
    """
    if space == "source":
        rows = _source_rows(plan, path)
        return rows[0] if rows else None
    if space == "clone":
        owner = _owning_template(plan, path)
        return None if owner is None else owner[1][0]
    raise ValueError(f"space must be 'source' or 'clone', got {space!r}.")


def path_env_ids(plan: ClonePlan, path: str) -> tuple[int, ...]:
    """Return the env fiber of a source-space ``path`` (the envs it maps to).

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.

    Returns:
        The ascending env ids populated from ``path``'s owning row, or an empty tuple
        when ``path`` is not owned by the plan.
    """
    rows = _source_rows(plan, path)
    return _row_env_ids(plan, rows[0]) if rows else ()


def path_maps_to_env(plan: ClonePlan, path: str, env_id: int) -> bool:
    """Return whether a source-space ``path`` is replicated to ``env_id``.

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.
        env_id: Target environment id.

    Returns:
        ``True`` when some owning row of ``path`` populates ``env_id``.
    """
    return any(bool(plan.clone_mask[row][env_id]) for row in _source_rows(plan, path))


def path_to_clone(plan: ClonePlan, path: str, env_id: int) -> str | None:
    """Push a source-space ``path`` forward to one environment's clone.

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.
        env_id: Target environment id.

    Returns:
        The clone path in ``env_id``, or ``None`` when ``path`` is unowned or the owning
        row does not populate ``env_id``. In heterogeneous plans where several rows share
        the source subtree, the row that both owns ``path`` and populates ``env_id`` is used.
    """
    for row in _source_rows(plan, path):
        if bool(plan.clone_mask[row][env_id]):
            return pth.rebase(path, plan.sources[row], plan.destinations[row].format(env_id))
    return None


def path_to_clones(plan: ClonePlan, path: str) -> tuple[str, tuple[int, ...]] | None:
    """Push a source-space ``path`` forward over its whole env fiber, lazily.

    Returns the clone template plus the populated env ids rather than a materialized list
    of concrete paths; format on demand with ``clone_template.format(env_id)``.

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.

    Returns:
        A ``(clone_template, env_ids)`` pair, where ``clone_template`` carries a ``"{}"``
        slot, or ``None`` when ``path`` is not owned by the plan. Coverage is left to the
        caller (``len(env_ids)`` versus the number of envs).
    """
    rows = _source_rows(plan, path)
    if not rows:
        return None
    row = rows[0]
    suffix = pth.relative_to(path, plan.sources[row]) or ""
    clone_template = plan.destinations[row] + suffix
    return clone_template, _row_env_ids(plan, row)


def _owning_template(plan: ClonePlan, path_expr: str) -> tuple[str, list[int], str] | None:
    """Resolve the most-specific destination template owning ``path_expr``.

    Returns:
        ``(template, rows, suffix)`` where ``rows`` are all rows sharing the winning
        template (in row order) and ``suffix`` is the asset-relative tail, or ``None``
        when no template owns ``path_expr``.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally specific
            templates (a genuine ambiguity). Nested templates do not conflict: the most
            specific (shortest-suffix) one wins.
    """
    candidates = _clone_candidates(plan, path_expr)
    if not candidates:
        return None
    min_suffix_len = min(len(suffix) for _, suffix, _ in candidates)
    owning_templates = {template for template, suffix, _ in candidates if len(suffix) == min_suffix_len}
    if len(owning_templates) > 1:
        raise ValueError(f"path_expr {path_expr!r}: matches multiple destination templates {sorted(owning_templates)}.")
    template = next(iter(owning_templates))
    rows = [row for cand_template, _, row in candidates if cand_template == template]
    suffix = next(suffix for cand_template, suffix, _ in candidates if cand_template == template)
    return template, rows, suffix


def path_to_source(plan: ClonePlan, path_expr: str) -> tuple[str, str, str] | None:
    """Pull a clone-space expression back to its source, destination glob, and asset suffix.

    Finds the rows whose destination template owns ``path_expr`` (same matching logic as
    :func:`iter_sources`) and splits the expression at that template so the asset-relative
    suffix is returned for downstream walks.

    Args:
        plan: Active clone plan.
        path_expr: Destination-side path expression (e.g. a sensor's ``prim_path``, with
            ``.*`` env wildcard).

    Returns:
        A ``(source_asset_path, dest_glob_prefix, asset_suffix)`` tuple, where
        ``asset_suffix`` is the part of ``path_expr`` beyond the matching row's destination
        template (empty when ``path_expr`` equals the template). Returns ``None`` when
        ``path_expr`` matches no row, or when the matching rows have no active env, letting
        callers fall back to direct stage resolution.

        Partial-env coverage is supported: when the matching rows cover only a subset of
        envs (an asset present in some envs but not others, as in heterogeneous scenes),
        the returned destination glob resolves to just those envs.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally specific
            destination templates (a genuine ambiguity).
    """
    owner = _owning_template(plan, path_expr)
    if owner is None:
        return None
    template, rows, suffix = owner
    # Partial-env coverage (the union of matching rows misses some envs) is expected for
    # heterogeneous scenes: an asset present in only a subset of envs (e.g. one robot type
    # per task group). The destination glob below resolves only to the envs that actually
    # received the asset, and callers (via the scene Selector) map those to global env ids.
    # Resolution must still walk a source that exists on stage, so prefer the first matching
    # row with at least one active env over an inactive fallback source.
    active_rows = [row for row in rows if plan.clone_mask[row].any()]
    if not active_rows:
        return None
    return plan.sources[active_rows[0]], template.replace("{}", "*"), suffix or ""


def iter_sources(plan: ClonePlan, path_expr: str) -> Iterator[tuple[str, str, str, tuple[int, ...]]]:
    """Yield clone-plan entries whose destinations own a path expression.

    Example:
        For an entry with source root ``"/World/source/Robot"``, destination template
        ``"/World/scenes/{}/Robot"``, and populated env ids ``(0, 2)``, querying
        ``"/World/scenes/.*/Robot/base"`` yields ``("/World/source/Robot",
        "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 2))``.

    Args:
        plan: Clone plan to query.
        path_expr: Destination prim path or path expression, matched against each
            destination template by treating the template's ``"{}"`` field as the
            populated environment slot.

    Yields:
        Tuples ``(source_root, destination_template, source_path, env_ids)`` for the
        nearest matching destination root. Multiple source variants with the same
        destination root are preserved, in row order; rows populating no env are skipped.
    """
    candidates = [
        (template, suffix, row)
        for template, suffix, row in _clone_candidates(plan, path_expr)
        if _row_env_ids(plan, row)
    ]
    if not candidates:
        return
    min_suffix_len = min(len(suffix) for _, suffix, _ in candidates)
    for template, suffix, row in candidates:
        if len(suffix) != min_suffix_len:
            continue
        template_norm = template.rstrip("/") or "/"
        source_root = plan.sources[row].rstrip("/") or "/"
        source_path = source_root + suffix if source_root != "/" else suffix or "/"
        yield source_root, template_norm, source_path, _row_env_ids(plan, row)
