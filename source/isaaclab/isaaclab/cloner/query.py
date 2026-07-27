# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Queries over the source/clone relation a :class:`~isaaclab.cloner.ClonePlan` describes.

A plan relates two prim-path spaces. The *source* space holds prototype paths, which exist
once on the stage; the *clone* space holds the per-env destination paths and the globs that
stand for them. Each plan row ``r`` contributes a source root ``S[r]``, a destination
template ``D[r]``, and the set of envs ``E[r]`` its mask row selects. The plan is the union
of those rows, so the relation is partial in both directions: a prototype reaches only the
envs its row populates, and an env holds only the assets whose rows cover it.

Environment ids are not mask columns. A mask column ``j`` stands for the environment
:attr:`~isaaclab.cloner.ClonePlan.env_ids`\\ ``[j]``, which is what
:func:`~isaaclab.cloner.replicate` formats into the destination template, and which is only
incidentally ``j`` for the contiguous plans the built-in constructors emit. Every query
here takes and returns environment ids, translating through the plan's ``env_ids``.

Three queries generate everything the cloner asks of a plan:

* :func:`path_to_clone` — the *push*: a source path and one env id to that env's clone path.
* :func:`path_env_ids` — the *fiber*: a source path to the envs it reaches.
* :func:`path_to_source` — the *pull*: a clone-space expression back to its prototype.

:func:`iter_sources` is the pull for callers that must see every variant behind one
destination template, rather than a single representative.

Access these through the package::

    import isaaclab.cloner as cloner

    cloner.query.path_to_clone(plan, "/World/envs/env_0/Robot/base", env_id=2)
    cloner.query.path_to_source(plan, "/World/envs/env_.*/Robot/base")

**Ownership.** A path belongs to the nearest row that contains it: on the source side the
deepest source root, on the clone side the template leaving the shortest suffix below it.
Both sides use that same nearest-owner rule. Rows tying at that depth are the variants of
one asset, and the env picks between them: the push takes the variant populating the env it
was asked for, and the pull takes the variant populating the env its path names, falling
back to the first populated variant only when the path names no env at all.

**Laws.** For a source path ``p`` owned by row ``r``, writing ``tail`` for
``path.relative_to(p, S[r])``:

* **Q1 (factorization)** for ``e`` in ``path_env_ids(plan, p)``,
  ``path_to_clone(plan, p, e) == path.rebase(p, S[r], D[r].format(e))``, which is
  ``D[r].format(e) + tail``.
* **Q2 (domain)** ``path_to_clone(plan, p, e) is not None`` exactly when ``e`` is in
  ``path_env_ids(plan, p)``.
* **Q3 (round trip)** for ``e`` in ``path_env_ids(plan, p)``,
  ``path_to_source(plan, path_to_clone(plan, p, e))`` yields ``(source, glob, suffix)`` with
  ``source + suffix == p``. A pushed path is concrete, so the pull reads the env back out of
  it; pass ``env_id`` explicitly to resolve a wildcard expression against one env.

Compositions stay at the call site rather than becoming API. Fan-out over the whole fiber is
``[path_to_clone(plan, p, e) for e in path_env_ids(plan, p)]``, and membership is
``e in path_env_ids(plan, p)``.

The module imports the path primitives aliased as ``pth`` because ``path`` is used here as a
parameter name.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, NamedTuple

from . import path as pth

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


class ResolvedSource(NamedTuple):
    """A clone-space expression resolved back to the prototype it was cloned from."""

    source_path: str
    """Prototype path on the stage, to be read or walked in place of the clone."""

    destination_glob: str
    """The owning destination template with its clone slot replaced by ``*``."""

    asset_suffix: str
    """The part of the queried expression below the owning template (``""`` at its root)."""


class SourceMatch(NamedTuple):
    """One populated plan row behind a destination template."""

    source_root: str
    """The row's prototype root."""

    destination_template: str
    """The row's destination template, with ``"{}"`` for the env id."""

    source_path: str
    """The queried expression rebased onto :attr:`source_root`."""

    env_ids: tuple[int, ...]
    """Environment ids this row populates, ascending."""


def _row_env_ids(plan: ClonePlan, row: int) -> tuple[int, ...]:
    """Env ids populated from a plan row: the plan's env ids at the row's ``True`` columns."""
    columns = plan.clone_mask[row].nonzero(as_tuple=False).flatten().tolist()
    if plan.env_ids is None:
        return tuple(int(column) for column in columns)
    return tuple(int(plan.env_ids[column]) for column in columns)


def _column_for_env_id(plan: ClonePlan, env_id: int) -> int | None:
    """Mask column standing for ``env_id``, or ``None`` when the plan does not target it.

    Guards the mask against out-of-range and negative ids, which plain indexing would either
    raise on or silently wrap around.
    """
    num_columns = plan.clone_mask.shape[1]
    if plan.env_ids is None:
        return env_id if 0 <= env_id < num_columns else None
    columns = (plan.env_ids == env_id).nonzero(as_tuple=False).flatten().tolist()
    return int(columns[0]) if columns else None


def _source_rows(plan: ClonePlan, path: str) -> list[int]:
    """Rows whose source subtree owns ``path``, nearest owner only, in row order.

    Source-side counterpart of :func:`_clone_rows`: when several source roots contain
    ``path`` (a prototype nested inside another prototype), the deepest one owns it. Rows
    tying at that depth are the variants of one asset and are all returned.
    """
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
    The suffix does not depend on how many digits a row's env ids happen to have, so a
    variant is never ranked out by the width of its env id.

    ``populated_only`` is the active-row policy: :func:`iter_sources` ranks only rows that
    populate an env, so a nearer but empty template does not hide a populated ancestor, while
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
        ``(template, rows, match)`` where ``rows`` are all rows sharing the winning template
        (in row order), or ``None`` when no template owns ``path_expr``.

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
    """Return the env fiber of a source-space ``path``: the envs it is replicated to.

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.

    Returns:
        The ascending env ids populated from ``path``'s owning rows, or an empty tuple when
        ``path`` is not owned by the plan.
    """
    env_ids: set[int] = set()
    for row in _source_rows(plan, path):
        env_ids.update(_row_env_ids(plan, row))
    return tuple(sorted(env_ids))


def path_to_clone(plan: ClonePlan, path: str, env_id: int) -> str | None:
    """Push a source-space ``path`` forward to one environment's clone.

    Args:
        plan: Active clone plan.
        path: Source-space prototype path.
        env_id: Target environment id.

    Returns:
        The clone path in ``env_id``, or ``None`` when ``path`` is unowned, ``env_id`` is not
        targeted by the plan, or no owning row populates it. Where several variants share the
        source subtree, the variant that populates ``env_id`` is used.
    """
    column = _column_for_env_id(plan, env_id)
    if column is None:
        return None
    for row in _source_rows(plan, path):
        if bool(plan.clone_mask[row][column]):
            return pth.rebase(path, plan.sources[row], plan.destinations[row].format(env_id))
    return None


def path_to_source(plan: ClonePlan, path_expr: str, env_id: int | None = None) -> ResolvedSource | None:
    """Pull a clone-space expression back to the prototype it was cloned from.

    Splits ``path_expr`` at the destination template that owns it, so the asset-relative
    suffix is returned for downstream walks.

    A *concrete* clone path names its environment in the template's clone slot, and that
    environment selects the variant to report — which is what makes this invert
    :func:`path_to_clone` for a heterogeneous asset. A *wildcard* expression
    (``.../env_.*/...``) names no environment and stands for all of them, so it resolves to
    the first populated variant unless ``env_id`` says which one to take.

    Args:
        plan: Active clone plan.
        path_expr: Destination-side path expression (e.g. a sensor's ``prim_path``, with
            ``.*`` env wildcard) or a concrete clone path.
        env_id: Environment whose variant to resolve. Defaults to the one named by
            ``path_expr`` when it is concrete, and to no particular environment otherwise.

    Returns:
        A :class:`ResolvedSource`, or ``None`` when ``path_expr`` matches no row, when the
        plan does not target the requested environment, or when no matching row populates it
        — letting callers fall back to direct stage resolution.

        Partial-env coverage is supported: when the matching rows cover only a subset of
        envs (an asset present in some envs but not others, as in heterogeneous scenes), the
        returned destination glob resolves to just those envs.

    Raises:
        ValueError: When ``path_expr`` is owned by multiple distinct, equally near
            destination templates (a genuine ambiguity).
    """
    owner = _owning_template(plan, path_expr)
    if owner is None:
        return None
    template, rows, matched = owner
    if env_id is None and matched.instance.isdigit():
        env_id = int(matched.instance)
    # Partial-env coverage (the union of matching rows misses some envs) is expected for
    # heterogeneous scenes: an asset present in only a subset of envs (e.g. one robot type
    # per task group). The destination glob below resolves only to the envs that actually
    # received the asset, and callers (via the scene Selector) map those to global env ids.
    # Resolution must still walk a source that exists on stage, so skip rows populating no
    # env at all.
    if env_id is None:
        rows = [row for row in rows if plan.clone_mask[row].any()]
    else:
        column = _column_for_env_id(plan, env_id)
        if column is None:
            return None
        rows = [row for row in rows if bool(plan.clone_mask[row][column])]
    if not rows:
        return None
    return ResolvedSource(plan.sources[rows[0]], template.replace("{}", "*"), matched.suffix)


def iter_sources(plan: ClonePlan, path_expr: str) -> Iterator[SourceMatch]:
    """Yield every populated plan row whose destination owns a path expression.

    Where :func:`path_to_source` names one variant, this yields them all, for callers that
    must visit each prototype behind a destination template (loading one mesh per variant,
    say). A wildcard expression is inherently one-to-many, and this is the query that says so.

    Example:
        For a row with source root ``"/World/source/Robot"``, destination template
        ``"/World/scenes/{}/Robot"``, and populated env ids ``(0, 2)``, querying
        ``"/World/scenes/.*/Robot/base"`` yields ``("/World/source/Robot",
        "/World/scenes/{}/Robot", "/World/source/Robot/base", (0, 2))``.

    Args:
        plan: Clone plan to query.
        path_expr: Destination prim path or path expression, matched against each destination
            template by treating the template's ``"{}"`` field as the populated environment
            slot.

    Yields:
        A :class:`SourceMatch` per row of the nearest owning destination template. Variants
        sharing that template are all yielded, in row order; rows populating no env are
        skipped.
    """
    for template, matched, row in _clone_rows(plan, path_expr, populated_only=True):
        template_norm = template.rstrip("/") or "/"
        source_root = plan.sources[row].rstrip("/") or "/"
        source_path = source_root + matched.suffix if source_root != "/" else matched.suffix or "/"
        yield SourceMatch(source_root, template_norm, source_path, _row_env_ids(plan, row))
