# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Derive prototype paths and clone relations from source declarations and destination variants."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np

from . import path as pth

if TYPE_CHECKING:
    from .clone_plan import ClonePlan


def _iter_prototypes(
    plan: ClonePlan, source_indices: Sequence[int] | None = None
) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield each populated prototype's path, destination template, and environment mask."""
    for index in range(len(plan.sources)) if source_indices is None else source_indices:
        cfg = plan.sources[index]
        choices = plan.destinations[index]
        variants = np.unique(choices[choices >= 0])
        if not len(variants):
            continue
        matched = pth.match(cfg.prim_path, plan.clone_template)
        template = plan.clone_template + matched.suffix
        spawn = getattr(cfg, "spawn", None)
        paths = None if spawn is None else vars(spawn).get("spawn_paths", (spawn.spawn_path,))
        for variant in variants:
            mask = choices == variant
            first = np.flatnonzero(mask)[0]
            env_id = first if plan.env_ids is None else plan.env_ids[first]
            source = template.format(int(env_id)) if paths is None else paths[variant]
            if source is None:
                raise ValueError(f"Active source {cfg.prim_path!r}, variant {variant}, has no prototype spawn path.")
            yield source, template, mask


def replication_mapping(
    plan: ClonePlan, source_indices: Sequence[int] | None = None
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Derive path arrays and a boolean mapping for low-level backend replication.

    Paths remain within the declared asset subtrees. Backends may batch those prototypes,
    but must not replace them with an undeclared ancestor.

    Args:
        plan: Source declarations and selected destination variants.
        source_indices: Indices into plan.sources consumed by this backend; None selects all.

    Returns:
        Source paths, destination templates, and a boolean [num_prototypes, num_envs] mask.
    """
    prototypes = tuple(_iter_prototypes(plan, source_indices))
    count = plan.destinations.shape[1]
    if not prototypes:
        return (), (), np.empty((0, count), dtype=np.bool_)
    return (
        tuple(src for src, _, _ in prototypes),
        tuple(dst for _, dst, _ in prototypes),
        np.stack([mask for _, _, mask in prototypes]),
    )


def _clone_sources(
    plan: ClonePlan, path_expr: str, *, populated_only: bool
) -> list[tuple[int, str, pth.TemplateMatch]]:
    """Find the nearest declared owner, retaining absent variants for exact source resolution."""
    candidates = []
    for index, cfg in enumerate(plan.sources):
        populated = (plan.destinations[index] >= 0).any()
        if not populated and (populated_only or getattr(cfg, "spawn", None) is None):
            continue
        namespace = pth.match(cfg.prim_path, plan.clone_template)
        if namespace is None:
            continue
        template = plan.clone_template + namespace.suffix
        matched = pth.match(path_expr, template)
        if matched is not None:
            candidates.append((index, template, matched))
    if not candidates:
        return []
    nearest = min(len(matched.suffix) for _, _, matched in candidates)
    candidates = [candidate for candidate in candidates if len(candidate[2].suffix) == nearest]
    return candidates


def path_env_ids(plan: ClonePlan, path: str) -> tuple[int, ...]:
    """Return environment ids receiving a declared prototype path or one of its descendants.

    Args:
        plan: Clone plan to query.
        path: Concrete prototype prim path.

    Returns:
        Ascending destination environment ids, empty when no source owns the path.
    """
    prototypes = [prototype for prototype in _iter_prototypes(plan) if pth.under(path, prototype[0])]
    nearest = max((len(source.rstrip("/")) for source, _, _ in prototypes), default=0)
    return tuple(
        sorted(
            {
                int(column if plan.env_ids is None else plan.env_ids[column])
                for source, _, mask in prototypes
                if len(source.rstrip("/")) == nearest
                for column in np.flatnonzero(mask)
            }
        )
    )


def path_to_clone(plan: ClonePlan, path: str, env_id: int) -> str | None:
    """Resolve a prototype descendant to its clone in one environment.

    Args:
        plan: Clone plan to query.
        path: Concrete prototype prim path.
        env_id: Destination environment id, not a matrix column.

    Returns:
        The clone path, or None when the prototype does not populate that environment.
    """
    prototypes = [prototype for prototype in _iter_prototypes(plan) if pth.under(path, prototype[0])]
    nearest = max((len(source.rstrip("/")) for source, _, _ in prototypes), default=0)
    for source, template, mask in prototypes:
        if len(source.rstrip("/")) != nearest:
            continue
        columns = np.flatnonzero(mask)
        if env_id in (columns if plan.env_ids is None else plan.env_ids[columns]):
            return pth.rebase(path, source, template.format(env_id))
    return None


def path_to_source(plan: ClonePlan, path_expr: str, env_id: int | None = None) -> tuple[str, str, str] | None:
    """Resolve a clone-side expression to its declared prototype.

    Args:
        plan: Clone plan to query.
        path_expr: Concrete clone path or clone-side path expression.
        env_id: Destination environment id. A concrete expression selects its own environment;
            otherwise the first populated variant is selected.

    Returns:
        Source root, destination expression, and asset suffix, or None for an absent instance.
    """
    for index, template, matched in _clone_sources(plan, path_expr, populated_only=False):
        selected_env = env_id
        if selected_env is None and matched.instance.isdigit():
            selected_env = int(matched.instance)
        for source, _, mask in _iter_prototypes(plan, (index,)):
            columns = np.flatnonzero(mask)
            if selected_env is None or selected_env in (columns if plan.env_ids is None else plan.env_ids[columns]):
                return source, template.format("[^/]+"), matched.suffix
    return None


def iter_sources(plan: ClonePlan, path_expr: str) -> Iterator[tuple[str, str, str, tuple[int, ...]]]:
    """Yield every populated prototype behind the nearest owning destination declaration.

    Args:
        plan: Clone plan to query.
        path_expr: Clone-side prim path or path expression.

    Yields:
        Source root, destination template, prototype descendant path, and destination environment ids.
    """
    for index, template, matched in _clone_sources(plan, path_expr, populated_only=True):
        for source, _, mask in _iter_prototypes(plan, (index,)):
            columns = np.flatnonzero(mask)
            env_ids = columns if plan.env_ids is None else plan.env_ids[columns]
            yield (
                source,
                template,
                pth.rebase(path_expr, template.format(matched.instance), source),
                tuple(map(int, env_ids)),
            )
