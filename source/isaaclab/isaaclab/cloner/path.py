# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segment-boundary-safe prim-path primitives for the cloner.

A prim path is a sequence of ``/``-delimited segments, not a character string, and the stdlib
string operations cross those boundaries silently: :meth:`str.startswith` reports that
``".../Robot"`` contains ``".../RobotArm"``. This module encodes the boundary semantics once.

Two kinds of prefix appear in the cloner. A *root* is a concrete prefix path
(``"/World/envs/env_0"``); :func:`relative_to`, :func:`under` and :func:`rebase` work against
one. A *template* carries a single ``"{}"`` clone slot standing for one segment
(``"/World/envs/env_{}/Robot"``); :func:`split`, :func:`match` and :func:`relativize` work
against one. Reach them through the package, as ``cloner.path.rebase(...)``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from typing import NamedTuple

import numpy as np

from .clone_plan import ClonePlan


def get_asset_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
    """Select asset-prototype IDs by their declared cfg paths, without expanding instances.

    Args:
        plan: Host asset declarations and their numeric topology.
        path_expr: Exact cfg ``prim_path`` or a regular expression matching the complete declared
            path string. None selects all definitions, including unused prototypes.

    Returns:
        Ascending asset-prototype IDs, shape [num_matches], dtype int32, each included once.
        Generated native paths are not matched.
    """
    if path_expr is None:
        return np.arange(len(plan.asset_cfgs), dtype=np.int32)
    pattern = re.compile(path_expr)
    paths = (cfg.prim_path for cfg in plan.asset_cfgs)
    return np.fromiter(
        (index for index, path in enumerate(paths) if path == path_expr or pattern.fullmatch(path)), dtype=np.int32
    )


def get_world_prototypes(plan: ClonePlan, path_expr: str | None = None) -> np.ndarray:
    """Select world-prototype IDs containing assets matched by their declared cfg paths.

    Args:
        plan: Host asset declarations and their numeric topology.
        path_expr: Asset-path filter interpreted by :func:`get_asset_prototypes`. None selects
            all world definitions, including empty and unused prototypes and shared world -1.

    Returns:
        Ascending world-prototype IDs, shape [num_matches], dtype int32, not destination world IDs.
        Filtering selects complete compositions; repeated asset memberships remain in the topology.
    """
    topology = plan.topology
    prototype_ids = np.arange(-1, len(topology.world_prototype_starts) - 2, dtype=np.int32)
    if path_expr is None:
        return prototype_ids
    matched_assets = np.isin(topology.world_prototypes, get_asset_prototypes(plan, path_expr))
    match_counts = np.r_[0, np.cumsum(matched_assets)]
    return prototype_ids[np.diff(match_counts[topology.world_prototype_starts]) > 0]


def get_instance_paths(plan: ClonePlan) -> tuple[tuple[int, str | None, str, np.ndarray], ...]:
    """Resolve native names for the plan's declared asset instances without accessing a stage.

    Args:
        plan: Host plan supplying declarations, membership, and the destination-world template.

    Returns:
        Asset-prototype ID, authored source path, destination template, and world IDs per named
        occurrence. Shared instances use world -1; unused occurrences have no source path.
        Repeated memberships inherit the prototype pose and receive distinct sibling names.
    """
    topology = plan.topology
    targets = {}
    sorted_world_ids = np.argsort(topology.world_prototype_layout, kind="stable")
    counts = np.bincount(topology.world_prototype_layout, minlength=len(topology.world_prototype_starts) - 2)
    offsets = np.r_[0, np.cumsum(counts)]
    for world_prototype_id in get_world_prototypes(plan):
        start, end = topology.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
        world_ids = (
            np.array([-1])
            if world_prototype_id == -1
            else sorted_world_ids[offsets[world_prototype_id] : offsets[world_prototype_id + 1]]
        )
        names = set()
        for asset_prototype_id in topology.world_prototypes[start:end]:
            cfg = plan.asset_cfgs[asset_prototype_id]
            matched = match(cfg.prim_path, plan.env_template)
            template = plan.env_template + matched.suffix if matched is not None else cfg.prim_path
            if world_prototype_id == -1:
                template = template.format("shared")
            elif matched is None:
                template = plan.env_template + "/" + cfg.prim_path.rsplit("/", 1)[-1]
            name, occurrence = template, 0
            while template in names:
                occurrence += 1
                template = f"{name}_{occurrence}"
            names.add(template)
            targets.setdefault((int(asset_prototype_id), template), []).append(world_ids)
    targets = [(index, template, np.concatenate(groups)) for (index, template), groups in targets.items()]
    source_paths = {}
    for asset_prototype_id, template, world_ids in targets:
        if len(world_ids) and asset_prototype_id not in source_paths:
            spawn = getattr(plan.asset_cfgs[asset_prototype_id], "spawn", None)
            path = getattr(spawn, "spawn_path", None)
            source_paths[asset_prototype_id] = path if path is not None else template.format(int(world_ids[0]))
    return tuple(
        (asset_prototype_id, source_paths.get(asset_prototype_id), template, world_ids)
        for asset_prototype_id, template, world_ids in targets
    )


def get_shared_paths(instances: Iterable[tuple[int, str | None, str, np.ndarray]]) -> tuple[str, ...]:
    """Return minimal shared roots from :func:`get_instance_paths`, without inferring membership from names."""
    paths = tuple(template for _, _, template, world_ids in instances if len(world_ids) and world_ids[0] == -1)
    return tuple(path for path in paths if not any(path != root and under(path, root) for root in paths))


def iter_subtree_copies(
    instances: Iterable[tuple[int, str | None, str, np.ndarray]],
) -> Iterator[tuple[int, str, str, np.ndarray]]:
    """Yield parent-first subtree copies, preserving independently sourced child overrides.

    Args:
        instances: Routed instance paths from :func:`get_instance_paths`.

    Yields:
        Asset-prototype ID, source path, destination template, and world IDs requiring a copy.
    """
    instances = sorted(
        (instance for instance in instances if len(instance[3])),
        key=lambda item: item[2].count("/"),
    )
    for index, (asset_prototype_id, source, destination, world_ids) in enumerate(instances):
        covered = np.zeros(len(world_ids), dtype=np.bool_)
        redundant = covered.copy()
        for _, parent_source, parent_destination, parent_world_ids in reversed(instances[:index]):
            if destination == parent_destination or not under(destination, parent_destination):
                continue
            inherited = np.isin(world_ids, parent_world_ids) & ~covered
            if rebase(source, parent_source, parent_destination) == destination:
                redundant |= inherited
            covered |= inherited
        if not redundant.all():
            yield asset_prototype_id, source, destination, world_ids[~redundant]


class TemplateMatch(NamedTuple):
    """The ``"{}"`` text a template captured (``"3"``, or a wildcard ``".*"``), and the path below it."""

    instance: str
    suffix: str


def split(template: str) -> tuple[str, str]:
    """Split a clone destination template around its ``"{}"`` clone slot.

    The clone slot represents one concrete environment/instance path segment.

    Args:
        template: Destination path template with exactly one ``"{}"`` for the instance id.

    Returns:
        The ``(prefix, suffix)`` strings around the clone slot. A trailing slash is
        insignificant, so an instance-root template (``".../env_{}"``) yields an empty suffix.

    Raises:
        ValueError: If ``template`` does not hold exactly one clone slot. A second slot would
            survive into the suffix and break the later ``str.format`` that fills the first.
    """
    template = template.rstrip("/") or "/"
    slots = template.count("{}")
    if slots != 1:
        raise ValueError(f"Clone destination template must contain exactly one '{{}}', found {slots}: {template!r}.")
    prefix, _, suffix = template.partition("{}")
    return prefix, suffix


def match(path_expr: str, template: str) -> TemplateMatch | None:
    """Match ``path_expr`` against a destination template, capturing the instance slot.

    The ``"{}"`` slot matches one path segment's worth of text: a concrete id (``3``) or a
    wildcard standing for one segment (``.*``, ``[^/]+``). Recovering that text is the only way to tell which instance a
    concrete clone path belongs to without slicing the string by hand.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        A :class:`TemplateMatch` with the captured instance text and the asset-relative
        suffix, or ``None`` when ``path_expr`` is not under the template's instance root.

    Example:
        >>> match("/World/envs/env_3/Robot/base", "/World/envs/env_{}/Robot")
        TemplateMatch(instance='3', suffix='/base')
    """
    prefix, template_suffix = split(template)
    # the slot holds one segment's worth of text: a concrete id, or a wildcard standing for one.
    # A segment-safe wildcard is written as a character class, whose text contains a '/' that is
    # not a separator, so it is matched as a class rather than by the one-segment alternative.
    pattern = re.compile(re.escape(prefix) + r"(\[\^?[^]]*\][*+?]?|[^/]+)" + re.escape(template_suffix))
    matched = pattern.match(path_expr)
    if matched is None:
        return None
    suffix = path_expr[matched.end() :]
    if suffix and not suffix.startswith("/"):
        return None
    return TemplateMatch(matched.group(1), suffix)


def relative_to(path: str, root: str) -> str | None:
    """Strip a concrete ``root`` prefix off ``path`` on a segment boundary.

    Unlike slicing or :meth:`str.removeprefix`, this returns ``None`` rather than a
    mid-segment remainder when ``path`` is not under ``root``.

    Args:
        path: Path to make relative.
        root: Concrete subtree root. A trailing slash is insignificant, and ``"/"`` is the
            root of every path.

    Returns:
        The suffix below ``root`` (starting with ``/``, or ``""`` when ``path`` equals
        ``root``), or ``None`` when ``path`` is not under ``root``.

    """
    root = root.rstrip("/") or "/"
    if path == root:
        return ""
    # "/" prefixes every path but contributes no segment of its own.
    prefix = "" if root == "/" else root
    if not path.startswith(prefix):
        return None
    suffix = path[len(prefix) :]
    return suffix if suffix.startswith("/") else None


def under(path: str, root: str) -> bool:
    """Return whether ``path`` lies within the subtree rooted at ``root``.

    Boundary-correct membership test: unlike :meth:`str.startswith`, it does not match
    across a segment boundary (``".../Robot"`` does not contain ``".../RobotArm"``).

    Args:
        path: Candidate descendant path.
        root: Concrete subtree root.

    Returns:
        ``True`` when ``path`` equals ``root`` or is a descendant of it.
    """
    return relative_to(path, root) is not None


def rebase(path: str, src_root: str, dst_root: str) -> str:
    """Rebase ``path`` from one concrete root prefix onto another on a segment boundary.

    Unlike :meth:`str.replace`, this swaps only a boundary-aligned prefix and touches only
    the leading occurrence.

    Args:
        path: Path to rebase.
        src_root: Concrete source root prefix.
        dst_root: Concrete destination root prefix.

    Returns:
        The rebased path, or ``path`` unchanged when it is not under ``src_root``.
    """
    suffix = relative_to(path, src_root)
    if suffix is None:
        return path
    return (dst_root.rstrip("/") + suffix) or "/"


def relativize(path_expr: str, template: str) -> str | None:
    """Return the part of ``path_expr`` below a template's instance root.

    The suffix half of :func:`match`, for callers that do not need the captured instance.

    Args:
        path_expr: Path or path expression on the clone (destination) side.
        template: Destination path template with ``"{}"`` for the instance id.

    Returns:
        The asset-relative suffix (starting with ``/``, or ``""`` when ``path_expr`` is
        exactly the template root), or ``None`` when ``path_expr`` is not under the root.

    """
    matched = match(path_expr, template)
    return None if matched is None else matched.suffix
