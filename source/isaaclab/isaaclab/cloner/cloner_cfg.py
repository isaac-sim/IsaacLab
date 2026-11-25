# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..utils import configclass
from .cloner_strategies import sequential

if TYPE_CHECKING:
    from ..sim.spawners.spawner_cfg import SpawnerCfg

DEFAULT_ENV_TEMPLATE = "/World/envs/env_{}"
"""Default path template for a replicated env prim; ``{}`` marks the environment index."""


def expand_env_regex_ns(path_expr: str, env_template: str = DEFAULT_ENV_TEMPLATE) -> str:
    """Replace the ``{ENV_REGEX_NS}`` macro with the environment namespace it stands for.

    The macro spares a configuration from spelling the namespace, and with it the segment
    wildcard that names one environment. :class:`~isaaclab.scene.InteractiveScene` expands it
    against its own template for the assets it collects; assets built outside the scene (a
    direct environment builds its own) go through here instead.

    Args:
        path_expr: Prim path expression, with or without the macro.
        env_template: Environment path template whose ``{}`` marks the environment index.

    Returns:
        ``path_expr`` with the macro replaced, unchanged when it holds no macro.
    """
    # a plain replace, not str.format: the rest of the expression may hold braces of its own
    return path_expr.replace("{ENV_REGEX_NS}", env_template.format("[^/]+"))


@configclass
class InclusionSet:
    """Legal world layouts for explicitly selected scene assets."""

    assets: list[str] = MISSING
    """Scene asset names active in this clone combination."""

    weight: float = 1.0
    """Relative sampling weight for this clone combination."""

    match_tags: list[str] = []
    """Semantic tag types whose values must overlap across the selected assets.

    Each selected variant must share a value for every listed tag type. Multiple values for a
    tag express alternatives: a fixture supporting several sizes can match any one of them.
    Assets not claimed by any inclusion set remain independent and do not participate in matching.
    Empty keeps the existing independent Cartesian product. Matching groups remain factorized
    until the scene's clone strategy selects concrete variants for its environments.
    """

    sample_by: Literal["category", "entry"] = "entry"
    """How the clone strategy samples compatible assets.

    ``"entry"`` samples assets directly. ``"category"`` first samples a ``category`` semantic
    tag value uniformly, then an asset uniformly within it, independently for each named asset.
    Category sampling requires every candidate to have a ``category`` tag; multiple values put
    an asset in multiple categories. Both modes apply after compatibility matching, without
    enumerating category combinations. The clone strategy owns both sampling steps.
    """


@configclass
class CloneCfg:
    """Configuration for environment replication.

    Holds the knobs :class:`~isaaclab.scene.InteractiveScene` forwards to
    :func:`~isaaclab.cloner.make_clone_plan` when building per-env layouts.
    """

    clone_strategy: Callable[[np.ndarray, int], np.ndarray] = sequential
    """Function selecting world-prototype indices from relative weights. Default is :func:`sequential`."""

    clone_combinations: list[InclusionSet] = []
    """Legal scene-asset combinations for heterogeneous clone planning.

    Each entry names the assets that are active in one legal combination.
    Assets not referenced by any entry are active in every combination. An
    empty list keeps the homogeneous/default behavior.

    Inclusion sets declare semantic compatibility through :attr:`InclusionSet.match_tags`.
    Planning resolves compatible groups before replication. The clone strategy selects groups
    and concrete variants; inclusion definitions do not own environment counts or random seeds.
    """

    clone_template: str = DEFAULT_ENV_TEMPLATE
    """Path template for every replicated env prim, where ``{}`` is the environment index.

    The regex form used to expand ``{ENV_REGEX_NS}`` cfg macros is
    ``clone_template.format("[^/]+")``, which confines the slot to one path segment.
    """

    replicate_physics: bool | None = None
    """Deprecated and ignored. Clone contexts are selected per asset."""

    def __post_init__(self) -> None:
        if self.replicate_physics is not None:
            warnings.warn("CloneCfg.replicate_physics is deprecated and ignored.", DeprecationWarning, stacklevel=2)


def make_valid_clone_combinations(
    asset_names: Sequence[str],
    variant_cfgs: Sequence[Sequence[SpawnerCfg]],
    clone_combinations: Sequence[InclusionSet] | None = None,
    *,
    all_asset_names: Sequence[str] | None = None,
) -> tuple[tuple[tuple[int | tuple, ...], ...], np.ndarray]:
    """Resolve inclusion membership and semantic compatibility before replication.

    Repeated names create repeated instances. Unclaimed assets join every world independently.
    Without matching or category sampling, preserve the existing exhaustive product. Otherwise return
    compact compatible groups: each slot lists its candidate prototype IDs, without expanding
    the Cartesian product within a group. Concrete selection belongs to clone-plan construction.

    Args:
        asset_names: Names of the replicated asset declarations.
        variant_cfgs: Concrete spawner configurations for each replicated declaration, including
            semantic tags inherited from their multi-asset wrapper.
        clone_combinations: Legal world definitions; None uses the default independent product.
        all_asset_names: Scene names, including shared assets that do not enter world memberships.

    Returns:
        World definitions and their relative weights. A definition contains concrete prototype IDs
        or tuples of alternative IDs per slot. Category sampling adds a nested tuple per tag value.
        Inclusion weight is split equally across definitions.

    Raises:
        ValueError: Candidate groups are empty or misaligned, asset names are unknown, or no
            compatible world exists.
    """
    if len(asset_names) != len(variant_cfgs) or any(not candidates for candidates in variant_cfgs):
        raise ValueError("Each asset requires a nonempty group of variant configurations.")
    known = set(asset_names if all_asset_names is None else all_asset_names)
    combinations = clone_combinations or (InclusionSet(assets=list(dict.fromkeys(asset_names))),)
    claimed = set().union(*(combination.assets for combination in combinations))
    offsets = np.cumsum([0, *(len(candidates) for candidates in variant_cfgs)])
    configurations = [cfg for candidates in variant_cfgs for cfg in candidates]
    worlds, weights = [], []
    for combination in combinations:
        if combination.sample_by not in ("category", "entry"):
            raise ValueError("InclusionSet.sample_by must be 'category' or 'entry'.")
        if unknown := set(combination.assets) - known:
            raise ValueError(f"Unknown assets in clone combination: {sorted(unknown)}.")
        names, choices, candidates = [], [], []
        for index, name in enumerate(asset_names):
            count = combination.assets.count(name) if name in claimed else 1
            names.extend([name] * count)
            choices.extend([tuple(range(int(offsets[index]), int(offsets[index + 1])))] * count)
            candidates.extend([variant_cfgs[index]] * count)
        start = len(worlds)
        if not combination.match_tags and combination.sample_by == "entry":
            worlds.extend(itertools.product(*choices))
        else:
            buckets = []
            common = None
            for name, ids, configs in zip(names, choices, candidates, strict=True):
                if name not in combination.assets:
                    buckets.append(None)
                    continue
                groups = {}
                for prototype, cfg in zip(ids, configs, strict=True):
                    tags = cfg.semantic_tags or ()
                    values = [
                        tuple(dict.fromkeys(value for tag, value in tags if tag == key))
                        for key in combination.match_tags
                    ]
                    for key in itertools.product(*values):
                        groups.setdefault(key, []).append(prototype)
                buckets.append(groups)
                common = list(groups) if common is None else [key for key in common if key in groups]
            for key in [()] if common is None else common:
                slots = []
                for name, group, ids in zip(names, buckets, choices, strict=True):
                    compatible = tuple(group[key]) if group is not None else ids
                    if combination.sample_by == "category" and name in combination.assets:
                        categories = {}
                        for prototype in compatible:
                            values = dict.fromkeys(
                                value
                                for tag, value in configurations[prototype].semantic_tags or ()
                                if tag == "category"
                            )
                            if not values:
                                raise ValueError(f"Asset {name!r} has a candidate without 'category' tags.")
                            for value in values:
                                categories.setdefault(value, []).append(prototype)
                        slots.append(tuple(tuple(category) for category in categories.values()))
                    else:
                        slots.append(compatible)
                worlds.append(tuple(slots))
            # Identical groups can arise when all candidates carry several matching values.
            worlds[start:] = dict.fromkeys(worlds[start:])
        count = len(worlds) - start
        if not count:
            raise ValueError("An inclusion set must define at least one compatible world.")
        weights.extend([combination.weight / count] * count)
    return tuple(worlds), np.asarray(weights, dtype=np.float64)


def add(this: CloneCfg, other: InclusionSet) -> CloneCfg:
    """Append one clone combination to ``this`` and return it.

    Args:
        this: Configuration that accumulates the combination.
        other: Clone combination to append.

    Returns:
        ``this``, with the combination appended.
    """
    this.clone_combinations = [*this.clone_combinations, other]
    return this
