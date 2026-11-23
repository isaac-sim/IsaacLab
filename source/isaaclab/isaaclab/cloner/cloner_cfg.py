# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

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

    def worlds(
        self, asset_names: Sequence[str], variants: Sequence[Sequence[SpawnerCfg]]
    ) -> Iterable[Mapping[str, Sequence[SpawnerCfg]]]:
        """Select legal worlds from the available spawner configurations.

        Override this method in the task to select compatible variants using their existing
        ``semantic_tags`` or other configuration properties. No matching policy is built into core.
        The default independently combines all candidates.

        Args:
            asset_names: Scene name for each active declaration instance. Repeated instances and
                rigid object collection members have separate entries, possibly with the same name.
            variants: Candidate spawner configurations for each entry in ``asset_names``. These
                include inherited multi-asset semantic tags and must be treated as read-only.

        Yields:
            Asset names mapped to selected candidate configurations, one per active instance.
            For repeated names, selections follow the corresponding declaration order. Return
            the supplied configurations themselves; copies and foreign configurations are invalid.
            The inclusion weight is divided equally over the generated worlds.
        """
        for choices in itertools.product(*variants):
            world = {}
            for name, choice in zip(asset_names, choices, strict=True):
                world.setdefault(name, []).append(choice)
            yield world


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

    Each inclusion set defines its legal world layouts through :meth:`InclusionSet.worlds`.
    Task-local inclusion subclasses are resolved before replication; replication
    consumes the resulting world prototypes without interpreting inclusion rules.
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
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Resolve inclusion definitions into flat asset-prototype memberships and relative weights.

    Repeated names create repeated instances. Unclaimed assets join every world. Each inclusion
    set selects configurations directly, without enumerating excluded Cartesian combinations.
    Candidate groups follow replicated scene declaration order, including repeated instances.

    Args:
        asset_names: Names of the replicated asset declarations.
        variant_cfgs: Concrete spawner configurations for each replicated declaration.
        clone_combinations: Legal world definitions; None uses the default independent product.
        all_asset_names: Scene names, including shared assets that do not enter world memberships.

    Returns:
        World memberships indexing the flat variant library, and one relative weight per world.

    Raises:
        ValueError: Candidate groups are empty or misaligned, selected assets or configurations
            are invalid, or a definition produces no worlds.
    """
    if len(asset_names) != len(variant_cfgs) or any(not candidates for candidates in variant_cfgs):
        raise ValueError("Each asset requires a nonempty group of variant configurations.")
    known = set(asset_names if all_asset_names is None else all_asset_names)
    combinations = clone_combinations or (InclusionSet(assets=list(dict.fromkeys(asset_names))),)
    claimed = set().union(*(combination.assets for combination in combinations))
    offsets = np.cumsum([0, *(len(candidates) for candidates in variant_cfgs)])
    worlds, weights = [], []
    for combination in combinations:
        if unknown := set(combination.assets) - known:
            raise ValueError(f"Unknown assets in clone combination: {sorted(unknown)}.")
        names, choices, starts = [], [], []
        for index, name in enumerate(asset_names):
            count = combination.assets.count(name) if name in claimed else 1
            names.extend([name] * count)
            choices.extend([variant_cfgs[index]] * count)
            starts.extend([offsets[index]] * count)
        candidate_ids = [{id(cfg): index for index, cfg in enumerate(candidates)} for candidates in choices]
        num_worlds = len(worlds)
        for world in combination.worlds(names, choices):
            if set(world) != set(names):
                raise ValueError(f"World assets must be {sorted(set(names))}, got {sorted(world)}.")
            if any(len(world[name]) != names.count(name) for name in world):
                raise ValueError("World selections must provide one configuration per active asset instance.")
            selections = {name: iter(configs) for name, configs in world.items()}
            members = []
            for name, candidates, start in zip(names, candidate_ids, starts, strict=True):
                selected = next(selections[name])
                if id(selected) not in candidates:
                    raise ValueError(f"Selected configuration for {name!r} is not one of its supplied variants.")
                members.append(int(start + candidates[id(selected)]))
            worlds.append(tuple(members))
        count = len(worlds) - num_worlds
        if not count:
            raise ValueError("An inclusion set must define at least one world layout.")
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
