# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import MISSING

import numpy as np

from ..utils import configclass
from .cloner_strategies import sequential

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

    layouts: Sequence[Sequence[int]] | None = None
    """Explicit variant-index rows; None uses the independent product.

    Columns follow active scene declaration order; repeated instances and members of a rigid
    object collection each have their own column. Unclaimed assets also contribute columns,
    once per declaration. Indices are local to each declaration's spawner, starting at zero.
    """

    def worlds(self, asset_names: Sequence[str], variants: Sequence[range]) -> Iterable[Sequence[int]]:
        """Generate legal layouts without constructing combinations that the definition excludes.

        Override this method to pair or otherwise couple asset choices. The default returns
        :attr:`layouts` when supplied, otherwise the Cartesian product of the candidate ranges.
        The inclusion set's weight is divided equally over the generated layouts.

        Args:
            asset_names: Active scene declaration name of each column, including repeated instances.
            variants: Candidate variant indices for each column, as ``range(variant_count)``.

        Returns:
            Rows selecting one local variant index per column, in the supplied order.
        """
        return itertools.product(*variants) if self.layouts is None else self.layouts


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
    Explicit rows and custom inclusion subclasses are resolved before replication; replication
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
    variant_counts: Sequence[int],
    clone_combinations: Sequence[InclusionSet] | None = None,
    *,
    all_asset_names: Sequence[str] | None = None,
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    """Resolve inclusion definitions into flat asset-prototype memberships and relative weights.

    Repeated names create repeated instances. Unclaimed assets join every world. Each inclusion
    set owns the layouts it generates; explicit or coupled layouts never expand into an independent
    Cartesian product. Columns follow replicated scene declaration order, including repeated instances.

    Args:
        asset_names: Names of the replicated asset declarations.
        variant_counts: Number of concrete prototypes for each declaration.
        clone_combinations: Legal world definitions; None uses the default independent product.
        all_asset_names: Scene names, including shared assets that do not enter world memberships.

    Returns:
        World memberships indexing the flat variant library, and one relative weight per world.

    Raises:
        ValueError: A declaration, asset name, layout width, or variant index is invalid, or a
            definition produces no worlds.
    """
    if len(asset_names) != len(variant_counts) or any(count <= 0 for count in variant_counts):
        raise ValueError("Each asset requires one positive variant count.")
    known = set(asset_names if all_asset_names is None else all_asset_names)
    combinations = clone_combinations or (InclusionSet(assets=list(dict.fromkeys(asset_names))),)
    claimed = set().union(*(combination.assets for combination in combinations))
    offsets = np.cumsum([0, *variant_counts])
    worlds, weights = [], []
    for combination in combinations:
        if unknown := set(combination.assets) - known:
            raise ValueError(f"Unknown assets in clone combination: {sorted(unknown)}.")
        names, choices, starts = [], [], []
        for index, name in enumerate(asset_names):
            count = combination.assets.count(name) if name in claimed else 1
            names.extend([name] * count)
            choices.extend([range(variant_counts[index])] * count)
            starts.extend([offsets[index]] * count)
        layouts = combination.worlds(names, choices)
        num_worlds = len(worlds)
        for layout in layouts:
            if len(layout) != len(choices):
                raise ValueError(f"World layout requires {len(choices)} variant indices, got {len(layout)}.")
            for variant, candidates in zip(layout, choices, strict=True):
                is_index = isinstance(variant, (int, np.integer)) and not isinstance(variant, bool)
                if not is_index or not 0 <= variant < len(candidates):
                    raise ValueError(
                        f"Invalid variant index {variant!r}: expected an integer in [0, {len(candidates)})."
                    )
            worlds.append(tuple(int(start + variant) for start, variant in zip(starts, layout, strict=True)))
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
