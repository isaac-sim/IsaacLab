# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .cloner_strategies import sequential


@configclass
class InclusionSet:
    """Legal clone combination defined by explicitly listing active assets."""

    assets: list[str] = MISSING
    """Scene asset names active in this clone combination."""

    weight: int = 1
    """Relative sampling weight for this clone combination."""


@configclass
class CloneCfg:
    """Configuration for environment replication.

    Holds the knobs :class:`~isaaclab.scene.InteractiveScene` forwards to
    :func:`~isaaclab.cloner.make_clone_plan` when building per-env layouts.
    """

    clone_strategy: Callable[..., object] = sequential
    """Function used to build prototype-to-environment mapping. Default is :func:`sequential`."""

    clone_combinations: list[InclusionSet] = []
    """Legal scene-asset combinations for heterogeneous clone planning.

    Each entry names the assets that are active in one legal combination.
    Assets not referenced by any entry are active in every combination. An
    empty list keeps the homogeneous/default behavior.
    """

    device: str = "cpu"
    """Torch device on which mapping buffers are allocated."""

    clone_regex: str = "/World/envs/env_.*"
    """Regex matching every replicated env prim. Used to expand ``{ENV_REGEX_NS}`` cfg macros."""


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
