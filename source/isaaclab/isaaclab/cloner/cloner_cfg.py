# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

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

    clone_template: str = DEFAULT_ENV_TEMPLATE
    """Path template for every replicated env prim, where ``{}`` is the environment index.

    The regex form used to expand ``{ENV_REGEX_NS}`` cfg macros is
    ``clone_template.format("[^/]+")``, which confines the slot to one path segment.
    """

    replicate_physics: bool = True
    """Whether physics replication clones each environment. Default is True.

    If False, cloning is USD-only: the physics engine parses the per-env USD prims directly
    instead of replicating env_0's parsed structure. Applied by :func:`~isaaclab.cloner.replicate`.
    """


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
