# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration terms for different managers (experimental, Warp-first).

This module is a passthrough to :mod:`isaaclab.managers.manager_term_cfg` except for
the following term configs which are overridden for Warp-first execution:

- :class:`ObservationTermCfg`
- :class:`RewardTermCfg`
- :class:`TerminationTermCfg`
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers.manager_term_cfg import *  # noqa: F401,F403
from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg as _ManagerTermBaseCfg
from isaaclab.utils import configclass


@configclass
class RewardTermCfg(_ManagerTermBaseCfg):
    """Configuration for a reward term.

    The function is expected to write the (unweighted) reward values into a
    pre-allocated Warp buffer provided by the manager.

    Expected signature:

    - ``func(env, out, **params) -> None``

    where ``out`` is a Warp array of shape ``(num_envs,)`` with float32 dtype.
    """

    func: Callable[..., None] = MISSING
    """The function to be called to fill the pre-allocated reward buffer."""

    weight: float = MISSING
    """The weight of the reward term."""


@configclass
class TerminationTermCfg(_ManagerTermBaseCfg):
    """Configuration for a termination term (experimental, Warp-first).

    The function is expected to write termination flags into a pre-allocated Warp buffer.

    Expected signature:

    - ``func(env, out, **params) -> None``

    where ``out`` is a Warp array of shape ``(num_envs,)`` with boolean dtype.
    """

    func: Callable[..., None] = MISSING
    """The function to be called to fill the pre-allocated termination buffer."""

    time_out: bool = False
    """Whether the termination term contributes towards episodic timeouts. Defaults to False."""


@configclass
class ObservationTermCfg(_ManagerTermBaseCfg):
    """Configuration for an observation term (experimental, Warp-first).

    The function is expected to write observation values into a pre-allocated Warp buffer provided
    by the observation manager.

    Expected signature:

    - ``func(env, out, **params) -> None``

    where ``out`` is a Warp array of shape ``(num_envs, obs_term_dim)`` with float32 dtype.

    Notes:
    - The stable fields (noise/modifiers/history) are kept for config compatibility, but the
      experimental Warp-first observation manager may not support all of them initially.
    """

    func: Callable[..., None] = MISSING
    """The function to be called to fill the pre-allocated observation buffer."""

    # Keep stable configuration fields for compatibility with existing task configs.
    modifiers: list[ModifierCfg] | None = None  # noqa: F405
    noise: NoiseCfg | NoiseModelCfg | None = None  # noqa: F405
    clip: tuple[float, float] | None = None
    scale: tuple[float, ...] | float | None = None
    history_length: int = 0
    flatten_history_dim: bool = True
