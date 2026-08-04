# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for sampling strategies."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils.configclass import configclass

from .sampling_strategies import (
    BetaSamplingStrategy,
    FrontierSamplingStrategy,
    UniformSamplingStrategy,
)


@configclass
class BetaSamplingStrategyCfg:
    """Blueprint for a :class:`BetaSamplingStrategy`.

    Score shape:

    .. code-block:: text

        score
          ^
          |             /\\
          |            /  \\
          |___________/    \\___________
          +----------------------------> success_rate
          0          target             1
    """

    class_type: type[BetaSamplingStrategy] | str = "{DIR}.sampling_strategies:BetaSamplingStrategy"
    """Runtime strategy class."""
    weight: float = 1.0
    """Multiplier on this strategy's score before sampler normalization."""
    plot: bool = False
    """Whether terrain scatter diagnostics should include this strategy's attribution panel."""
    target: float = 0.66
    """Success rate where the Beta score peaks."""
    kappa: float = 1.0
    """Peak sharpness around :attr:`target`; larger values concentrate more mass near the target."""
    success_rate_bind: str = MISSING
    """Expression evaluated in the sampler's ``bind_ns`` resolving to a ``[num_items]`` success-rate Tensor.

    Use ``"success_rates"`` when the calling curriculum term injects its own
    rates tensor as the ``success_rates`` kwarg.
    """


@configclass
class FrontierSamplingStrategyCfg:
    """Blueprint for a :class:`FrontierSamplingStrategy`.

    Score shape:

    .. code-block:: text

        score
          ^
          |          learned neighbor
          |                 *
          |              *  |
          |           *     |  frontier score rises where
          |        *        |  similar tasks are more solved
          |_____*___________v____________
          +-----------------------------> task feature space
              low-rate task
    """

    class_type: type[FrontierSamplingStrategy] | str = "{DIR}.sampling_strategies:FrontierSamplingStrategy"
    """Runtime strategy class."""
    weight: float = 1.0
    """Multiplier on this strategy's score before sampler normalization."""
    plot: bool = False
    """Whether terrain scatter diagnostics should include this strategy's attribution panel."""
    k: int = 8
    """Number of nearest neighbors in task feature space."""
    dilation_steps: int = 1
    """Number of graph max-propagation steps over the kNN neighborhood."""
    success_rate_bind: str = MISSING
    """Expression evaluated in the sampler's ``bind_ns`` resolving to a ``[num_items]`` success-rate Tensor."""


@configclass
class UniformSamplingStrategyCfg:
    """Blueprint for a :class:`UniformSamplingStrategy`.

    Score shape:

    .. code-block:: text

        score
          ^
          |-----------------------------
          |
          |
          +----------------------------> item index
    """

    class_type: type[UniformSamplingStrategy] | str = "{DIR}.sampling_strategies:UniformSamplingStrategy"
    """Runtime strategy class."""
    weight: float = 1.0
    """Multiplier on this strategy's constant score before sampler normalization."""
    plot: bool = False
    """Whether terrain scatter diagnostics should include this strategy's attribution panel."""


SamplingStrategyCfg = BetaSamplingStrategyCfg | FrontierSamplingStrategyCfg | UniformSamplingStrategyCfg
"""Discriminated union of sampling-strategy cfg types."""
