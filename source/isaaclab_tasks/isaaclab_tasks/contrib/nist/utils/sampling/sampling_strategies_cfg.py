# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for sampling strategies."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.nist.utils.sampling.sampling_strategies import (
    BetaSamplingStrategy,
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
    target: float = 0.66
    """Success rate where the Beta score peaks."""
    kappa: float = 1.0
    """Peak sharpness around :attr:`target`; larger values concentrate more mass near the target."""


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


SamplingStrategyCfg = BetaSamplingStrategyCfg | UniformSamplingStrategyCfg
"""Discriminated union of sampling-strategy cfg types."""
