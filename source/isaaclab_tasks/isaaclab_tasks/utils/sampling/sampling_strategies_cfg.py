# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for per-item sampling scores."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class SamplingScoreCfg:
    """Configure a scorer constructed from its config and called with ``score(metrics, out)``."""

    class_type: type | str = MISSING
    """Runtime scorer class, accepting this config as its only constructor argument."""
    weight: float = 1.0
    """Non-negative contribution to the sampler's weighted score sum; negative values are ignored."""


@configclass
class TargetSuccessScoreCfg(SamplingScoreCfg):
    """Configure a target-success kernel whose analytic peak is one.

    The score is ``(rate / target) ** (kappa * target)`` times
    ``((1 - rate) / (1 - target)) ** (kappa * (1 - target))``.
    At targets zero and one, the score is ``(1 - rate) ** kappa`` and
    ``rate ** kappa``, respectively. Zero concentration gives a constant score of one.
    """

    class_type: type | str = "{DIR}.sampling_strategies:TargetSuccessScore"
    metric: str = "success_rate"
    """Caller-owned metric containing success rates in [0, 1], shape [num_items]."""
    target: float = 0.5
    """Success rate where the score is one, clamped to [0, 1]."""
    kappa: float = 2.0
    """Concentration around the target; larger values narrow the peak without changing its height."""


@configclass
class UniformScoreCfg(SamplingScoreCfg):
    """Configure a constant score of one, independent of the supplied metrics."""

    class_type: type | str = "{DIR}.sampling_strategies:UniformScore"


@configclass
class BetaSamplingStrategyCfg(SamplingScoreCfg):
    """Configure the legacy unnormalized Beta kernel.

    The kernel's peak height depends on concentration. Use :class:`TargetSuccessScoreCfg`
    for a peak of one whose mixture weight is independent of concentration.

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

    class_type: type | str = "{DIR}.sampling_strategies:BetaSamplingStrategy"
    """Runtime strategy class."""
    target: float = 0.66
    """Success rate where the Beta score peaks."""
    kappa: float = 1.0
    """Narrow the raw kernel while changing its peak height as concentration increases.

    Use :class:`TargetSuccessScoreCfg` for a concentration-independent peak height.
    """


@configclass
class UniformSamplingStrategyCfg(SamplingScoreCfg):
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

    class_type: type | str = "{DIR}.sampling_strategies:UniformSamplingStrategy"
    """Runtime strategy class."""


SamplingStrategyCfg = BetaSamplingStrategyCfg | UniformSamplingStrategyCfg
"""Compatibility union of the legacy built-in sampling-strategy configurations."""
