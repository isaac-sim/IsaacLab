# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for native reset-state scoring."""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class SamplingScoreCfg:
    """Configure a scorer accepting this config and implementing ``score(metrics, out)``."""

    class_type: type | str = MISSING
    """Runtime scorer class."""
    weight: float = 1.0
    """Non-negative contribution before normalization."""


@configclass
class BetaSamplingScoreCfg(SamplingScoreCfg):
    """Configure a Beta kernel score with a peak of one at the target metric value."""

    class_type: type | str = "{DIR}.sampling_scores:BetaSamplingScore"
    metric: str = "success_rate"
    """Name of the per-entry metric, with values in [0, 1]."""
    target: float = 0.5
    """Metric value where the score is one."""
    kappa: float = 2.0
    """Concentration; zero gives a constant score, larger values narrow its unit-height peak."""
