# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`Sampler`."""

from __future__ import annotations

from dataclasses import field

from isaaclab.utils import configclass

from .sampling_strategies_cfg import SamplingScoreCfg


@configclass
class SamplerCfg:
    """Blueprint for a :class:`Sampler`."""

    class_type: type | str = "{DIR}.sampler:Sampler"
    """Runtime sampler class."""
    strategies: list[SamplingScoreCfg] = field(default_factory=list)
    """Weighted score configurations; custom scorers implement ``score(metrics, out)``."""
    eps: float = 1e-3
    """Additive floor on each item's weighted score before probability normalization."""
