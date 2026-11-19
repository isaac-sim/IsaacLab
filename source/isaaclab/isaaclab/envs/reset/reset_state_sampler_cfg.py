# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for mask-native reset-state selection."""

from isaaclab.utils import configclass

from .sampling_scores_cfg import BetaSamplingScoreCfg, SamplingScoreCfg


@configclass
class ResetStateSamplerCfg:
    """Configure recent outcomes and weighted selection independently of physical state storage."""

    history_length: int = 50
    """Completed outcomes retained independently for each catalog entry."""
    scores: list[SamplingScoreCfg] = [BetaSamplingScoreCfg(weight=0.25)]
    """Weighted scorers consuming per-entry metrics. An empty list selects uniformly."""
    uniform_weight: float = 0.0501
    """Weight added to every entry before normalization. A zero-total group draws uniformly."""
    seed: int = 0
    """Initial seed of the independent per-environment random streams."""
