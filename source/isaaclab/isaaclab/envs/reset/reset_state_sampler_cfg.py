# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for mask-native reset-state selection."""

from isaaclab.utils import configclass

from .sampling_cfg import SamplingScoreCfg, TargetSuccessScoreCfg, UniformScoreCfg


@configclass
class ResetStateSamplerCfg:
    """Configure recent outcomes and weighted selection independently of physical state storage."""

    history_length: int = 50
    """Completed outcomes retained independently for each catalog entry."""
    scores: list[SamplingScoreCfg] = [TargetSuccessScoreCfg(weight=0.25), UniformScoreCfg(weight=0.05)]
    """Weighted native scorers consuming named per-entry metrics."""
    eps: float = 1.0e-4
    """Additive score floor before normalization; a zero-total partition falls back to uniform."""
    seed: int = 0
    """Initial seed of the independent per-environment random streams."""
