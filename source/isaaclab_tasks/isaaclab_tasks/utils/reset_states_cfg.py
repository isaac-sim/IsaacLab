# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for success-guided reset-row selection."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sampling import SamplerCfg, TargetSuccessScoreCfg, UniformScoreCfg


@configclass
class ResetStateSamplerCfg:
    """Configure outcome tracking and sampling independently of a task's state representation."""

    class_type: type | str = "{DIR}.reset_states:ResetStateSampler"
    """Runtime sampler implementation."""

    monitored_history_len: int = 50
    """Episodes remembered for each reset row."""

    sampling: SamplerCfg = SamplerCfg(
        strategies=[TargetSuccessScoreCfg(target=0.5, kappa=2.0, weight=0.25), UniformScoreCfg(weight=0.05)],
        eps=1.0e-4,
    )
    """Sampling strategies; the default favors intermediate success while retaining all rows."""
