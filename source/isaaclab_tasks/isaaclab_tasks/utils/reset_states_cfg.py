# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for success-guided reset-row selection."""

from __future__ import annotations

from isaaclab.utils import configclass

from .sampling import BetaSamplingStrategyCfg, SamplerCfg, UniformSamplingStrategyCfg
from .success_monitor_cfg import SuccessMonitorCfg


@configclass
class ResetStateSamplerCfg:
    """Configure outcome tracking and sampling independently of a task's state representation."""

    class_type: type | str = "{DIR}.reset_states:ResetStateSampler"
    """Runtime sampler implementation."""

    success_monitor: SuccessMonitorCfg = SuccessMonitorCfg(monitored_history_len=50)
    """Rolling episode history for each reset row.

    Leave the monitor's target, concentration, and temperature at their defaults;
    :attr:`sampling` owns reset-row selection.
    """

    sampling: SamplerCfg = SamplerCfg(
        strategies=[BetaSamplingStrategyCfg(target=0.5, kappa=2.0), UniformSamplingStrategyCfg(weight=0.05)],
        eps=1.0e-4,
    )
    """Sampling strategies; the default favors intermediate success while retaining all rows."""
