# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for per-slot success-rate monitoring."""

from __future__ import annotations

from isaaclab.utils import configclass


@configclass
class SuccessMonitorCfg:
    """Configuration for :class:`SuccessMonitor`."""

    class_type: type | str = "{DIR}.success_monitor:SuccessMonitor"
    """Monitor implementation, resolved when the environment starts."""

    monitored_history_len: int = 10
    """Episodes remembered per slot."""

    target_success_rate: float = 0.5
    """Success rate favored by sampling, in ``[0, 1]``."""

    kappa: float = 1.0
    """Concentration around :attr:`target_success_rate`; zero is uniform."""

    temperature: float = 1.0
    """Sampling-weight temperature, at or above ``1.0``."""
