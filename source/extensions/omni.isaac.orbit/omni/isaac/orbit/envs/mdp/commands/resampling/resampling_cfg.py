# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from omni.isaac.orbit.managers import ResamplingTermCfg
from omni.isaac.orbit.utils import configclass

"""
Fixed frequency resampling term.
"""


@configclass
class FixedFrequencyCfg(ResamplingTermCfg):
    """Configuration for the fixed frequency resampling term."""

    resampling_time_range: tuple[float, float] = MISSING
