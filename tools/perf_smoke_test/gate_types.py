# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared model states for the performance smoke test"""

from enum import Enum


class OracleVerdict(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"
    HARD_FAILURE = "HARD_FAILURE"


class BisectVerdict(str, Enum):
    GOOD = "GOOD"
    BAD = "BAD"
    SKIP = "SKIP"


class FailurePhase(str, Enum):
    IMPORT = "import"
    INIT = "init"
    RUNTIME = "runtime"
    OOM = "oom"
    HANG = "hang"
    DRIVER = "driver"
    CONFIG_MISMATCH = "config_mismatch"


class ThresholdSource(str, Enum):
    NO_BASELINE = "no_baseline"
    INSUFFICIENT_WINDOW = "insufficient_window"
    ROLLING_WINDOW = "rolling_window"
    HARD_FLOOR = "hard_floor"
    NOT_APPLICABLE = "n/a"
