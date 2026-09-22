# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Running statistics shared by the sampling recorders."""

from __future__ import annotations

import math


class RunningStats:
    """Welford online mean and sample standard deviation with a running maximum.

    Attributes:
        mean: Running mean of the recorded samples.
        std: Sample standard deviation, zero for fewer than two samples.
        peak: Largest recorded sample, zero before the first sample.
        n: Number of recorded samples.
    """

    def __init__(self) -> None:
        self.mean = 0.0
        self.std = 0.0
        self.peak = 0.0
        self.n = 0
        self._m2 = 0.0

    def update(self, value: float) -> None:
        """Record one sample."""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self._m2 += delta * (value - self.mean)
        if self.n > 1:
            self.std = math.sqrt(self._m2 / (self.n - 1))
        self.peak = max(self.peak, float(value))


def bytes_to_gb(value: float) -> float:
    """Convert bytes to gigabytes, rounded to two decimals."""
    return round(value / (1024**3), 2)
