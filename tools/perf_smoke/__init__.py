# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Performance smoke gate: compare a PR's runtime benchmark against develop's history.

* :mod:`metrics`  -- the runtime bundle's shape
* :mod:`contract` -- which runs are comparable
* :mod:`store`    -- I/O or touching credentials
* :mod:`compare`  -- pure verdict logic
* :mod:`report`   -- presentation
* :mod:`cli`      -- wiring
"""

from .metrics import METRICS, Metric, PerfSmokeError

__all__ = ["METRICS", "Metric", "PerfSmokeError"]
