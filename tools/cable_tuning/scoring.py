# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Cost function for a single tuning run.

Lower is better. Tiered weights ensure each higher-priority failure mode strictly
dominates everything below it:

  NaN  >>  incomplete state machine  >>  exploded magnitudes  >>  slow settle  >>
  poor goal tracking  >>  residual cable oscillation.
"""

from __future__ import annotations

from typing import Mapping


WEIGHT_NAN = 1.0e6
WEIGHT_STATE = 1.0e3
WEIGHT_EXPLODED = 1.0e2
WEIGHT_SETTLE = 1.0
WEIGHT_GOAL = 10.0
WEIGHT_OSC = 1.0

# Target: max_state_reached == 5 means LIFT_OBJECT held to episode end.
_MAX_STATE_TARGET = 5


def score_summary(summary: Mapping[str, float]) -> float:
    """Combine summary scalars into a single cost (lower = better).

    Required keys: ``nan_flag``, ``max_state_reached``, ``exploded_flag``,
    ``settle_time_s``, ``mean_goal_pos_error_lift``, ``cable_oscillation_rms``.
    """
    cost = 0.0
    cost += WEIGHT_NAN * float(summary["nan_flag"])
    cost += WEIGHT_STATE * (_MAX_STATE_TARGET - int(summary["max_state_reached"]))
    cost += WEIGHT_EXPLODED * float(summary["exploded_flag"])
    cost += WEIGHT_SETTLE * float(summary["settle_time_s"])
    cost += WEIGHT_GOAL * float(summary["mean_goal_pos_error_lift"])
    cost += WEIGHT_OSC * float(summary["cable_oscillation_rms"])
    return cost
