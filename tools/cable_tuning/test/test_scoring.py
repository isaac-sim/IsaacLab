# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the run-summary scorer."""

from __future__ import annotations

from tools.cable_tuning.scoring import score_summary


def _summary(**overrides):
    base = {
        "nan_flag": 0,
        "max_state_reached": 5,
        "exploded_flag": 0,
        "settle_time_s": 1.0,
        "mean_goal_pos_error_lift": 0.04,
        "cable_oscillation_rms": 0.5,
    }
    base.update(overrides)
    return base


def test_score_perfect_run_is_small() -> None:
    cost = score_summary(_summary(settle_time_s=0.0, mean_goal_pos_error_lift=0.0, cable_oscillation_rms=0.0))
    assert cost == 0.0


def test_nan_dominates() -> None:
    nan_cost = score_summary(_summary(nan_flag=1))
    perfect = score_summary(_summary())
    assert nan_cost > perfect + 1e5  # NaN swamps every other term


def test_incomplete_state_dominates_settle() -> None:
    incomplete = score_summary(_summary(max_state_reached=3))
    slow_but_complete = score_summary(_summary(settle_time_s=5.0))
    assert incomplete > slow_but_complete


def test_exploded_dominates_tracking() -> None:
    exploded = score_summary(_summary(exploded_flag=1))
    poor_track = score_summary(_summary(mean_goal_pos_error_lift=1.0))
    assert exploded > poor_track


def test_score_components_ordering() -> None:
    a = score_summary(_summary(settle_time_s=1.0))
    b = score_summary(_summary(settle_time_s=2.0))
    assert b > a
    a = score_summary(_summary(mean_goal_pos_error_lift=0.02))
    b = score_summary(_summary(mean_goal_pos_error_lift=0.10))
    assert b > a
