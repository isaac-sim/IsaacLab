# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import sys

import pytest

pytestmark = pytest.mark.unit


def test_recording_defers_command_resampling(monkeypatch, tmp_path):
    """Recording defers command resampling beyond a realistic demonstration."""
    from isaaclab_teleop.entrypoints import record_demos

    monkeypatch.setattr(sys, "argv", ["record_demos.py"])
    record_demos._parse_args(["--task", "Isaac-Reach-Franka", "--teleop_device", "spacemouse"])
    cfg, _, _ = record_demos.create_environment_config(str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60
