# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression coverage for recording command configuration."""

import math
import runpy
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_recording_defers_command_resampling(monkeypatch, tmp_path):
    """Recording defers command resampling beyond a realistic demonstration."""
    script = Path(__file__).resolve().parents[3] / "scripts/tools/record_demos.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--task", "Isaac-Reach-Franka", "--teleop_device", "spacemouse"])
    recorder = runpy.run_path(str(script), run_name="record_demos_test")
    cfg, _, _ = recorder["create_environment_config"](str(tmp_path), "reach")

    resampling_time_range = cfg.commands.ee_pose.resampling_time_range
    assert all(math.isfinite(bound) for bound in resampling_time_range)
    assert min(resampling_time_range) > 24 * 60 * 60
