# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CLI tests for ``scripts/benchmarks/benchmark_cameras.py`` that do not launch a simulator."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = ROOT / "scripts" / "benchmarks" / "benchmark_cameras.py"


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, cwd=ROOT, timeout=30
    )


def test_help_describes_unified_camera_interface():
    result = _run_cli(["--help"])

    assert result.returncode == 0
    assert "--num_cameras" in result.stdout
    assert "Deprecated alias for --num_cameras" in result.stdout


def test_cli_rejects_camera_count_that_cannot_be_distributed_across_environments():
    result = _run_cli(
        ["--task", "Isaac-Cartpole", "--num_cameras", "3", "--task_num_cameras_per_env", "2"]
    )

    assert result.returncode == 2
    assert "camera count must be divisible" in result.stderr


def test_cli_rejects_negative_count_even_when_another_camera_kind_is_selected():
    result = _run_cli(["--num_cameras", "-1", "--num_ray_caster_cameras", "1"])

    assert result.returncode == 2
    assert "--num_cameras cannot be negative" in result.stderr


def test_cli_rejects_unknown_options_instead_of_forwarding_them_to_hydra():
    result = _run_cli(["--num_cameras", "1", "--headless"])

    assert result.returncode == 2
    assert "unrecognized arguments: --headless" in result.stderr
