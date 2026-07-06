# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the runtime benchmark entry point."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"


def test_runtime_writes_all_requested_formats(tmp_path):
    """The runtime entry point writes schema and OmniPerf data in one run."""
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_frames",
        "20",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,omniperf",
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"runtime.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    device_lines = [line for line in res.stdout.splitlines() if "Environment device" in line]
    assert device_lines and device_lines[-1].endswith(": cpu"), f"unexpected device output: {device_lines}"

    files = sorted(tmp_path.glob("*.json"))
    schema_files = [path for path in files if path.name.endswith("_schema.json")]
    omniperf_files = [path for path in files if path.name.endswith("_omniperf.json")]
    assert len(schema_files) == len(omniperf_files) == 1

    schema_data = json.loads(schema_files[0].read_text())
    assert schema_data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert "runtime" in json.loads(omniperf_files[0].read_text())
