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


@pytest.mark.parametrize("measure_sync_step", [False, True], ids=["default", "synchronized_breakdown"])
def test_runtime_writes_all_requested_formats(tmp_path, measure_sync_step: bool):
    """The runtime entry point writes schema and OmniPerf data in one run."""
    cmd = [
        str(ROOT / "isaaclab.sh"),
        "-p",
        "scripts/benchmarks/runtime.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--num_steps",
        "20",
        "--warmup_steps",
        "0",
        "--seed",
        "0",
        "--device",
        "cpu",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,omniperf",
        *(["--measure_sync_step"] if measure_sync_step else []),
        "presets=newton_mjwarp",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(
            f"runtime benchmark rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}"
        )

    device_lines = [line for line in res.stdout.splitlines() if "Environment device" in line]
    assert device_lines and device_lines[-1].endswith(": cpu"), f"unexpected device output: {device_lines}"

    files = sorted(tmp_path.glob("*.json"))
    schema_files = [path for path in files if path.name.endswith("_schema.json")]
    omniperf_files = [path for path in files if path.name.endswith("_omniperf.json")]
    assert len(schema_files) == len(omniperf_files) == 1

    schema_data = json.loads(schema_files[0].read_text())
    assert schema_data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert schema_data["runtime"]["iterations_completed"] == 20
    assert schema_data["extra"] is None
    assert schema_data["runtime"]["startup_time_s"]["first_step"] > 0.0
    timing = schema_data["runtime"]["environment_step_timing"]
    assert timing["warmup_steps"] == 0
    assert timing["environment_step_calls"] == 20
    assert timing["environment_step_fps"]["mean"] > 0
    if measure_sync_step:
        assert timing["simulation_step_calls"] > 0
        assert timing["simulation_step_time_s"]["mean"] > 0.0
        assert timing["outside_simulation_step_time_s"]["mean"] >= 0.0
        assert timing["outside_simulation_step_fraction"] >= 0.0
        assert timing["measurement_mode"] == "serialized_synchronized"
    else:
        assert timing["simulation_step_calls"] is None
        assert timing["simulation_step_time_s"] is None
        assert timing["outside_simulation_step_time_s"] is None
        assert timing["outside_simulation_step_fraction"] is None
        assert timing["measurement_mode"] == "host_return"
    assert timing["environment_step_time_s"]["std"] >= 0.0
    omniperf_data = json.loads(omniperf_files[0].read_text())
    assert omniperf_data["benchmark_info"]["environment_step_measurement_mode"] == timing["measurement_mode"]
    assert omniperf_data["benchmark_info"]["environment_step_warmup_steps"] == 0
    if measure_sync_step:
        assert "Mean Serialized Diagnostic Total FPS" in omniperf_data["runtime"]
        assert "Mean Total FPS" not in omniperf_data["runtime"]
    else:
        assert "Mean Total FPS" in omniperf_data["runtime"]
        assert "Mean Serialized Diagnostic Total FPS" not in omniperf_data["runtime"]
