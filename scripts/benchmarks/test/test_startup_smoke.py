# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for the startup benchmark entry point."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"

_EXPECTED_PHASES = {"app_launch", "python_imports", "task_config", "env_creation", "first_step"}


def test_startup_writes_startup_bundle(tmp_path, require_isaacsim):
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/startup.py",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--seed",
        "0",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"startup.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1, f"expected one JSON file, found {[path.name for path in files]}"
    data = json.loads(files[0].read_text())

    assert data["schema_version"] == "1.0", f"unexpected schema_version: {data['schema_version']}"
    assert data["run"]["framework"] is None, "startup bundle should have framework=null"
    assert data["run"]["duration_s"] >= sum(phase["total_time_s"] for phase in data["phases"].values())

    assert set(data["phases"].keys()) == _EXPECTED_PHASES, f"unexpected phases: {set(data['phases'].keys())}"

    for phase_name, phase in data["phases"].items():
        assert phase["total_time_s"] > 0, f"phase '{phase_name}' has total_time_s <= 0"

    assert any(
        isinstance(function.get("calls"), int)
        for phase in data["phases"].values()
        for function in phase.get("top_functions", [])
    )

    assert "config" in data, "StartupBundle missing 'config' field"
    assert isinstance(data["config"]["top_n"], int), "config.top_n should be an integer"
