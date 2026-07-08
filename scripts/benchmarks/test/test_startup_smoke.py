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


def test_startup_writes_startup_bundle(tmp_path):
    """The startup entry point profiles imports and writes both requested formats."""
    whitelist = tmp_path / "whitelist.yaml"
    whitelist.write_text('python_imports:\n  - "*isaaclab_tasks*importer:_walk_packages"\n')

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
        "--top_n",
        "30",
        "--output_path",
        str(tmp_path),
        "--benchmark_formatter",
        "schema,omniperf",
        "--whitelist_config",
        str(whitelist),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"startup.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")

    files = list(tmp_path.glob("*.json"))
    schema_files = [path for path in files if path.name.endswith("_schema.json")]
    omniperf_files = [path for path in files if path.name.endswith("_omniperf.json")]
    assert len(schema_files) == len(omniperf_files) == 1

    data = json.loads(schema_files[0].read_text())
    assert set(data["phases"]) == _EXPECTED_PHASES
    assert data["config"]["whitelist"] == str(whitelist)
    assert data["phases"]["python_imports"]["top_functions"]

    phase_total_s = sum(phase["total_time_s"] for phase in data["phases"].values())
    assert data["run"]["duration_s"] >= phase_total_s
    assert data["run"]["duration_s"] - phase_total_s < 0.25
    assert set(json.loads(omniperf_files[0].read_text())) >= _EXPECTED_PHASES
