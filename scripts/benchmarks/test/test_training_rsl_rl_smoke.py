# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke test for scripts/benchmarks/training.py with --rl_library rsl_rl."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"


def test_training_rsl_rl_writes_training_bundle(tmp_path, load_training_bundle):
    sh = ROOT / "isaaclab.sh"
    cmd = [
        str(sh),
        "-p",
        "scripts/benchmarks/training.py",
        "--rl_library",
        "rsl_rl",
        "--task",
        _TASK,
        "--num_envs",
        "16",
        "--max_iterations",
        "5",
        "--seed",
        "0",
        "--benchmark_formatter",
        "schema,omniperf",
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=900)
    if res.returncode != 0:
        pytest.fail(f"training.py rc={res.returncode}\nSTDOUT:\n{res.stdout[-2000:]}\nSTDERR:\n{res.stderr[-2000:]}")
    data = load_training_bundle(tmp_path)
    omniperf_files = list(tmp_path.glob("*_omniperf.json"))
    assert len(omniperf_files) == 1
    omniperf_data = json.loads(omniperf_files[0].read_text())
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] == "rsl_rl"
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert data["runtime"]["startup_time_s"]["python_imports"] > 0
    assert data["runtime"]["startup_time_s"]["task_config"] > 0
    assert 1 <= data["runtime"]["iterations_completed"] <= 5
    assert data["runtime"]["total_fps"]["mean"] > 0
    assert data["learning"]["reward"]["series_per_iter"] is not None
    assert len(data["learning"]["reward"]["series_per_iter"]) >= 1
    assert data["learning"]["reward"]["final_ema"] is not None
    assert omniperf_data["runtime"]["Mean Total FPS"] == pytest.approx(data["runtime"]["total_fps"]["mean"])
    assert omniperf_data["startup"]["Python Imports Time"] > 0
    assert omniperf_data["startup"]["Task Creation and Start Time"] > 0
    assert omniperf_data["train"]["Last Reward"] == pytest.approx(data["learning"]["reward"]["final_raw"])
