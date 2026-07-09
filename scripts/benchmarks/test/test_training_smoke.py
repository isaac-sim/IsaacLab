# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Smoke tests for the training benchmark entry point."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"


@pytest.mark.parametrize(
    ("library", "num_envs", "formatter", "expect_reward_series", "expect_success_rate", "expect_checkpoint"),
    [
        ("rl_games", 512, "schema", True, False, False),
        ("rsl_rl", 16, "schema,omniperf", True, False, False),
        ("sb3", 16, "schema", False, True, True),
        ("skrl", 16, "schema", True, True, False),
    ],
)
def test_training_writes_bundle(
    tmp_path,
    load_training_bundle,
    library: str,
    num_envs: int,
    formatter: str,
    expect_reward_series: bool,
    expect_success_rate: bool,
    expect_checkpoint: bool,
):
    """Each supported RL library writes its expected training benchmark data."""
    cmd = [
        str(ROOT / "isaaclab.sh"),
        "-p",
        "scripts/benchmarks/training.py",
        "--rl_library",
        library,
        "--task",
        _TASK,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        "5",
        "--seed",
        "0",
        "--benchmark_formatter",
        formatter,
        "--output_path",
        str(tmp_path),
        "presets=newton_mjwarp",
        "--headless",
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(
            f"training.py rc={result.returncode}\nSTDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )

    data = load_training_bundle(tmp_path)
    assert data["schema_version"] == "1.0"
    assert data["run"]["framework"] == library
    assert data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert data["runtime"]["startup_time_s"]["python_imports"] > 0
    assert data["runtime"]["startup_time_s"]["task_config"] > 0
    assert 1 <= data["runtime"]["iterations_completed"] <= 5
    assert data["runtime"]["total_fps"]["mean"] > 0
    assert data["learning"]["reward"]["series_per_iter"] is not None
    assert data["learning"]["reward"]["final_ema"] is not None

    if expect_reward_series:
        assert len(data["learning"]["reward"]["series_per_iter"]) >= 1
    if expect_success_rate:
        assert data["success_rate"] is not None
    if expect_checkpoint:
        assert Path(data["checkpoint_path"]).is_file()
    if "omniperf" in formatter:
        omniperf_files = list(tmp_path.glob("*_omniperf.json"))
        assert len(omniperf_files) == 1
        omniperf_data = json.loads(omniperf_files[0].read_text())
        assert omniperf_data["runtime"]["Mean Total FPS"] == pytest.approx(data["runtime"]["total_fps"]["mean"])
