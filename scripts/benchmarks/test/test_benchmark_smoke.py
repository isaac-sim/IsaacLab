# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke tests for training and playing benchmarked policies."""

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]

_TASK = "Isaac-Cartpole-Direct"
_PLAY_BUNDLE_KEYS = {"run", "versions", "hardware", "runtime", "resources"}


def _load_play_bundle(output_path: Path) -> dict:
    """Load the schema play bundle from an output directory."""
    for path in output_path.glob("*.json"):
        data = json.loads(path.read_text())
        if data.keys() >= _PLAY_BUNDLE_KEYS:
            return data
    pytest.fail(f"no play bundle found in {output_path}")


def _run(command: list[str]) -> None:
    """Run a benchmark command and report its trailing output on failure."""
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        pytest.fail(
            f"{Path(command[2]).name} rc={result.returncode}\n"
            f"STDOUT:\n{result.stdout[-2000:]}\nSTDERR:\n{result.stderr[-2000:]}"
        )


@pytest.mark.parametrize(
    (
        "library",
        "num_envs",
        "max_iterations",
        "formatter",
        "expect_reward_series",
        "expect_success_rate",
        "expect_checkpoint",
    ),
    [
        ("rl_games", 512, 20, "schema", True, False, False),
        ("rsl_rl", 16, 20, "schema,omniperf", True, False, False),
        ("sb3", 16, 70, "schema", False, True, True),
        ("skrl", 16, 20, "schema", True, True, False),
    ],
)
def test_training_and_play_write_bundles(
    tmp_path,
    load_training_bundle,
    library: str,
    num_envs: int,
    max_iterations: int,
    formatter: str,
    expect_reward_series: bool,
    expect_success_rate: bool,
    expect_checkpoint: bool,
):
    """Each RL library trains and plays a policy with benchmark output."""
    training_output = tmp_path / "training"
    play_output = tmp_path / "play"
    common_args = [
        "--rl_library",
        library,
        "--task",
        _TASK,
        "--num_envs",
        str(num_envs),
        "presets=newton_mjwarp",
        "--headless",
        "--benchmark_formatter",
        formatter,
    ]

    _run(
        [
            str(ROOT / "isaaclab.sh"),
            "-p",
            "scripts/benchmarks/training.py",
            *common_args[:6],
            "--max_iterations",
            str(max_iterations),
            *common_args[6:],
            "--output_path",
            str(training_output),
        ]
    )
    training_data = load_training_bundle(training_output)
    assert training_data["schema_version"] == "1.0"
    assert training_data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert training_data["runtime"]["startup_time_s"]["python_imports"] > 0
    assert training_data["runtime"]["startup_time_s"]["task_config"] > 0
    assert 1 <= training_data["runtime"]["iterations_completed"] <= max_iterations
    assert training_data["run"]["framework"] == library
    assert training_data["runtime"]["total_fps"]["mean"] > 0
    assert training_data["learning"]["reward"]["series_per_iter"] is not None
    assert training_data["learning"]["reward"]["final_ema"] is not None
    if expect_reward_series:
        assert len(training_data["learning"]["reward"]["series_per_iter"]) >= 1
    if expect_success_rate:
        assert training_data["success_rate"] is not None
    if expect_checkpoint:
        assert Path(training_data["checkpoint_path"]).is_file()

    _run(
        [
            str(ROOT / "isaaclab.sh"),
            "-p",
            "scripts/benchmarks/play.py",
            *common_args,
            "--num_frames",
            "250",
            "--checkpoint",
            "latest",
            "--output_path",
            str(play_output),
        ]
    )
    play_data = _load_play_bundle(play_output)
    assert play_data["run"]["framework"] == library
    assert play_data["runtime"]["total_fps"]["mean"] > 0
    assert play_data["checkpoint_path"]
    assert play_data["reward"] is not None
    assert "mean" in play_data["reward"]

    if "omniperf" in formatter:
        training_omniperf = json.loads(next(training_output.glob("*_omniperf.json")).read_text())
        play_omniperf = json.loads(next(play_output.glob("*_omniperf.json")).read_text())
        assert training_omniperf["runtime"]["Mean Total FPS"] == pytest.approx(
            training_data["runtime"]["total_fps"]["mean"]
        )
        assert play_omniperf["runtime"]["Mean Total FPS"] == pytest.approx(play_data["runtime"]["total_fps"]["mean"])
