# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""End-to-end smoke tests for training and playing benchmarked policies."""

import importlib.util
import json
import subprocess
import sys
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


def _load_adapter(library: str, workflow: str):
    path = ROOT / "scripts" / "benchmarks" / library / f"benchmark_{library}_{workflow}.py"
    spec = importlib.util.spec_from_file_location(f"benchmark_{library}_{workflow}_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
@pytest.mark.parametrize(("workflow", "argument"), [("play", "--num_frames"), ("train", "--max_iterations")])
def test_adapters_reject_non_positive_workloads(library: str, workflow: str, argument: str, monkeypatch, capsys):
    """Benchmark adapters reject workloads that cannot produce timing samples."""
    module = _load_adapter(library, workflow)
    argv = ["--task", _TASK, argument, "0", "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    with pytest.raises(SystemExit) as exc_info:
        module._parse_args(argv)

    assert exc_info.value.code == 2
    assert f"argument {argument}: must be greater than zero" in capsys.readouterr().err


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
@pytest.mark.parametrize("workflow", ["play", "train"])
def test_adapters_reject_negative_warmup_steps(library: str, workflow: str, monkeypatch, capsys):
    """Benchmark adapters reject negative warm-up counts."""
    module = _load_adapter(library, workflow)
    warmup_argument = "--warmup_frames" if workflow == "play" else "--warmup_steps"
    argv = ["--task", _TASK, warmup_argument, "-1", "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    with pytest.raises(SystemExit) as exc_info:
        module._parse_args(argv)

    assert exc_info.value.code == 2
    assert f"argument {warmup_argument}: must be non-negative" in capsys.readouterr().err


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
@pytest.mark.parametrize("workflow", ["play", "train"])
def test_adapters_default_to_one_warmup_step(library: str, workflow: str, monkeypatch):
    """Benchmark adapters exclude the first environment step by default."""
    module = _load_adapter(library, workflow)
    argv = ["--task", _TASK, "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    args = module._parse_args(argv)[0]

    warmup_count = args.warmup_frames if workflow == "play" else args.warmup_steps
    assert warmup_count == 1


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
@pytest.mark.parametrize("workflow", ["play", "train"])
def test_adapters_accept_short_synchronized_step_flag(library: str, workflow: str, monkeypatch):
    """Benchmark adapters expose the concise synchronized-step diagnostic flag."""
    module = _load_adapter(library, workflow)
    argv = ["--task", _TASK, "--measure_sync_step", "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    args = module._parse_args(argv)[0]

    assert args.measure_sync_step is True


@pytest.mark.parametrize("library", ["rsl_rl", "rl_games", "skrl", "sb3"])
def test_play_adapters_accept_warmup_larger_than_measured_workload(library: str, monkeypatch):
    """Play warm-up adds calls without consuming the measured workload."""
    module = _load_adapter(library, "play")
    argv = ["--task", _TASK, "--num_frames", "2", "--warmup_frames", "3", "--headless"]
    monkeypatch.setattr(sys, "argv", ["benchmark", *argv])

    args = module._parse_args(argv)[0]

    assert args.num_frames == 2
    assert args.warmup_frames == 3


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
    assert training_data["schema_version"] == "1.2"
    assert training_data["run"]["config"]["physics_backend"] == "newton_mjwarp"
    assert training_data["runtime"]["startup_time_s"]["python_imports"] > 0
    assert training_data["runtime"]["startup_time_s"]["task_config"] > 0
    assert 1 <= training_data["runtime"]["iterations_completed"] <= max_iterations
    assert training_data["run"]["framework"] == library
    assert training_data["runtime"]["total_fps"]["mean"] > 0
    training_timing = training_data["runtime"]["environment_step_timing"]
    assert training_timing["environment_step_calls"] > 0
    assert training_timing["environment_step_fps"]["mean"] > 0
    assert training_timing["simulation_step_calls"] is None
    assert training_timing["simulation_step_time_s"] is None
    assert training_timing["outside_simulation_step_time_s"] is None
    assert training_timing["outside_simulation_step_fraction"] is None
    assert training_timing["measurement_mode"] == "host_return"
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
    play_timing = play_data["runtime"]["environment_step_timing"]
    assert play_timing["environment_step_calls"] == 250
    assert play_timing["environment_step_fps"]["mean"] > 0
    assert play_timing["simulation_step_calls"] is None
    assert play_timing["simulation_step_time_s"] is None
    assert play_timing["outside_simulation_step_time_s"] is None
    assert play_timing["outside_simulation_step_fraction"] is None
    assert play_timing["measurement_mode"] == "host_return"
    assert play_data["checkpoint_path"]
    assert play_data["reward"] is not None
    assert "mean" in play_data["reward"]

    if library == "rsl_rl":
        synchronized_play_output = tmp_path / "play_synchronized"
        _run(
            [
                str(ROOT / "isaaclab.sh"),
                "-p",
                "scripts/benchmarks/play.py",
                *common_args[:6],
                "--measure_sync_step",
                *common_args[6:],
                "--num_frames",
                "10",
                "--checkpoint",
                "latest",
                "--output_path",
                str(synchronized_play_output),
            ]
        )
        synchronized_play_data = _load_play_bundle(synchronized_play_output)
        synchronized_timing = synchronized_play_data["runtime"]["environment_step_timing"]
        assert synchronized_timing["environment_step_calls"] == 10
        assert synchronized_timing["simulation_step_calls"] > 0
        assert synchronized_timing["simulation_step_time_s"]["mean"] > 0.0
        assert synchronized_timing["outside_simulation_step_time_s"]["mean"] >= 0.0
        assert synchronized_timing["outside_simulation_step_fraction"] >= 0.0
        assert synchronized_timing["measurement_mode"] == "serialized_synchronized"

    if "omniperf" in formatter:
        training_omniperf = json.loads(next(training_output.glob("*_omniperf.json")).read_text())
        play_omniperf = json.loads(next(play_output.glob("*_omniperf.json")).read_text())
        assert training_omniperf["runtime"]["Mean Total FPS"] == pytest.approx(
            training_data["runtime"]["total_fps"]["mean"]
        )
        assert play_omniperf["runtime"]["Mean Total FPS"] == pytest.approx(play_data["runtime"]["total_fps"]["mean"])
        assert training_omniperf["benchmark_info"]["environment_step_measurement_mode"] == "host_return"
        assert play_omniperf["benchmark_info"]["environment_step_measurement_mode"] == "host_return"
        if library == "rsl_rl":
            synchronized_omniperf = json.loads(next(synchronized_play_output.glob("*_omniperf.json")).read_text())
            assert (
                synchronized_omniperf["benchmark_info"]["environment_step_measurement_mode"]
                == "serialized_synchronized"
            )
            assert "Mean Serialized Diagnostic Total FPS" in synchronized_omniperf["runtime"]
            assert "Mean Total FPS" not in synchronized_omniperf["runtime"]
