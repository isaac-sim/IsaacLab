# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint reporting for training benchmark adapters."""

from pathlib import Path

import pytest

from isaaclab.benchmark.entrypoints import training


@pytest.mark.parametrize(
    ("backend", "subdir", "filename"),
    (
        ("rl_games", "nn", "last_Cartpole_ep_50.pth"),
        ("skrl", "checkpoints", "agent_4800.pt"),
        ("skrl", "checkpoints", "best_agent.pt"),
    ),
)
def test_resolve_training_checkpoint_path_matches_backend_filename(
    tmp_path: Path, backend: str, subdir: str, filename: str
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / subdir
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / filename
    checkpoint.touch()

    assert training._resolve_training_checkpoint_path(str(run_dir), backend) == str(checkpoint)


def test_resolve_training_checkpoint_path_returns_none_without_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert training._resolve_training_checkpoint_path(str(run_dir), "skrl") is None


@pytest.mark.parametrize(
    ("entrypoint", "source"),
    [("train", "explicit"), ("benchmark", "explicit"), ("train", "resume"), ("benchmark", "resume")],
)
def test_rsl_rl_entrypoints_load_checkpoint(tmp_path: Path, monkeypatch, entrypoint: str, source: str) -> None:
    """Both callers load an explicit teacher or an existing student before starting learning."""
    import contextlib
    import sys
    from unittest.mock import MagicMock

    from isaaclab import app, benchmark
    from isaaclab.benchmark import capture
    from isaaclab.benchmark.entrypoints.backends.rsl_rl import benchmark_train_rsl_rl

    from isaaclab_rl import rsl_rl
    from isaaclab_rl.entrypoints import common
    from isaaclab_rl.entrypoints.backends import cli_args_rsl_rl, runner_rsl_rl, train_rsl_rl
    from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg

    from isaaclab_tasks import utils

    monkeypatch.chdir(tmp_path)
    checkpoint = (
        tmp_path / "teacher.pt" if source == "explicit" else tmp_path / "logs/rsl_rl/student/2000-01-01/model_10.pt"
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    # Keep the real CLI update and checkpoint policy; replace simulation, logging, and the runner.
    argv = ["--task", "Isaac-Cartpole", "--device", "cpu"]
    if source == "explicit":
        argv += ["--checkpoint", str(checkpoint)]
    monkeypatch.setattr(sys, "argv", ["train.py"])
    args = train_rsl_rl._parse_args(argv)
    agent_cfg = RslRlDistillationRunnerCfg(
        experiment_name="student", seed=1, num_steps_per_env=1, max_iterations=1, resume=source == "resume"
    )
    env_cfg = MagicMock()
    env_cfg.scene.num_envs = 1
    env = MagicMock()
    runner = MagicMock()
    runner.load.side_effect = RuntimeError("checkpoint loaded")
    runner.learn.side_effect = AssertionError("Learning started before loading the checkpoint")
    monkeypatch.setattr(runner_rsl_rl, "create_rsl_rl_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(rsl_rl, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    for owner in (common, train_rsl_rl):
        for name in ("pre_launch_video_config", "apply_env_overrides", "apply_video_recording", "show_run_summary"):
            monkeypatch.setattr(owner, name, lambda *args, **kwargs: None)
        monkeypatch.setattr(owner, "create_isaaclab_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(common, "wrap_sensor_capture", lambda env, *args: env)
    monkeypatch.setattr(train_rsl_rl, "wrap_sensor_capture", lambda env, *args: env)
    monkeypatch.setattr(train_rsl_rl, "resolve_task_config", lambda *args: (env_cfg, agent_cfg))
    monkeypatch.setattr(utils, "resolve_task_config", lambda *args: (env_cfg, agent_cfg))
    monkeypatch.setattr(train_rsl_rl, "startup_screen", MagicMock())
    monkeypatch.setattr(train_rsl_rl, "launch_simulation", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(app, "launch_simulation", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(train_rsl_rl, "create_rsl_rl_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(runner_rsl_rl, "RslRlVecEnvWrapper", lambda env, **kwargs: env)
    monkeypatch.setattr(benchmark, "BaseIsaacLabBenchmark", MagicMock())
    monkeypatch.setattr(capture, "run_config_from_env_cfg", lambda cfg: None)
    args.benchmark_formatter = "schema"
    args.output_path = str(tmp_path)
    args.measure_sync_step = False
    args.warmup_steps = 0
    monkeypatch.setattr(benchmark_train_rsl_rl, "_parse_args", lambda argv: (args, [], cli_args_rsl_rl))

    with pytest.raises(RuntimeError, match="checkpoint loaded"):
        train_rsl_rl._run(args) if entrypoint == "train" else benchmark_train_rsl_rl.run([])

    assert runner.load.call_args.args == (str(checkpoint),)
    assert runner.load.call_args.kwargs["map_location"] == "cpu"
    assert agent_cfg.load_checkpoint == RslRlDistillationRunnerCfg().load_checkpoint
    runner.learn.assert_not_called()


def test_training_checkpoint_policy_has_one_owner() -> None:
    """Entrypoints must not own another checkpoint policy or import one another."""
    import ast

    from isaaclab.benchmark.entrypoints.backends.rsl_rl import benchmark_train_rsl_rl

    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    for module in (train_rsl_rl, benchmark_train_rsl_rl):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name in {"_resolve_checkpoint", "_resolve_resume_path"}
            for node in ast.walk(tree)
        )
        if module is benchmark_train_rsl_rl:
            assert not any(
                isinstance(node, ast.ImportFrom) and (node.module or "").endswith("train_rsl_rl")
                for node in ast.walk(tree)
            )
