# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint reporting for training benchmark adapters."""

from pathlib import Path
from types import SimpleNamespace

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


def _rsl_rl_resume_path(tmp_path: Path, *, class_name: str, checkpoint: str | None = None, resume: bool = False):
    from isaaclab.benchmark.entrypoints.backends.rsl_rl.benchmark_train_rsl_rl import _resolve_resume_path

    args_cli = SimpleNamespace(checkpoint=checkpoint, task="Isaac-Velocity-Rough-G1-Depth-Student", agent=None)
    agent_cfg = SimpleNamespace(
        class_name=class_name, resume=resume, load_run=".*", load_checkpoint=checkpoint or r"model_.*\.pt"
    )
    return _resolve_resume_path(args_cli, agent_cfg, str(tmp_path / "logs" / "student"))


def test_rsl_rl_distillation_benchmark_requires_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """Distillation without ``--checkpoint`` fails on the missing teacher, not on an empty student log."""
    with pytest.raises(ValueError, match="requires --checkpoint"):
        _rsl_rl_resume_path(tmp_path, class_name="DistillationRunner")


def test_rsl_rl_distillation_benchmark_loads_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """An explicit teacher file is used without searching the student experiment's runs."""
    teacher = tmp_path / "teacher" / "model_1499.pt"
    teacher.parent.mkdir()
    teacher.touch()

    assert _rsl_rl_resume_path(tmp_path, class_name="DistillationRunner", checkpoint=str(teacher)) == str(teacher)
    with pytest.raises(FileNotFoundError):
        _rsl_rl_resume_path(tmp_path, class_name="DistillationRunner", checkpoint=str(teacher.with_name("x.pt")))


def test_rsl_rl_benchmark_resume_searches_experiment_runs(tmp_path: Path) -> None:
    """A configured resume without ``--checkpoint`` continues this experiment's latest run."""
    checkpoint = tmp_path / "logs" / "student" / "2026-11-17_10-00-00" / "model_10.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert _rsl_rl_resume_path(tmp_path, class_name="OnPolicyRunner", resume=True) == str(checkpoint)
    assert _rsl_rl_resume_path(tmp_path, class_name="OnPolicyRunner") is None
