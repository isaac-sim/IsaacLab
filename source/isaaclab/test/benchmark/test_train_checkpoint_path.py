# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Checkpoint reporting for training benchmark adapters."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def _rsl_rl_resume_path(tmp_path: Path, *, distillation: bool, checkpoint: str | None = None, resume: bool = False):
    from isaaclab.benchmark.entrypoints.backends.rsl_rl.benchmark_train_rsl_rl import _resolve_resume_path

    from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlOnPolicyRunnerCfg

    args_cli = SimpleNamespace(checkpoint=checkpoint, task="Isaac-Velocity-Rough-G1-Depth-Student", agent=None)
    # The resolver dispatches on the config type, so build spec'd stand-ins instead of full runner configs.
    agent_cfg = MagicMock(spec=RslRlDistillationRunnerCfg if distillation else RslRlOnPolicyRunnerCfg)
    agent_cfg.resume = resume
    agent_cfg.load_run = ".*"
    agent_cfg.load_checkpoint = checkpoint or r"model_.*\.pt"
    return _resolve_resume_path(args_cli, agent_cfg, str(tmp_path / "logs" / "student"))


def test_rsl_rl_distillation_benchmark_requires_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """Distillation without ``--checkpoint`` fails on the missing teacher, not on an empty student log."""
    with pytest.raises(ValueError, match="requires --checkpoint"):
        _rsl_rl_resume_path(tmp_path, distillation=True)


def test_rsl_rl_distillation_benchmark_loads_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """An explicit teacher file is used without searching the student experiment's runs."""
    teacher = tmp_path / "teacher" / "model_1499.pt"
    teacher.parent.mkdir()
    teacher.touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(teacher)) == str(teacher)
    with pytest.raises(FileNotFoundError):
        _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(teacher.with_name("x.pt")))


def test_rsl_rl_distillation_benchmark_loads_latest_checkpoint_of_teacher_run_directory(tmp_path: Path) -> None:
    """A teacher run directory resolves to its newest ``model_*.pt``, not to a pattern built from the directory path."""
    run = tmp_path / "teacher" / "2026-11-17_10-00-00"
    run.mkdir(parents=True)
    for name in ("model_5.pt", "model_10.pt", "events.out.tfevents"):
        (run / name).touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(run)) == str(run / "model_10.pt")


def test_rsl_rl_benchmark_resume_searches_experiment_runs(tmp_path: Path) -> None:
    """A configured resume without ``--checkpoint`` continues the newest run, not an older run with later iterations."""
    older = tmp_path / "logs" / "student" / "2026-11-16_10-00-00" / "model_99.pt"
    older.parent.mkdir(parents=True)
    older.touch()
    checkpoint = tmp_path / "logs" / "student" / "2026-11-17_10-00-00" / "model_10.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=False, resume=True) == str(checkpoint)
    assert _rsl_rl_resume_path(tmp_path, distillation=False) is None
