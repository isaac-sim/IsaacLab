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


def test_resolve_training_checkpoint_path_uses_natural_order(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "nn"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "last_Cartpole_ep_9.pth").touch()
    checkpoint = checkpoint_dir / "last_Cartpole_ep_10.pth"
    checkpoint.touch()

    assert training._resolve_training_checkpoint_path(str(run_dir), "rl_games") == str(checkpoint)


def test_resolve_training_checkpoint_path_returns_none_without_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    assert training._resolve_training_checkpoint_path(str(run_dir), "skrl") is None
