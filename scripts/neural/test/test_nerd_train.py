# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``scripts/neural/train.py``, which trains the model of a task on the data of ``record.py``."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("newton_neural", reason="needs the newton-neural extra")

import torch
from newton_neural.data import TrajectoryDataset
from newton_neural.models import NeRDBase
from newton_neural.solvers import SolverNeRD

from isaaclab.utils import validate

from scripts.neural import train

# The helpers of the ``isaaclab_neural`` tests resolve the tasks and launch the simulation.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "source" / "isaaclab_neural" / "test"))

from nerd_test_utils import (  # noqa: E402
    FRANKA_REACH,
    WORLD_COUNT,
    cuda_device,
    environment,
    nerd_preset,
    record_dataset,
    task_cfg,
)

pytestmark = [pytest.mark.integration, pytest.mark.kitless]


def _preset_checkpoint() -> Path:
    """The file that the preset of the model of the reach task loads, relative to the working directory."""
    env_cfg = task_cfg(FRANKA_REACH, 1, "cpu", overrides=(f"physics={nerd_preset(FRANKA_REACH)}",))
    checkpoint = Path(env_cfg.sim.physics.solver_cfg.checkpoint)
    assert not checkpoint.is_absolute()
    return checkpoint


@pytest.fixture(scope="module")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A working directory from which the reach task was recorded without naming a file, for all tests here."""
    folder = tmp_path_factory.mktemp("reach")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        record_dataset(FRANKA_REACH, None, num_envs=64, num_frames=128, device=cuda_device())
    return folder


def _train(workspace: Path, checkpoint: Path, *settings: str) -> None:
    """Train for two steps on the dataset in ``workspace``, with ``settings`` of the recipe on the command line.

    The trained model goes to the file ``checkpoint``.
    """
    dataset = workspace / _preset_checkpoint().parent / "dataset"
    files = ["--dataset", str(dataset), "--checkpoint", str(checkpoint)]
    train.main(["--task", FRANKA_REACH, *files, "--max_steps", "2", "--device", "cpu", *settings])


def test_default_files(workspace: Path, monkeypatch: pytest.MonkeyPatch):
    """Train where the task was recorded, without naming a file, and expect the task to run on the model."""
    device = cuda_device()
    monkeypatch.chdir(workspace)
    checkpoint = _preset_checkpoint()
    env_cfg = task_cfg(FRANKA_REACH, WORLD_COUNT, device, overrides=(f"physics={nerd_preset(FRANKA_REACH)}",))

    dataset = TrajectoryDataset(checkpoint.parent / "dataset")
    untrained = NeRDBase.load(dataset.path / checkpoint.name)
    assert not checkpoint.exists()
    train.main(["--task", FRANKA_REACH, "--max_steps", "50", "--device", device])

    # Training fits the normalization, which takes the frame from the data, and changes the weights.
    trained = NeRDBase.load(checkpoint)
    assert not untrained.fitted
    assert trained.fitted
    assert trained.frame_dt == dataset.frame_dt
    weights = zip(trained.parameters(), untrained.parameters(), strict=True)
    assert any(not torch.equal(after, before) for after, before in weights)

    # The preset of the model finds the file, and the task runs on it.
    validate(env_cfg)
    with environment(FRANKA_REACH, env_cfg) as env:
        assert isinstance(env.sim.backend_manager.backend.solver, SolverNeRD)
        env.reset()
        for _ in range(8):
            action = torch.randn(WORLD_COUNT, env.action_manager.total_action_dim, device=device)
            observations, *_ = env.step(action)
        assert observations["policy"].isfinite().all()


def test_model_is_saved_before_the_report(workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A report that fails after the training leaves the trained model in its file."""

    def failing_report(*args: Any) -> None:
        raise RuntimeError("no report")

    monkeypatch.setattr(train, "report", failing_report)
    checkpoint = tmp_path / "model.pt"
    with pytest.raises(RuntimeError, match="no report"):
        _train(workspace, checkpoint)
    assert NeRDBase.load(checkpoint).fitted


def test_settings_that_cannot_train_end_before_the_training(
    workspace: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Settings that the dataset cannot serve end with their reason on the command line, before any training."""
    checkpoint = tmp_path / "model.pt"
    requests = {
        # The report rolls the model out, which takes at least one horizon, none of them negative, and episodes
        # that are long enough.
        "train.horizons, got ()": ["agent.train.horizons=[]"],
        "train.horizons, got (1, -1)": ["agent.train.horizons=[1,-1]"],
        "the held out robots have no episode of": ["agent.train.horizons=[1,200]"],
        # The model and its data follow the recipe. Only the setting with another value is named, not the recorded
        # value of a tuple written as a list, not the sizes of the recording or the physics it came from, which the
        # recording decides, and not the settings of the training, which a run may change.
        "was recorded with other values of state_speed than this run names": [
            "agent.state_speed=2.5",
            "agent.hold_frames=[1,60]",
            "agent.record.num_frames=7",
            "agent.reference_preset=newton_steel",
            "agent.train.batch_size=3",
        ],
    }
    for reason, settings in requests.items():
        with pytest.raises(SystemExit):
            _train(workspace, checkpoint, *settings)
        captured = capsys.readouterr()
        assert reason in captured.err
        assert "Training for" not in captured.out
        assert not checkpoint.exists()


def test_missing_dataset_names_the_record_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Training without a dataset of the record script ends with the command that records one."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        train.main(["--task", FRANKA_REACH])
    assert f"scripts/neural/record.py --task {FRANKA_REACH}" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []

    # A folder with a model file and without the settings of a recording is no dataset of the record script either.
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / _preset_checkpoint().name).touch()
    with pytest.raises(SystemExit):
        train.main(["--task", FRANKA_REACH, "--dataset", str(dataset)])
    assert f"{dataset} holds no dataset of record.py" in capsys.readouterr().err
