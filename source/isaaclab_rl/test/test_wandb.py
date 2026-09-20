# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import isaaclab_rl.utils.wandb as wandb_utils

pytestmark = pytest.mark.unit


def _stub_wandb_run(monkeypatch: pytest.MonkeyPatch, file_names: list[str]):
    """Replace the remote wandb boundary with a minimal in-memory run."""
    requested_runs = []
    downloads = []
    files = [
        SimpleNamespace(
            name=name,
            download=lambda root, replace, name=name: downloads.append((name, root, replace)),
        )
        for name in file_names
    ]
    run = SimpleNamespace(files=lambda: files)
    api = SimpleNamespace(run=lambda path: requested_runs.append(path) or run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)
    return requested_runs, downloads


def test_recognizes_wandb_checkpoint_references():
    """Recognize wandb references without claiming ordinary checkpoint selectors."""
    assert wandb_utils.is_wandb_checkpoint("https://wandb.ai/entity/project/runs/run-id")
    assert wandb_utils.is_wandb_checkpoint("wandb:entity/project/run-id")
    assert not wandb_utils.is_wandb_checkpoint("latest")
    assert not wandb_utils.is_wandb_checkpoint(None)


@pytest.mark.parametrize(
    ("reference", "expected_file"),
    [
        ("https://wandb.ai/entity/project/runs/run-id", "model_100.pt"),
        ("https://wandb.ai/entity/project/runs/run-id?checkpoint=10", "model_10.pt"),
        ("wandb:entity/project/run-id", "model_100.pt"),
    ],
)
def test_resolves_checkpoint_reference(reference, expected_file, monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Parse supported references, select the checkpoint, and download it."""
    requested_runs, downloads = _stub_wandb_run(monkeypatch, ["model_10.pt", "model_best.pt", "model_100.pt"])

    path = wandb_utils.resolve_wandb_checkpoint(reference, download_dir=str(tmp_path))

    target_dir = str(tmp_path / "project" / "run-id")
    assert path == os.path.join(target_dir, expected_file)
    assert requested_runs == ["entity/project/run-id"]
    assert downloads == [(expected_file, target_dir, True)]


def test_rejects_unrecognized_checkpoint_reference():
    """Reject paths that are not wandb run references."""
    with pytest.raises(ValueError, match="not a recognized"):
        wandb_utils.resolve_wandb_checkpoint("logs/rsl_rl/run/model_100.pt")


def test_resolves_wandb_entity_in_precedence_order(monkeypatch: pytest.MonkeyPatch):
    """Prefer an explicit entity, followed by the two supported environment variables."""
    monkeypatch.setenv("WANDB_ENTITY", "entity")
    monkeypatch.setenv("WANDB_USERNAME", "username")
    assert wandb_utils.resolve_wandb_entity("explicit") == "explicit"
    assert wandb_utils.resolve_wandb_entity() == "entity"

    monkeypatch.delenv("WANDB_ENTITY")
    assert wandb_utils.resolve_wandb_entity() == "username"
    monkeypatch.delenv("WANDB_USERNAME")
    assert wandb_utils.resolve_wandb_entity() is None


def test_announces_new_and_existing_runs(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    """Announce a generated run id without replacing an existing one."""
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.setattr(
        wandb_utils, "wandb", SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "new-id")), raising=False
    )

    wandb_utils.announce_new_run("project", "entity")
    assert os.environ["WANDB_RUN_ID"] == "new-id"
    assert "wandb:entity/project/new-id" in capsys.readouterr().out

    monkeypatch.setenv("WANDB_RUN_ID", "existing-id")
    wandb_utils.announce_new_run("project")
    assert os.environ["WANDB_RUN_ID"] == "existing-id"
    assert "wandb:<entity>/project/existing-id" in capsys.readouterr().out


def test_get_model_checkpoint_uses_environment_entity(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Use the configured entity when a caller does not provide one."""
    requested_runs, _ = _stub_wandb_run(monkeypatch, ["model_1.pt"])
    monkeypatch.setenv("WANDB_ENTITY", "entity")

    wandb_utils.get_model_checkpoint(run_id="run-id", project="project", download_dir=str(tmp_path))

    assert requested_runs == ["entity/project/run-id"]


@pytest.mark.parametrize(
    ("file_names", "checkpoint", "message"),
    [
        (["config.yaml"], None, "No model checkpoints found"),
        (["model_10.pt"], 999, "iteration 999"),
    ],
)
def test_get_model_checkpoint_reports_missing_models(file_names, checkpoint, message, monkeypatch, tmp_path):
    """Report when a run has no usable checkpoint or lacks the requested iteration."""
    _stub_wandb_run(monkeypatch, file_names)

    with pytest.raises(ValueError, match=message):
        wandb_utils.get_model_checkpoint(
            run_id="run-id",
            project="project",
            checkpoint=checkpoint,
            wandb_entity="entity",
            download_dir=str(tmp_path),
        )


def test_get_model_checkpoint_validates_dependencies_and_entity(monkeypatch: pytest.MonkeyPatch):
    """Validate required local configuration before contacting wandb."""
    monkeypatch.delattr(wandb_utils, "wandb", raising=False)
    with pytest.raises(ImportError, match="wandb"):
        wandb_utils.get_model_checkpoint(run_id="run-id", project="project", wandb_entity="entity")

    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(), raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_USERNAME", raising=False)
    with pytest.raises(ValueError, match="wandb entity is required"):
        wandb_utils.get_model_checkpoint(run_id="run-id", project="project")
