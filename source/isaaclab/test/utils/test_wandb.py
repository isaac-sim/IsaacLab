# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

import isaaclab.utils.wandb as wandb_utils

pytestmark = pytest.mark.unit


class _FakeFile:
    """Stand-in for a ``wandb.apis.public.File``."""

    def __init__(self, name: str):
        self.name = name
        self.downloaded_to: str | None = None

    def download(self, root: str, replace: bool = True):
        self.downloaded_to = root


class _FakeRun:
    def __init__(self, file_names: list[str]):
        self._files = [_FakeFile(name) for name in file_names]

    def files(self):
        return self._files


class _FakeApi:
    def __init__(self, run: _FakeRun):
        self._run = run
        self.requested_path: str | None = None

    def run(self, path: str):
        self.requested_path = path
        return self._run


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("https://wandb.ai/my_entity/my_project/runs/abc123", True),
        ("https://wandb.ai/my_entity/my_project/runs/abc123?checkpoint=100", True),
        ("wandb:my_entity/my_project/abc123", True),
        ("logs/rsl_rl/Isaac-Cartpole/2024-01-01_00-00-00/model_100.pt", False),
        ("pretrained", False),
        ("latest", False),
        (None, False),
    ],
)
def test_is_wandb_checkpoint(path, expected):
    """Test that only wandb run URLs and the wandb: shorthand are recognized."""
    assert wandb_utils.is_wandb_checkpoint(path) is expected


def test_resolve_wandb_checkpoint_parses_url_and_downloads_latest(monkeypatch: pytest.MonkeyPatch):
    """Test that a run URL without a checkpoint query downloads the latest checkpoint."""
    calls = []
    monkeypatch.setattr(
        wandb_utils,
        "get_model_checkpoint",
        lambda **kwargs: calls.append(kwargs) or "/tmp/model_200.pt",
    )

    path = wandb_utils.resolve_wandb_checkpoint("https://wandb.ai/my_entity/my_project/runs/abc123")

    assert path == "/tmp/model_200.pt"
    assert calls == [
        {
            "run_id": "abc123",
            "project": "my_project",
            "checkpoint": None,
            "wandb_entity": "my_entity",
            "download_dir": "logs/wandb_checkpoints",
        }
    ]


def test_resolve_wandb_checkpoint_parses_checkpoint_query(monkeypatch: pytest.MonkeyPatch):
    """Test that a ?checkpoint=<iteration> query selects a specific checkpoint iteration."""
    calls = []
    monkeypatch.setattr(
        wandb_utils,
        "get_model_checkpoint",
        lambda **kwargs: calls.append(kwargs) or "/tmp/model_100.pt",
    )

    wandb_utils.resolve_wandb_checkpoint("https://wandb.ai/my_entity/my_project/runs/abc123?checkpoint=100")

    assert calls[0]["checkpoint"] == 100


def test_resolve_wandb_checkpoint_parses_shorthand_uri(monkeypatch: pytest.MonkeyPatch):
    """Test that the wandb: shorthand resolves entity, project, and run id."""
    calls = []
    monkeypatch.setattr(wandb_utils, "get_model_checkpoint", lambda **kwargs: calls.append(kwargs) or "/tmp/model_1.pt")

    wandb_utils.resolve_wandb_checkpoint("wandb:my_entity/my_project/abc123")

    assert calls[0] == {
        "run_id": "abc123",
        "project": "my_project",
        "checkpoint": None,
        "wandb_entity": "my_entity",
        "download_dir": "logs/wandb_checkpoints",
    }


def test_resolve_wandb_checkpoint_rejects_unrecognized_path():
    """Test that a non-wandb path raises a clear error instead of silently failing."""
    with pytest.raises(ValueError, match="not a recognized"):
        wandb_utils.resolve_wandb_checkpoint("logs/rsl_rl/Isaac-Cartpole/model_100.pt")


@pytest.mark.parametrize(
    ("explicit", "env", "expected"),
    [
        ("explicit_entity", {"WANDB_ENTITY": "env_entity"}, "explicit_entity"),
        (None, {"WANDB_ENTITY": "env_entity"}, "env_entity"),
        (None, {"WANDB_USERNAME": "env_username"}, "env_username"),
        (None, {"WANDB_ENTITY": "env_entity", "WANDB_USERNAME": "env_username"}, "env_entity"),
        (None, {}, None),
    ],
)
def test_resolve_wandb_entity_precedence(monkeypatch: pytest.MonkeyPatch, explicit, env, expected):
    """Test that an explicit entity wins, then WANDB_ENTITY, then WANDB_USERNAME."""
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_USERNAME", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    assert wandb_utils.resolve_wandb_entity(explicit) == expected


def test_announce_new_run_pins_run_id_and_prints_shorthand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Test that a fresh run id is generated, pinned, and printed as a resumable shorthand."""
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.setattr(
        wandb_utils, "wandb", SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "newid")), raising=False
    )

    wandb_utils.announce_new_run("my_project", "my_entity")

    assert os.environ["WANDB_RUN_ID"] == "newid"
    output = capsys.readouterr().out
    assert "wandb:my_entity/my_project/newid" in output


def test_announce_new_run_respects_pinned_run_id(monkeypatch: pytest.MonkeyPatch):
    """Test that an already-set WANDB_RUN_ID (e.g. resuming a run) is not overwritten."""
    monkeypatch.setenv("WANDB_RUN_ID", "existing_id")
    monkeypatch.setattr(
        wandb_utils, "wandb", SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "newid")), raising=False
    )

    wandb_utils.announce_new_run("my_project", "my_entity")

    assert os.environ["WANDB_RUN_ID"] == "existing_id"


def test_announce_new_run_without_entity_omits_shorthand(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    """Test that an unresolved entity prints a placeholder instead of a broken shorthand."""
    monkeypatch.delenv("WANDB_RUN_ID", raising=False)
    monkeypatch.setattr(
        wandb_utils, "wandb", SimpleNamespace(util=SimpleNamespace(generate_id=lambda: "newid")), raising=False
    )

    wandb_utils.announce_new_run("my_project", None)

    output = capsys.readouterr().out
    assert "wandb:<entity>/my_project/newid" in output


def test_announce_new_run_is_a_noop_without_wandb_installed(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that announcing a run without wandb installed does not raise."""
    monkeypatch.delattr(wandb_utils, "wandb", raising=False)

    wandb_utils.announce_new_run("my_project", "my_entity")


def test_get_model_checkpoint_downloads_latest_by_default(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that the checkpoint with the highest iteration is downloaded when none is requested."""
    run = _FakeRun(["model_10.pt", "model_100.pt", "model_20.pt", "other_file.json"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)

    path = wandb_utils.get_model_checkpoint(
        run_id="abc123", project="my_project", wandb_entity="my_entity", download_dir=str(tmp_path)
    )

    assert api.requested_path == "my_entity/my_project/abc123"
    assert path == str(tmp_path / "my_project" / "abc123" / "model_100.pt")


def test_get_model_checkpoint_ignores_non_numeric_model_files(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that files like model_best.pt do not crash iteration parsing or get picked as latest."""
    run = _FakeRun(["model_10.pt", "model_best.pt", "model_final.pt"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)

    path = wandb_utils.get_model_checkpoint(
        run_id="abc123", project="my_project", wandb_entity="my_entity", download_dir=str(tmp_path)
    )

    assert path == str(tmp_path / "my_project" / "abc123" / "model_10.pt")


def test_get_model_checkpoint_downloads_requested_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that a specific checkpoint iteration is selected when requested."""
    run = _FakeRun(["model_10.pt", "model_100.pt", "model_20.pt"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)

    path = wandb_utils.get_model_checkpoint(
        run_id="abc123", project="my_project", checkpoint=20, wandb_entity="my_entity", download_dir=str(tmp_path)
    )

    assert path == str(tmp_path / "my_project" / "abc123" / "model_20.pt")


def test_get_model_checkpoint_uses_wandb_entity_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that the entity falls back to the WANDB_ENTITY environment variable."""
    run = _FakeRun(["model_1.pt"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)
    monkeypatch.setenv("WANDB_ENTITY", "env_entity")

    wandb_utils.get_model_checkpoint(run_id="abc123", project="my_project", download_dir=str(tmp_path))

    assert api.requested_path == "env_entity/my_project/abc123"


def test_get_model_checkpoint_raises_on_missing_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that requesting an unavailable checkpoint iteration raises a clear error."""
    run = _FakeRun(["model_10.pt"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)

    with pytest.raises(ValueError, match="iteration 999"):
        wandb_utils.get_model_checkpoint(
            run_id="abc123", project="my_project", checkpoint=999, wandb_entity="my_entity", download_dir=str(tmp_path)
        )


def test_get_model_checkpoint_raises_when_no_checkpoints_found(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Test that a run with no model files raises a clear error."""
    run = _FakeRun(["config.yaml"])
    api = _FakeApi(run)
    monkeypatch.setattr(wandb_utils, "wandb", SimpleNamespace(Api=lambda: api), raising=False)

    with pytest.raises(ValueError, match="No model checkpoints found"):
        wandb_utils.get_model_checkpoint(
            run_id="abc123", project="my_project", wandb_entity="my_entity", download_dir=str(tmp_path)
        )


def test_get_model_checkpoint_raises_import_error_without_wandb(monkeypatch: pytest.MonkeyPatch):
    """Test that a missing wandb installation raises ImportError instead of AttributeError."""
    monkeypatch.delattr(wandb_utils, "wandb", raising=False)

    with pytest.raises(ImportError, match="wandb"):
        wandb_utils.get_model_checkpoint(run_id="abc123", project="my_project", wandb_entity="my_entity")
