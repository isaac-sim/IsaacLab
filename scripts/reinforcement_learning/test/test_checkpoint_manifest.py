# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RL_SCRIPTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_SCRIPTS_DIR))

from common import RUN_MANIFEST_FILENAME, resolve_checkpoint_selector, write_run_manifest  # noqa: E402


def _set_created_at(run_dir: Path, created_at: str) -> None:
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = created_at
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_write_run_manifest_records_normalized_run_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"

    write_run_manifest(
        str(run_dir),
        library="rsl_rl",
        task="example:Isaac-Cartpole-Direct-Play",
        metadata={"agent": "rsl_rl_cfg_entry_point"},
    )

    manifest = json.loads((run_dir / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert manifest["version"] == 1
    assert manifest["library"] == "rsl_rl"
    assert manifest["task"] == "Isaac-Cartpole-Direct"
    assert manifest["metadata"] == {"agent": "rsl_rl_cfg_entry_point"}
    assert manifest["created_at"]


def test_latest_selects_naturally_last_checkpoint_from_newest_compatible_run(tmp_path: Path) -> None:
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    incompatible_run = tmp_path / "incompatible"
    for run_dir in (old_run, new_run, incompatible_run):
        (run_dir / "checkpoints").mkdir(parents=True)

    write_run_manifest(str(old_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "ppo"})
    write_run_manifest(str(new_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "ppo"})
    write_run_manifest(str(incompatible_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "mappo"})
    _set_created_at(old_run, "2026-01-01T00:00:00+00:00")
    _set_created_at(new_run, "2026-01-02T00:00:00+00:00")
    _set_created_at(incompatible_run, "2026-01-03T00:00:00+00:00")

    (old_run / "checkpoints" / "agent_100.pt").touch()
    (new_run / "checkpoints" / "agent_9.pt").touch()
    expected = new_run / "checkpoints" / "agent_10.pt"
    expected.touch()
    (incompatible_run / "checkpoints" / "agent_20.pt").touch()

    checkpoint = resolve_checkpoint_selector(
        str(tmp_path),
        "latest",
        library="skrl",
        task="Isaac-Cartpole-Play",
        checkpoint_pattern=r".*\.pt",
        other_dirs=["checkpoints"],
        metadata={"algorithm": "ppo"},
    )

    assert checkpoint == str(expected.resolve())


def test_best_prefers_final_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_run_manifest(str(run_dir), library="sb3", task="Isaac-Cartpole")
    (run_dir / "model_2000_steps.zip").touch()
    expected = run_dir / "model.zip"
    expected.touch()

    checkpoint = resolve_checkpoint_selector(
        str(tmp_path),
        "best",
        library="sb3",
        task="Isaac-Cartpole",
        checkpoint_pattern=r"model(?:_.*)?\.zip",
        preferred_checkpoint_pattern=r"model\.zip",
    )

    assert checkpoint == str(expected.resolve())


def test_latest_skips_newer_run_without_checkpoint(tmp_path: Path) -> None:
    complete_run = tmp_path / "complete"
    incomplete_run = tmp_path / "incomplete"
    complete_run.mkdir()
    incomplete_run.mkdir()
    write_run_manifest(str(complete_run), library="rsl_rl", task="Isaac-Cartpole")
    write_run_manifest(str(incomplete_run), library="rsl_rl", task="Isaac-Cartpole")
    _set_created_at(complete_run, "2026-01-01T00:00:00+00:00")
    _set_created_at(incomplete_run, "2026-01-02T00:00:00+00:00")
    expected = complete_run / "model_10.pt"
    expected.touch()

    checkpoint = resolve_checkpoint_selector(
        str(tmp_path),
        "latest",
        library="rsl_rl",
        task="Isaac-Cartpole",
        checkpoint_pattern=r"model_.*\.pt",
    )

    assert checkpoint == str(expected.resolve())


def test_latest_rejects_unmanifested_historical_run(tmp_path: Path) -> None:
    historical_run = tmp_path / "2025-01-01_00-00-00"
    historical_run.mkdir()
    (historical_run / "model_10.pt").touch()

    with pytest.raises(ValueError, match="current unified training entrypoint"):
        resolve_checkpoint_selector(
            str(tmp_path),
            "latest",
            library="rsl_rl",
            task="Isaac-Cartpole",
            checkpoint_pattern=r"model_.*\.pt",
        )
