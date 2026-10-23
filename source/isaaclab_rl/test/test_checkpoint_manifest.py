# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from isaaclab_rl.entrypoints.common import RUN_MANIFEST_FILENAME, resolve_checkpoint_selector, write_run_manifest
from isaaclab_rl.entrypoints.policy_ghosts import _launch_argv, discover_checkpoints, relaunch_argv


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
        task="example:Isaac-Cartpole-Direct",
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
    incomplete_run = tmp_path / "incomplete"
    for run_dir in (old_run, new_run, incompatible_run, incomplete_run):
        (run_dir / "checkpoints").mkdir(parents=True)

    write_run_manifest(str(old_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "ppo"})
    write_run_manifest(str(new_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "ppo"})
    write_run_manifest(str(incompatible_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "mappo"})
    write_run_manifest(str(incomplete_run), library="skrl", task="Isaac-Cartpole", metadata={"algorithm": "ppo"})
    _set_created_at(incomplete_run, "2026-01-04T00:00:00+00:00")
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
        task="Isaac-Cartpole",
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


def test_recursive_selector_resolves_rlinf_checkpoint(tmp_path: Path, monkeypatch) -> None:
    """RLinf accepts selectors and paths, but never guesses between checkpoint files."""
    from isaaclab_rl.entrypoints.backends.cli_args_rlinf import resolve_rlinf_checkpoint

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    for step in (2, 10):
        expected = run_dir / "checkpoints" / f"global_step_{step}" / "actor" / "model_state_dict" / "full_weights.pt"
        expected.parent.mkdir(parents=True)
        expected.touch()
    write_run_manifest(str(run_dir), library="rlinf", task="Isaac-Cartpole", metadata={"config_name": "ppo"})
    kwargs = dict(log_root_path=str(tmp_path), task="Isaac-Cartpole", config_name="ppo")
    paths = (expected, expected.parents[2])
    for checkpoint in ("latest", "best", *(str(path.relative_to(tmp_path)) for path in paths)):
        assert resolve_rlinf_checkpoint(checkpoint, **kwargs) == str(expected)
    with pytest.raises(ValueError):
        resolve_rlinf_checkpoint(str(run_dir), **kwargs)
    with pytest.raises(FileNotFoundError):
        resolve_rlinf_checkpoint("missing.pt", **kwargs)


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


def test_discover_checkpoints_lists_compatible_runs_newest_first_in_natural_order(tmp_path: Path) -> None:
    old_run, new_run, other_task_run = tmp_path / "old", tmp_path / "new", tmp_path / "other"
    for run_dir, task in ((old_run, "Isaac-Cartpole"), (new_run, "Isaac-Cartpole"), (other_task_run, "Isaac-Ant")):
        run_dir.mkdir()
        (run_dir / "model_10.pt").touch()
        (run_dir / "model_9.pt").touch()
        (run_dir / "notes.txt").touch()
        write_run_manifest(str(run_dir), library="rsl_rl", task=task)
    _set_created_at(old_run, "2026-01-01T00:00:00+00:00")
    _set_created_at(new_run, "2026-01-02T00:00:00+00:00")

    found = discover_checkpoints(
        str(tmp_path), library="rsl_rl", task="Isaac-Cartpole", checkpoint_pattern=r"model_.*\.pt"
    )

    assert [(c.run, c.name) for c in found] == [
        ("new", "model_9.pt"),
        ("new", "model_10.pt"),
        ("old", "model_9.pt"),
        ("old", "model_10.pt"),
    ]
    assert all(Path(c.path).is_absolute() for c in found)


_PLAY = ["python", "isaaclab", "play", "--task", "Isaac-Cartpole-v0", "--num_envs", "4"]


def test_relaunch_argv_adds_policy_ghosts_once() -> None:
    enabled = relaunch_argv(_PLAY, enable=True)

    assert enabled == [*_PLAY, "--policy_ghosts"]
    assert relaunch_argv(enabled, enable=True) == enabled


def test_relaunch_argv_removes_policy_ghosts_with_their_options() -> None:
    argv = [*_PLAY, "--policy_ghosts", "--policy_ghost_slots", "8", "--policy_ghost_root=/runs", "--seed", "1"]

    assert relaunch_argv(argv, enable=False) == [*_PLAY, "--seed", "1"]


def test_launch_argv_reuses_the_original_command_line(monkeypatch) -> None:
    monkeypatch.setattr(sys, "orig_argv", ["python", "/venv/bin/isaaclab", "play", "--task", "T"])
    monkeypatch.setattr(sys, "argv", ["/venv/bin/isaaclab", "play", "--task", "T"])

    assert _launch_argv() == ["python", "/venv/bin/isaaclab", "play", "--task", "T"]


def test_launch_argv_runs_the_cli_with_arguments_a_launcher_script_set(monkeypatch) -> None:
    monkeypatch.setattr(sys, "orig_argv", ["python", "/tmp/launch.py"])
    monkeypatch.setattr(sys, "argv", ["isaaclab", "play", "--task", "T"])

    argv = _launch_argv()

    assert argv[1:3] == ["-c", "from isaaclab.cli import cli; cli()"] and argv[3:] == ["play", "--task", "T"]
    assert relaunch_argv(argv, enable=True)[-1] == "--policy_ghosts"
