# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_rl.entrypoints.common import RUN_MANIFEST_FILENAME, resolve_checkpoint_selector, write_run_manifest


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


def test_latest_prefers_periodic_checkpoint_over_best_name(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    write_run_manifest(str(run_dir), library="skrl", task="Isaac-Cartpole")
    # skrl writes the best checkpoint as "best_agent.pt", which sorts after the periodic "agent_<timestep>.pt"
    best = run_dir / "checkpoints" / "best_agent.pt"
    best.touch()

    def resolve(selector: str) -> str:
        return resolve_checkpoint_selector(
            str(tmp_path),
            selector,
            library="skrl",
            task="Isaac-Cartpole",
            checkpoint_pattern=r".*",
            other_dirs=["checkpoints"],
            latest_checkpoint_pattern=r"agent_.*",
        )

    assert resolve("latest") == str(best.resolve())
    (run_dir / "checkpoints" / "agent_1000.pt").touch()
    newest = run_dir / "checkpoints" / "agent_2000.pt"
    newest.touch()
    assert resolve("latest") == str(newest.resolve())
    assert resolve("best") == str(best.resolve())


@pytest.mark.parametrize("config_name", ["openarm_reach", "ant"])
def test_latest_includes_rl_games_score_to_win_checkpoint(tmp_path: Path, config_name: str) -> None:
    """rl_games saves the checkpoint that reaches ``score_to_win`` without the ``last_`` prefix of periodic saves.

    ``ant`` sorts before ``last_`` and ``openarm_reach`` after it, so neither name order nor a ``last_`` filter
    finds the newest epoch for both.
    """
    run_dir = tmp_path / "run"
    (run_dir / "nn").mkdir(parents=True)
    write_run_manifest(str(run_dir), library="rl_games", task="Isaac-Task")
    for name in (f"last_{config_name}_ep_200_rew_10.pth", f"{config_name}.pth"):
        (run_dir / "nn" / name).touch()
    expected = run_dir / "nn" / f"{config_name}_ep_250_rew_10001.pth"
    expected.touch()

    checkpoint = resolve_checkpoint_selector(
        str(tmp_path),
        "latest",
        library="rl_games",
        task="Isaac-Task",
        checkpoint_pattern=r".*\.pth",
        other_dirs=["nn"],
        preferred_checkpoint_pattern=rf"{re.escape(config_name)}\.pth",
        latest_checkpoint_pattern=rf"(?:last_)?({re.escape(config_name)}_(?:ep|frame)_.*)\.pth",
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


def _rsl_rl_resume_path(tmp_path: Path, *, distillation: bool, checkpoint: str | None = None, resume: bool = False):
    from isaaclab_rl.entrypoints.backends.cli_args_rsl_rl import resolve_rsl_rl_training_checkpoint
    from isaaclab_rl.rsl_rl import RslRlDistillationRunnerCfg, RslRlOnPolicyRunnerCfg

    args_cli = SimpleNamespace(checkpoint=checkpoint, task="Isaac-Velocity-Rough-G1-Depth-Student", agent=None)
    agent_cfg = RslRlDistillationRunnerCfg() if distillation else RslRlOnPolicyRunnerCfg()
    agent_cfg.resume = resume
    agent_cfg.load_run = ".*"
    agent_cfg.load_checkpoint = r"model_.*\.pt"
    return resolve_rsl_rl_training_checkpoint(args_cli, agent_cfg, str(tmp_path / "logs" / "student"))


def test_rsl_rl_distillation_training_requires_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """Existing student runs do not supply a teacher unless resume is requested."""
    existing = tmp_path / "logs" / "student" / "existing"
    existing.mkdir(parents=True)
    (existing / "model_10.pt").touch()
    with pytest.raises(ValueError, match="requires --checkpoint"):
        _rsl_rl_resume_path(tmp_path, distillation=True)


def test_rsl_rl_distillation_training_loads_explicit_teacher_checkpoint(tmp_path: Path) -> None:
    """An explicit teacher file is used without searching the student experiment's runs."""
    teacher = tmp_path / "teacher" / "model_1499.pt"
    teacher.parent.mkdir()
    teacher.touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(teacher)) == str(teacher)
    with pytest.raises(FileNotFoundError):
        _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(teacher.with_name("x.pt")))


def test_rsl_rl_training_selects_latest_file_in_exact_directory(tmp_path: Path, monkeypatch) -> None:
    """A teacher run directory resolves to its newest ``model_*.pt``, not to a pattern built from the directory path."""
    monkeypatch.chdir(tmp_path)
    run = tmp_path / "teacher[1]"
    run.mkdir(parents=True)
    for name in ("model_5.pt", "model_10.pt", "events.out.tfevents"):
        (run / name).touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(run)) == str(run / "model_10.pt")

    sibling = run.with_name(run.name + "-retry")
    sibling.mkdir()
    (sibling / "model_99.pt").touch()
    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(run)) == str(run / "model_10.pt")
    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=str(run) + "/") == str(run / "model_10.pt")
    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=run.name) == str(run / "model_10.pt")
    (run / "model_100.pt").mkdir()
    assert _rsl_rl_resume_path(tmp_path, distillation=True, checkpoint=run.name) == str(run / "model_10.pt")


def test_rsl_rl_training_resume_searches_experiment_runs(tmp_path: Path) -> None:
    """A configured resume without ``--checkpoint`` continues the newest run, not an older run with later iterations."""
    older = tmp_path / "logs" / "student" / "2026-11-16_10-00-00" / "model_99.pt"
    older.parent.mkdir(parents=True)
    older.touch()
    checkpoint = tmp_path / "logs" / "student" / "2026-11-17_10-00-00" / "model_10.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()

    assert _rsl_rl_resume_path(tmp_path, distillation=False, resume=True) == str(checkpoint)
    assert _rsl_rl_resume_path(tmp_path, distillation=True, resume=True) == str(checkpoint)
    assert _rsl_rl_resume_path(tmp_path, distillation=False) is None
    teacher = tmp_path / "teacher.pt"
    teacher.touch()
    assert _rsl_rl_resume_path(tmp_path, distillation=True, resume=True, checkpoint=str(teacher)) == str(teacher)
