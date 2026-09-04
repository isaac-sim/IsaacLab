# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pretrained-checkpoint training utility."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab_tasks.utils.preset_target import PresetTarget

from scripts.tools.train_and_publish_checkpoints import (
    CheckpointJob,
    _build_core_jobs,
    _play_command,
    _select_physics_variants,
    _training_command,
    collect_pretrained_checkpoint,
    publish_pretrained_checkpoint,
)


def test_build_core_jobs_skips_unsupported_preset_without_normalizing_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported preset-only task must not abort construction of the supported core matrix."""
    task_spec = SimpleNamespace(
        id="Isaac-Unsupported-Core-Task",
        kwargs={
            "env_cfg_entry_point": "isaaclab_tasks.core.unsupported:UnsupportedEnvCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.unsupported:UnsupportedAgentCfg",
        },
    )
    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.gym.registry", {task_spec.id: task_spec})
    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.parse_env_cfg", lambda _: object())
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.enumerate_task_presets",
        lambda _: {PresetTarget.PHYSICS: ["newton_kamino"]},
    )
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.CheckpointBundle.backend_names",
        lambda _: pytest.fail("preset-only tasks must not normalize their unsupported default backend"),
    )
    args = Namespace(physics_backends="physx,newtonmjwarp", render_backends="rtx,newton")

    assert _build_core_jobs(args) == []


def test_job_commands_use_uv_run_isaaclab() -> None:
    """Training and playback must use the uv-managed Isaac Lab CLI."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="physx",
        render_backend="none",
        physics_selector="isaacsim_physx",
    )
    args = Namespace(max_iterations=None, num_envs=None)

    train_command = _training_command(job, args, smoke=False)
    play_command = _play_command(job, args, "/tmp/checkpoint.pt")

    assert train_command[:4] == ["uv", "run", "isaaclab", "train"]
    assert play_command[:4] == ["uv", "run", "isaaclab", "play"]
    assert train_command[-1] == "physics=isaacsim_physx"
    assert play_command[-1] == "physics=isaacsim_physx"


def test_select_physics_variants_uses_concrete_isaac_sim_physx() -> None:
    """The normalized PhysX job must not resolve through the automatic selector."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp"]

    selections = _select_physics_variants("Isaac-Test", variants, "physx", ["physx", "newtonmjwarp"])

    assert selections == [("physx", "isaacsim_physx"), ("newtonmjwarp", "newton_mjwarp")]


def test_select_physics_variants_includes_franka_osc_newton_mjwarp() -> None:
    """The effort-limited OSC task is supported by Newton MJWarp."""
    selections = _select_physics_variants(
        "Isaac-Reach-Franka-OSC", ["isaacsim_physx", "newton_mjwarp"], "physx", ["newtonmjwarp"]
    )

    assert selections == [("newtonmjwarp", "newton_mjwarp")]


def test_select_physics_variants_selects_coupled_newton_preset() -> None:
    """Coupled tasks must publish under the MJWarp backend using their proxy preset."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp_vbd_proxy"]

    selections = _select_physics_variants("Isaac-Test", variants, "newtonmjwarp", ["newtonmjwarp"])

    assert selections == [("newtonmjwarp", "newton_mjwarp_vbd_proxy")]


def test_select_physics_variants_does_not_fall_back_to_automatic_physx() -> None:
    """A task without a concrete Isaac Sim selector must not run as OvPhysX."""
    selections = _select_physics_variants("Isaac-Test", ["physx", "ovphysx"], "physx", ["physx"])

    assert selections == []


def test_legacy_job_stem_preserves_task_name() -> None:
    """Legacy jobs must keep separate experiment directories for each task."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    assert job.stem == "Isaac-Test"


def test_legacy_collection_preserves_task_directory(tmp_path: Path) -> None:
    """Legacy collected checkpoints must retain their task-specific directory."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    path = collect_pretrained_checkpoint(job, str(tmp_path), dry_run=True)

    assert path == str(tmp_path / "rsl_rl" / "Isaac-Test" / "checkpoint.pt")


def test_core_job_needs_the_completion_marker_to_count_as_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run killed after writing a checkpoint must be retried, not skipped."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob(
        workflow="rsl_rl", task_name="Isaac-Test", physics_backend="newtonmjwarp", render_backend="none"
    )
    run_dir = Path(job.log_root) / "2026-01-01_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "model_0.pt").write_bytes(b"")

    assert job.has_finished
    assert not job.is_trained

    job.mark_trained()

    assert job.is_trained


def test_legacy_job_counts_as_trained_without_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy runs predate the marker, so requiring one would retrain every published checkpoint."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")
    run_dir = Path(job.log_root) / "2026-01-01_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "model_0.pt").write_bytes(b"")

    assert job.is_trained


def test_publish_uses_collected_checkpoint_without_training_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Publishing a collected checkpoint must not require its original training logs."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="newtonmjwarp",
        render_backend="none",
    )
    collected_path = tmp_path / "rsl_rl" / "Isaac-Test_newtonmjwarp_none_rsl_rl.pt"
    collected_path.parent.mkdir()
    collected_path.touch()
    args = Namespace(
        dry_run=True,
        force_publish=True,
        output_dir=str(tmp_path),
        publish_root="omniverse://checkpoints",
    )

    assert publish_pretrained_checkpoint(job, args)
    assert (
        f"Publishing {collected_path} -> omniverse://checkpoints/rsl_rl/Isaac-Test_newtonmjwarp_none_rsl_rl.pt"
        in capsys.readouterr().out
    )
