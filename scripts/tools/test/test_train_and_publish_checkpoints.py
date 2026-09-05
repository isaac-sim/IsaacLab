# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pretrained-checkpoint training utility."""

import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaaclab.utils import Checkpoint

from isaaclab_rl.entrypoints.common import write_run_manifest

from isaaclab_tasks.utils.preset_target import PresetTarget

from scripts.tools.train_and_publish_checkpoints import (
    CheckpointJob,
    _build_core_jobs,
    _select_physics_variants,
    collect_pretrained_checkpoint,
    publish_pretrained_checkpoint,
)

_FE = Checkpoint(name="feature_extractor", run_glob="cnn_*.pth")


def _write_run(job: CheckpointJob, name: str, files: list[str], manifest: bool = True) -> Path:
    """Create a training run directory with the given files, as the unified train entrypoint would."""
    run = Path(job.log_root) / name
    run.mkdir(parents=True, exist_ok=True)
    for rel in files:
        (run / rel).parent.mkdir(parents=True, exist_ok=True)
        (run / rel).touch()
    if manifest:
        write_run_manifest(str(run), library=job.workflow, task=job.task_name)
    return run


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
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.resolve_task_config", lambda *_, **__: (object(), None)
    )
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


def test_from_task_reads_backends_and_companions_from_the_preset_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The job describes the config its selectors produce, not the task's default config."""
    seen = []

    def _resolve(task_name, agent_entry_point, overrides=None):
        seen.append((task_name, agent_entry_point, overrides))
        return SimpleNamespace(), None

    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.resolve_task_config", _resolve)
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.CheckpointBundle.backend_names",
        lambda _: ("newtonmjwarp", "newton"),
    )
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.CheckpointBundle.declared_companions", lambda _: (_FE,)
    )

    job = CheckpointJob.from_task("Isaac-Test", "rsl_rl", "newton_mjwarp", "newton_renderer")

    assert seen == [("Isaac-Test", None, ("physics=newton_mjwarp", "renderer=newton_renderer"))]
    assert (job.physics_backend, job.render_backend, job.companions) == ("newtonmjwarp", "newton", (_FE,))
    assert job.job_id == "rsl_rl:Isaac-Test:newtonmjwarp:newton"
    assert job.preset_args == ["physics=newton_mjwarp", "renderer=newton_renderer"]


def test_job_commands_use_uv_run_isaaclab() -> None:
    """Training and playback must use the uv-managed Isaac Lab CLI and name the experiment after the stem."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="physx",
        render_backend="none",
        physics_selector="isaacsim_physx",
    )
    args = Namespace(max_iterations=None, num_envs=None)

    train_command = job.train_command(args)
    play_command = job.play_command(args, "/tmp/checkpoint.pt")

    assert train_command[:4] == ["uv", "run", "isaaclab", "train"]
    assert "agent.experiment_name=Isaac-Test_physx_none_rsl_rl" in train_command
    assert play_command[:4] == ["uv", "run", "isaaclab", "play"]
    assert train_command[-1] == "physics=isaacsim_physx"
    assert play_command[-1] == "physics=isaacsim_physx"


def test_select_physics_variants_uses_concrete_isaac_sim_physx() -> None:
    """The normalized PhysX job must not resolve through the automatic selector."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp"]

    selections = _select_physics_variants(variants, "physx", ["physx", "newtonmjwarp"])

    assert selections == [("physx", "isaacsim_physx"), ("newtonmjwarp", "newton_mjwarp")]


def test_select_physics_variants_includes_franka_osc_newton_mjwarp() -> None:
    """The effort-limited OSC task is supported by Newton MJWarp."""
    selections = _select_physics_variants(["isaacsim_physx", "newton_mjwarp"], "physx", ["newtonmjwarp"])

    assert selections == [("newtonmjwarp", "newton_mjwarp")]


def test_select_physics_variants_selects_coupled_newton_preset() -> None:
    """Coupled tasks must publish under the MJWarp backend using their proxy preset."""
    variants = ["physx", "isaacsim_physx", "ovphysx", "newton_mjwarp_vbd_proxy"]

    selections = _select_physics_variants(variants, "newtonmjwarp", ["newtonmjwarp"])

    assert selections == [("newtonmjwarp", "newton_mjwarp_vbd_proxy")]


def test_select_physics_variants_does_not_fall_back_to_automatic_physx() -> None:
    """A task without a concrete Isaac Sim selector must not run as OvPhysX."""
    selections = _select_physics_variants(["physx", "ovphysx"], "physx", ["physx"])

    assert selections == []


def test_legacy_job_stem_preserves_task_name() -> None:
    """Legacy jobs must keep separate experiment directories for each task."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    assert job.stem == "Isaac-Test"
    assert job.job_id == "rsl_rl:Isaac-Test"


def test_legacy_collection_preserves_task_directory(tmp_path: Path) -> None:
    """Legacy collected checkpoints must retain their task-specific directory."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    path = collect_pretrained_checkpoint(job, str(tmp_path), dry_run=True)

    assert path == str(tmp_path / "rsl_rl" / "Isaac-Test" / "checkpoint.pt")


def test_trained_path_selects_the_preferred_policy_then_the_last_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The publish script picks the policy like ``--checkpoint best``; companions come from the same run."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob("skrl", "Isaac-Test", "physx", "none", (_FE,))
    assert job.latest_run is None and not job.has_run and not job.has_finished

    run = _write_run(job, "2026-09-02_10-00-00", ["checkpoints/agent_100.pt", "cnn_1_0.5.pth", "cnn_2_0.1.pth"])
    os.utime(run / "cnn_1_0.5.pth", (1_000_000, 1_000_000))
    assert job.trained_path() == str(run / "checkpoints" / "agent_100.pt")
    assert job.latest_run == str(run)
    assert job.trained_path(_FE) == str(run / "cnn_2_0.1.pth")

    (run / "checkpoints" / "best_agent.pt").touch()
    assert job.trained_path() == str(run / "checkpoints" / "best_agent.pt")
    assert job.has_run and job.has_finished


def test_runs_without_a_manifest_are_not_collected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only runs of the unified train entrypoint, which writes the manifest, are candidates."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob("rsl_rl", "Isaac-Test", "physx", "none")
    _write_run(job, "2026-01-01_00-00-00", ["model_0.pt"], manifest=False)

    assert job.has_run and not job.has_finished


def test_core_job_needs_the_completion_marker_to_count_as_trained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run killed after writing a checkpoint must be retried, not skipped."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob("rsl_rl", "Isaac-Test", "newtonmjwarp", "none")
    _write_run(job, "2026-01-01_00-00-00", ["model_0.pt"])

    assert job.has_finished
    assert not job.is_trained

    job.mark_trained()

    assert job.is_trained


def test_legacy_job_counts_as_trained_without_the_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy runs predate the marker, so requiring one would retrain every published checkpoint."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")
    _write_run(job, "2026-01-01_00-00-00", ["model_0.pt"])

    assert job.is_trained


def test_review_is_read_from_the_selected_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review lives beside the selected policy and is absent until written."""
    monkeypatch.chdir(tmp_path)
    job = CheckpointJob("rsl_rl", "Isaac-Test", "physx", "none")
    assert job.review_path is None and job.review is None

    run = _write_run(job, "run", ["model_10.pt"])
    assert job.review_path == str(run / "pretrained_checkpoint_review.json") and job.review is None
    Path(job.review_path).write_text('{"reviewed": true, "result": "accepted"}')
    assert job.review == {"reviewed": True, "result": "accepted"}


def test_publish_refuses_a_bundle_whose_declared_checkpoint_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A component needs its declared checkpoint to play, so publishing the policy alone must fail."""
    job = CheckpointJob("rsl_rl", "Isaac-Test", "newtonmjwarp", "none", (_FE,))
    collected_path = tmp_path / "rsl_rl" / "Isaac-Test_newtonmjwarp_none_rsl_rl.pt"
    collected_path.parent.mkdir()
    collected_path.touch()
    args = Namespace(
        dry_run=True,
        force_publish=True,
        output_dir=str(tmp_path),
        publish_root="omniverse://checkpoints",
    )

    assert not publish_pretrained_checkpoint(job, args)
    assert "its feature_extractor checkpoint was not collected" in capsys.readouterr().err


def test_publish_uses_collected_checkpoint_without_training_logs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Publishing a collected checkpoint must not require its original training logs."""
    job = CheckpointJob("rsl_rl", "Isaac-Test", "newtonmjwarp", "none")
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
