# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the pretrained-checkpoint training utility."""

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import pytest

from isaaclab_tasks.utils.hydra import collect_presets
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_tasks.utils.preset_cli import enumerate_task_presets
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

_FE = ("feature_extractor", "cnn_*.pth")


def test_cartpole_feature_presets_are_in_pretrained_checkpoint_matrix() -> None:
    """Every Cartpole feature policy for the preferred workflow must receive a distinct checkpoint."""
    task_spec = gym.spec("Isaac-Cartpole-Camera")
    workflow = task_spec.kwargs["default_agent"]
    agent_cfg = load_cfg_from_registry(task_spec.id, f"{workflow}_cfg_entry_point")
    feature_presets = set(collect_presets(agent_cfg)[""]) - {"default"}

    assert set(task_spec.kwargs["pretrained_checkpoint_preset_compatibility"][workflow]) == feature_presets


def test_checkpoint_preset_metadata_references_registered_variants() -> None:
    """Checkpoint declarations must name registered workflows and domain presets."""
    for task_spec in gym.registry.values():
        checkpoint_compatibility = task_spec.kwargs.get("pretrained_checkpoint_preset_compatibility", {})
        if not checkpoint_compatibility:
            continue

        preset_map = enumerate_task_presets(task_spec.id) or {}
        domain_presets = set(preset_map.get(PresetTarget.DOMAIN, ()))
        for workflow, preset_names in checkpoint_compatibility.items():
            assert f"{workflow}_cfg_entry_point" in task_spec.kwargs, (
                f"{task_spec.id}: unregistered {workflow} workflow"
            )
            assert len(preset_names) == len(set(preset_names)), (
                f"{task_spec.id}: duplicate {workflow} checkpoint preset"
            )
            assert not set(preset_names) - domain_presets, f"{task_spec.id}: unknown {workflow} checkpoint preset"


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
        "scripts.tools.train_and_publish_checkpoints.get_pretrained_checkpoint_backend_names",
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
        preset_names=("depth",),
        physics_selector="isaacsim_physx",
    )
    args = Namespace(max_iterations=None, num_envs=None)

    train_command = _training_command(job, args, smoke=False)
    play_command = _play_command(job, args, "/tmp/checkpoint.pt")

    assert train_command[:4] == ["uv", "run", "isaaclab", "train"]
    assert play_command[:4] == ["uv", "run", "isaaclab", "play"]
    assert train_command[-2:] == ["physics=isaacsim_physx", "presets=depth"]
    assert play_command[-2:] == ["physics=isaacsim_physx", "presets=depth"]


def test_build_core_jobs_includes_declared_checkpoint_presets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core jobs must include preset-specific checkpoints declared by the task."""
    task_spec = SimpleNamespace(
        id="Isaac-Test",
        kwargs={
            "env_cfg_entry_point": "isaaclab_tasks.core.test:TestEnvCfg",
            "rl_games_cfg_entry_point": "isaaclab_tasks.core.test:TestAgentCfg",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.core.test:TestAgentCfg",
            "pretrained_checkpoint_preset_compatibility": {"rl_games": ("depth",)},
        },
    )
    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.gym.registry", {task_spec.id: task_spec})
    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.parse_env_cfg", lambda _: object())
    monkeypatch.setattr("scripts.tools.train_and_publish_checkpoints.enumerate_task_presets", lambda _: {})
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.get_pretrained_checkpoint_backend_names",
        lambda _: ("physx", "rtx"),
    )
    args = Namespace(physics_backends="physx", render_backends="rtx")

    jobs = _build_core_jobs(args)

    assert [(job.workflow, job.preset_names) for job in jobs] == [("rsl_rl", ()), ("rl_games", ("depth",))]


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


def test_legacy_job_experiment_name_preserves_task_name() -> None:
    """Legacy jobs must keep separate experiment directories for each task."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    assert job.experiment_name == "Isaac-Test"


def test_legacy_collection_preserves_task_directory(tmp_path: Path) -> None:
    """Legacy collected checkpoints must retain their task-specific directory."""
    job = CheckpointJob(workflow="rsl_rl", task_name="Isaac-Test")

    path = collect_pretrained_checkpoint(job, str(tmp_path), dry_run=True)

    assert path == str(tmp_path / "rsl_rl" / "Isaac-Test" / "checkpoint.pt")


def test_recollecting_without_a_run_file_drops_the_previous_declared_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A re-collect that finds no declared file must not leave the last one beside the new policy."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="newtonmjwarp",
        render_backend="none",
        declared_checkpoints=(_FE,),
    )
    run_path = tmp_path / "run"
    run_path.mkdir()
    policy = run_path / "model.pt"
    policy.touch()
    (run_path / "cnn_100_0.1.pth").touch()
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.get_pretrained_checkpoint_path",
        lambda *a, **k: str(policy),
    )
    monkeypatch.setattr(
        "scripts.tools.train_and_publish_checkpoints.get_latest_job_run_path",
        lambda *a, **k: str(run_path),
    )
    output_dir = str(tmp_path / "out")

    destination = collect_pretrained_checkpoint(job, output_dir)
    collected_cnn = Path(destination).with_name("Isaac-Test_newtonmjwarp_none_rsl_rl_feature_extractor.pth")
    assert collected_cnn.is_file()

    # the next run trained a policy but wrote no CNN
    (run_path / "cnn_100_0.1.pth").unlink()
    collect_pretrained_checkpoint(job, output_dir)

    assert not collected_cnn.exists()


def test_publish_refuses_a_bundle_whose_declared_checkpoint_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A component needs its declared checkpoint to play, so publishing the policy alone must fail."""
    job = CheckpointJob(
        workflow="rsl_rl",
        task_name="Isaac-Test",
        physics_backend="newtonmjwarp",
        render_backend="none",
        declared_checkpoints=(_FE,),
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

    assert not publish_pretrained_checkpoint(job, args)
    assert "its feature_extractor checkpoint was not collected" in capsys.readouterr().err


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
        preset_names=("depth",),
    )
    collected_path = tmp_path / "rsl_rl" / "Isaac-Test_depth_newtonmjwarp_none_rsl_rl.pt"
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
        f"Publishing {collected_path} -> omniverse://checkpoints/rsl_rl/Isaac-Test_depth_newtonmjwarp_none_rsl_rl.pt"
        in capsys.readouterr().out
    )
