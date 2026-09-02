# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend-aware pretrained checkpoint paths."""

from pathlib import Path

import pytest
from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers import RendererCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_rl.utils import pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config


@configclass
class _CameraCfg:
    """Minimal camera config for renderer-backend discovery."""

    renderer_cfg: IsaacRtxRendererCfg | NewtonWarpRendererCfg = IsaacRtxRendererCfg()


@configclass
class _EnvCfg:
    """Minimal resolved environment config for backend discovery."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    camera: _CameraCfg | None = None


def test_get_pretrained_checkpoint_filename_includes_backends():
    """Test that backend-aware filenames follow the published naming pattern."""
    filename = pretrained_checkpoint.get_pretrained_checkpoint_filename(
        "rsl_rl",
        "Isaac-Cartpole",
        "newtonmjwarp",
        "rtx",
    )

    assert filename == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl.pt"


def test_get_pretrained_checkpoint_filename_includes_policy_variant():
    """Test that policy-contract variants receive distinct published filenames."""
    checkpoint = pretrained_checkpoint.PretrainedCheckpointCfg(workflow="rl_games", variant="depth")
    filename = pretrained_checkpoint.get_pretrained_checkpoint_filename(
        "rl_games",
        checkpoint.artifact_task_name("Isaac-Cartpole-Camera-Direct"),
        "newtonmjwarp",
        "newton",
    )

    assert filename == "Isaac-Cartpole-Camera-Direct_depth_newtonmjwarp_newton_rl_games.pth"


@pytest.mark.parametrize("task_name", ["Isaac-Cartpole-Camera", "Isaac-Cartpole-Camera-Direct"])
def test_select_pretrained_checkpoint_matches_cartpole_policy_preset(task_name: str):
    """Test that Cartpole policy presets select only compatible checkpoint declarations."""
    rgb_cfg, _ = resolve_task_config(task_name, "rl_games_cfg_entry_point", overrides=("presets=rgb",))
    depth_cfg, _ = resolve_task_config(task_name, "rl_games_cfg_entry_point", overrides=("presets=depth",))
    albedo_cfg, _ = resolve_task_config(task_name, "rl_games_cfg_entry_point", overrides=("presets=albedo",))

    rgb_checkpoint = pretrained_checkpoint.select_pretrained_checkpoint("rl_games", task_name, rgb_cfg)
    depth_checkpoint = pretrained_checkpoint.select_pretrained_checkpoint("rl_games", task_name, depth_cfg)

    assert rgb_checkpoint is not None and rgb_checkpoint.variant is None
    assert depth_checkpoint is not None and depth_checkpoint.variant == "depth"
    assert pretrained_checkpoint.select_pretrained_checkpoint("rsl_rl", task_name, depth_cfg) is None
    assert pretrained_checkpoint.select_pretrained_checkpoint("rl_games", task_name, albedo_cfg) is None


def test_get_published_pretrained_checkpoint_for_env_uses_selected_artifact(monkeypatch: pytest.MonkeyPatch):
    """Depth playback must fetch the depth artifact instead of the default RGB policy."""
    task_name = "Isaac-Cartpole-Camera-Direct"
    env_cfg, _ = resolve_task_config(task_name, "rl_games_cfg_entry_point", overrides=("presets=depth",))
    expected_backends = pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg)
    requested = {}

    def _get_published(workflow: str, artifact_task_name: str, *backends: str) -> str:
        requested.update(workflow=workflow, task_name=artifact_task_name, backends=backends)
        return "/tmp/checkpoint.pth"

    monkeypatch.setattr(pretrained_checkpoint, "get_published_pretrained_checkpoint", _get_published)

    path = pretrained_checkpoint.get_published_pretrained_checkpoint_for_env("rl_games", task_name, env_cfg)

    assert path == "/tmp/checkpoint.pth"
    assert requested == {
        "workflow": "rl_games",
        "task_name": "Isaac-Cartpole-Camera-Direct_depth",
        "backends": expected_backends,
    }


def test_get_pretrained_checkpoint_filename_preserves_legacy_layout():
    """Test that callers omitting both backends retain the legacy filename."""
    assert pretrained_checkpoint.get_pretrained_checkpoint_filename("rl_games", "Isaac-Cartpole") == "checkpoint.pth"


def test_get_log_root_path_preserves_legacy_task_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that callers omitting both backends retain the task-specific log root."""
    monkeypatch.chdir(tmp_path)

    path = pretrained_checkpoint.get_log_root_path("rsl_rl", "Isaac-Cartpole")

    assert path == str(tmp_path / "logs" / "rsl_rl" / "Isaac-Cartpole")


def test_get_pretrained_checkpoint_filename_requires_both_backends():
    """Test that partial backend identifiers are rejected."""
    with pytest.raises(ValueError, match="must be provided together"):
        pretrained_checkpoint.get_pretrained_checkpoint_filename(
            "rsl_rl",
            "Isaac-Cartpole",
            physics_backend="newtonmjwarp",
        )


def test_get_pretrained_checkpoint_filename_rejects_invalid_variant():
    """Test that variants cannot escape the flat workflow directory."""
    with pytest.raises(ValueError, match="Invalid checkpoint variant"):
        pretrained_checkpoint.PretrainedCheckpointCfg(workflow="rsl_rl", variant="../depth")


def test_get_pretrained_checkpoint_backend_names_identifies_physx_without_renderer():
    """Test backend discovery for a state-only PhysX task."""
    env_cfg = _EnvCfg(camera=None)

    assert pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg) == ("physx", "none")


def test_get_pretrained_checkpoint_backend_names_identifies_newton_renderer():
    """Test backend discovery for a Newton task using the Newton renderer."""
    env_cfg = _EnvCfg(
        sim=SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
        camera=_CameraCfg(renderer_cfg=NewtonWarpRendererCfg()),
    )

    assert pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg) == ("newtonmjwarp", "newton")


def test_get_pretrained_checkpoint_backend_names_rejects_other_newton_solvers():
    """Test that a non-MJWarp Newton solver is not mislabeled as MJWarp."""
    env_cfg = _EnvCfg(sim=SimulationCfg(physics=NewtonCfg(solver_cfg=KaminoPADMMSolverCfg())))

    with pytest.raises(ValueError, match="Unsupported Newton solver"):
        pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg)


def test_get_pretrained_checkpoint_backend_names_identifies_rtx_renderer():
    """Test backend discovery for a PhysX task using RTX rendering."""
    env_cfg = _EnvCfg(camera=_CameraCfg(renderer_cfg=IsaacRtxRendererCfg()))

    assert pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg) == ("physx", "rtx")


def test_get_pretrained_checkpoint_backend_names_identifies_automatic_rtx_renderer():
    """Test backend discovery for the runtime-selected RTX renderer."""
    env_cfg = _EnvCfg(camera=_CameraCfg(renderer_cfg=RendererCfg(renderer_type="auto_rtx")))

    assert pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg) == ("physx", "rtx")


def test_get_published_pretrained_checkpoint_path_uses_flat_workflow_directory(monkeypatch: pytest.MonkeyPatch):
    """Test that backend-aware published checkpoints are flat within the workflow directory."""
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")

    path = pretrained_checkpoint.get_published_pretrained_checkpoint_path(
        "skrl",
        "Isaac-Shadow-Handover-Direct",
        "newtonmjwarp",
        "none",
    )

    assert path == (
        "omniverse://IsaacLab/PretrainedCheckpoints/skrl/Isaac-Shadow-Handover-Direct_newtonmjwarp_none_skrl.pt"
    )


def test_get_published_pretrained_checkpoint_path_includes_policy_variant(monkeypatch: pytest.MonkeyPatch):
    """Test that variant checkpoints use a distinct flat published path."""
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    checkpoint = pretrained_checkpoint.PretrainedCheckpointCfg(workflow="rl_games", variant="depth")

    path = pretrained_checkpoint.get_published_pretrained_checkpoint_path(
        "rl_games",
        checkpoint.artifact_task_name("Isaac-Cartpole-Camera-Direct"),
        "newtonmjwarp",
        "newton",
    )

    assert path == (
        "omniverse://IsaacLab/PretrainedCheckpoints/rl_games/"
        "Isaac-Cartpole-Camera-Direct_depth_newtonmjwarp_newton_rl_games.pth"
    )


def test_get_pretrained_checkpoint_publish_path_uses_flat_workflow_directory(monkeypatch: pytest.MonkeyPatch):
    """Test that backend-aware uploads target the same flat layout used for downloads."""
    monkeypatch.setattr(
        pretrained_checkpoint, "PRETRAINED_CHECKPOINT_PATH", "omniverse://IsaacLab/PretrainedCheckpoints"
    )

    path = pretrained_checkpoint.get_pretrained_checkpoint_publish_path(
        "rsl_rl",
        "Isaac-Cartpole",
        "physx",
        "none",
    )

    assert path == "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"


def test_get_published_pretrained_checkpoint_downloads_to_flat_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that backend-aware downloads use the workflow cache directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    retrieved = {}

    def _retrieve_file_path(remote_path: str, download_dir: str) -> str:
        retrieved["remote_path"] = remote_path
        retrieved["download_dir"] = download_dir
        destination = Path(download_dir) / Path(remote_path).name
        destination.parent.mkdir(parents=True)
        destination.touch()
        return str(destination)

    monkeypatch.setattr(pretrained_checkpoint, "retrieve_file_path", _retrieve_file_path)

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl",
        "Isaac-Cartpole",
        "physx",
        "none",
    )

    expected_download_dir = str(Path(".pretrained_checkpoints") / "rsl_rl")
    assert path == str(Path(expected_download_dir) / "Isaac-Cartpole_physx_none_rsl_rl.pt")
    assert retrieved == {
        "remote_path": "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt",
        "download_dir": expected_download_dir,
    }
