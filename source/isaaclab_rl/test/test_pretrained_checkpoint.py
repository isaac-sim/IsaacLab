# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for backend-aware pretrained checkpoint paths."""

from pathlib import Path

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg

from isaaclab.renderers import RendererCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_rl.utils import pretrained_checkpoint


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
        "newton",
        "rtx",
    )

    assert filename == "Isaac-Cartpole_newton_rtx.pt"


def test_get_pretrained_checkpoint_filename_preserves_legacy_layout():
    """Test that callers omitting both backends retain the legacy filename."""
    assert pretrained_checkpoint.get_pretrained_checkpoint_filename("rl_games", "Isaac-Cartpole") == "checkpoint.pth"


def test_get_pretrained_checkpoint_filename_requires_both_backends():
    """Test that partial backend identifiers are rejected."""
    with pytest.raises(ValueError, match="must be provided together"):
        pretrained_checkpoint.get_pretrained_checkpoint_filename(
            "rsl_rl",
            "Isaac-Cartpole",
            physics_backend="newton",
        )


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

    assert pretrained_checkpoint.get_pretrained_checkpoint_backend_names(env_cfg) == ("newton", "newton")


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
        "newton",
        "none",
    )

    assert path == "omniverse://IsaacLab/PretrainedCheckpoints/skrl/Isaac-Shadow-Handover-Direct_newton_none.pt"


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

    assert path == "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none.pt"


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

    assert path == ".pretrained_checkpoints/rsl_rl/Isaac-Cartpole_physx_none.pt"
    assert retrieved == {
        "remote_path": "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none.pt",
        "download_dir": ".pretrained_checkpoints/rsl_rl",
    }
