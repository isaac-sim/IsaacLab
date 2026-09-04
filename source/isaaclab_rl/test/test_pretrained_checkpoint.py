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
from isaaclab.utils import Checkpoint
from isaaclab.utils.configclass import configclass

from isaaclab_rl.utils import pretrained_checkpoint


@configclass
class _CameraCfg:
    """Minimal camera config for renderer-backend discovery."""

    renderer_cfg: IsaacRtxRendererCfg | NewtonWarpRendererCfg = IsaacRtxRendererCfg()


@configclass
class _ExtractorCfg:
    """Minimal component config that declares a checkpoint of its own."""

    checkpoint: Checkpoint = Checkpoint(name="feature_extractor", run_glob="cnn_*.pth")
    frozen: Checkpoint = Checkpoint(name="vae", url="omniverse://IsaacLab/Contrib/vae.pt")


@configclass
class _EnvCfg:
    """Minimal resolved environment config for backend discovery."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    camera: _CameraCfg | None = None
    extractor: _ExtractorCfg | None = None


def test_get_pretrained_checkpoint_filename_includes_backends():
    """Test that backend-aware filenames follow the published naming pattern."""
    filename = pretrained_checkpoint.get_pretrained_checkpoint_filename(
        "rsl_rl",
        "Isaac-Cartpole",
        "newtonmjwarp",
        "rtx",
    )

    assert filename == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl.pt"


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


def _install_fake_retrieve(
    monkeypatch: pytest.MonkeyPatch,
    published_files: set[str],
) -> list[tuple[str, str]]:
    """Stub the Nucleus download with a local copy limited to ``published_files``.

    The stub mirrors the published tree under the download directory, as the real download does.
    """
    retrieved: list[tuple[str, str]] = []

    def _retrieve_file_path(remote_path: str, download_dir: str) -> str:
        retrieved.append((remote_path, download_dir))
        if remote_path not in published_files:
            raise FileNotFoundError(remote_path)
        destination = Path(download_dir) / Path(remote_path).parent.name / Path(remote_path).name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.touch()
        return str(destination.resolve())

    monkeypatch.setattr(pretrained_checkpoint, "retrieve_file_path", _retrieve_file_path)
    return retrieved


def test_get_published_pretrained_checkpoint_downloads_to_checkpoint_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that backend-aware downloads use a cache directory of their own."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    retrieved = _install_fake_retrieve(monkeypatch, {remote_path})

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl",
        "Isaac-Cartpole",
        "physx",
        "none",
    )

    expected_download_dir = str(Path(".pretrained_checkpoints") / "rsl_rl" / "Isaac-Cartpole_physx_none_rsl_rl")
    assert retrieved[0] == (remote_path, expected_download_dir)
    assert Path(path).name == "Isaac-Cartpole_physx_none_rsl_rl.pt"
    assert Path(path).is_relative_to(tmp_path / expected_download_dir)


def test_get_published_pretrained_checkpoint_downloads_the_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that a published feature-extractor checkpoint lands beside its policy checkpoint."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    published_root = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl"
    stem = "Isaac-Reorient-Cube-Shadow-Camera_physx_rtx_rsl_rl"
    _install_fake_retrieve(
        monkeypatch, {f"{published_root}/{stem}.pt", f"{published_root}/{stem}_feature_extractor.pth"}
    )

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl",
        "Isaac-Reorient-Cube-Shadow-Camera",
        "physx",
        "rtx",
        env_cfg=_EnvCfg(extractor=_ExtractorCfg()),
    )

    assert path is not None
    assert Path(path).parent.joinpath(f"{stem}_feature_extractor.pth").is_file()


def test_get_published_pretrained_checkpoint_tolerates_no_feature_extractor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that tasks without a published feature extractor still resolve their checkpoint."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    _install_fake_retrieve(monkeypatch, {remote_path})

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl",
        "Isaac-Cartpole",
        "physx",
        "none",
    )

    assert path is not None
    assert sorted(p.name for p in Path(path).parent.iterdir()) == ["Isaac-Cartpole_physx_none_rsl_rl.pt"]


@pytest.mark.parametrize(
    "checkpoint,expected",
    [
        (
            Checkpoint(name="feature_extractor", run_glob="cnn_*.pth"),
            "/logs/Isaac-Cartpole_physx_none_rsl_rl_feature_extractor.pth",
        ),
        (
            Checkpoint(name="encoder", run_glob="enc_*.safetensors"),
            "/logs/Isaac-Cartpole_physx_none_rsl_rl_encoder.safetensors",
        ),
    ],
)
def test_get_declared_checkpoint_path_keeps_the_declared_extension(checkpoint, expected):
    """Test that a published checkpoint keeps the extension the component declared."""
    path = pretrained_checkpoint.get_declared_checkpoint_path(
        "/logs/Isaac-Cartpole_physx_none_rsl_rl.pt", "rsl_rl", checkpoint
    )
    assert path == expected


def test_get_declared_checkpoints_discovers_nested_component_configs():
    """Test that a component config declaring a checkpoint is found without the task listing it."""
    assert pretrained_checkpoint.get_declared_checkpoints(_EnvCfg()) == []

    found = pretrained_checkpoint.get_declared_checkpoints(_EnvCfg(extractor=_ExtractorCfg()))

    # only run artifacts are published beside the policy; URL weights are the component's to fetch
    assert [(c.name, c.run_glob) for c in found] == [("feature_extractor", "cnn_*.pth")]


def test_get_published_pretrained_checkpoint_skips_the_companion_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that a task without a feature extractor makes no companion request at all."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    retrieved = _install_fake_retrieve(monkeypatch, {remote_path})

    pretrained_checkpoint.get_published_pretrained_checkpoint("rsl_rl", "Isaac-Cartpole", "physx", "none")

    assert [r[0] for r in retrieved] == [remote_path]
