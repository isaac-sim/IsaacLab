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


@configclass
class _CameraCfg:
    """Minimal camera config for renderer-backend discovery."""

    renderer_cfg: IsaacRtxRendererCfg | NewtonWarpRendererCfg = IsaacRtxRendererCfg()


@configclass
class _ExtractorCfg:
    """Minimal component config that declares a checkpoint of its own."""

    checkpoint_name: str = "feature_extractor"
    checkpoint_glob: str = "cnn_*.pth"


@configclass
class _EnvCfg:
    """Minimal resolved environment config for backend discovery."""

    sim: SimulationCfg = SimulationCfg(physics=PhysxCfg())
    camera: _CameraCfg | None = None
    extractor: _ExtractorCfg | None = None
    observation_params: dict = {}
    """Stands in for a manager term reaching the component config a second time."""


def test_get_pretrained_checkpoint_filename_includes_backends():
    """Test that backend-aware filenames follow the published naming pattern."""
    filename = pretrained_checkpoint.get_pretrained_checkpoint_filename(
        "rsl_rl",
        "Isaac-Cartpole",
        "newtonmjwarp",
        "rtx",
    )

    assert filename == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl.pt"


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (("presets=depth",), ("depth",)),
        (("presets=rgb",), ()),
        (("physics=newton_mjwarp", "renderer=newton_renderer"), ()),
    ],
)
def test_get_pretrained_checkpoint_preset_names_uses_non_default_domain_presets(overrides, expected):
    """Test that default aliases and typed backends do not duplicate checkpoint identity fields."""
    preset_names = pretrained_checkpoint.get_pretrained_checkpoint_preset_names(
        "Isaac-Cartpole-Camera-Direct", overrides
    )

    assert preset_names == expected


def test_get_pretrained_checkpoint_filename_preserves_legacy_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Test that callers omitting both backends retain the legacy filename."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["play.py", "presets=depth"])
    cached_path = Path(".pretrained_checkpoints/rl_games/Isaac-Cartpole/checkpoint.pth")
    cached_path.parent.mkdir(parents=True)
    cached_path.touch()

    assert pretrained_checkpoint.get_pretrained_checkpoint_filename("rl_games", "Isaac-Cartpole") == "checkpoint.pth"
    assert pretrained_checkpoint.get_published_pretrained_checkpoint("rl_games", "Isaac-Cartpole") == str(cached_path)


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
    declared = Path(path).parent / f"{stem}_feature_extractor.pth"
    assert declared.is_file()


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
    "name,run_glob,expected",
    [
        ("feature_extractor", "cnn_*.pth", "/logs/Isaac-Cartpole_physx_none_rsl_rl_feature_extractor.pth"),
        ("encoder", "enc_*.safetensors", "/logs/Isaac-Cartpole_physx_none_rsl_rl_encoder.safetensors"),
    ],
)
def test_declared_checkpoint_path_keeps_the_extension(name, run_glob, expected):
    """Test that a published checkpoint keeps the extension of the file the run writes."""
    path = pretrained_checkpoint.get_declared_checkpoint_path(
        "/logs/Isaac-Cartpole_physx_none_rsl_rl.pt", "rsl_rl", name, run_glob
    )
    assert path == expected


def test_declared_checkpoints_come_from_the_declaring_component():
    """A component's declaration is found without the task listing it, and once per name."""
    assert pretrained_checkpoint.get_declared_checkpoints(_EnvCfg()) == {}

    extractor = _ExtractorCfg()
    env_cfg = _EnvCfg(extractor=extractor, observation_params={"e": extractor})

    assert pretrained_checkpoint.get_declared_checkpoints(env_cfg) == {"feature_extractor": "cnn_*.pth"}


def test_get_published_pretrained_checkpoint_skips_the_declared_checkpoint_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that a task without a feature extractor requests no declared checkpoint at all."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    retrieved = _install_fake_retrieve(monkeypatch, {remote_path})

    pretrained_checkpoint.get_published_pretrained_checkpoint("rsl_rl", "Isaac-Cartpole", "physx", "none")

    assert [r[0] for r in retrieved] == [remote_path]


def test_get_published_pretrained_checkpoint_names_the_selected_presets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    """Test that a non-default domain preset qualifies both the published name and the cache directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    monkeypatch.setattr("sys.argv", ["play.py", "presets=depth"])
    stem = "Isaac-Cartpole-Camera-Direct_depth_newtonmjwarp_newton_rl_games"
    remote_path = f"omniverse://IsaacLab/PretrainedCheckpoints/rl_games/{stem}.pth"
    retrieved = _install_fake_retrieve(monkeypatch, {remote_path})

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rl_games", "Isaac-Cartpole-Camera-Direct", "newtonmjwarp", "newton"
    )

    expected_download_dir = str(Path(".pretrained_checkpoints") / "rl_games" / stem)
    assert retrieved == [(remote_path, expected_download_dir)]
    assert path is not None and Path(path).name == f"{stem}.pth"


def test_get_published_pretrained_checkpoint_reports_unpublished_checkpoint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    """Test that a checkpoint missing from the asset server names the location that was tried."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    _install_fake_retrieve(monkeypatch, set())

    path = pretrained_checkpoint.get_published_pretrained_checkpoint("rsl_rl", "Isaac-Cartpole", "physx", "none")

    assert path is None
    output = capsys.readouterr().out
    assert "A pre-trained checkpoint is currently unavailable for this task." in output
    assert "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt" in output
    assert "'physx' physics and 'none' render backends" in output


def test_get_published_pretrained_checkpoint_raises_on_unwritable_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Test that a local download failure is reported instead of being reported as unavailable."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pretrained_checkpoint, "ISAACLAB_NUCLEUS_DIR", "omniverse://IsaacLab")
    cause = PermissionError(13, "Permission denied", str(tmp_path / ".pretrained_checkpoints"))

    def _retrieve_file_path(remote_path: str, download_dir: str) -> str:
        raise cause

    monkeypatch.setattr(pretrained_checkpoint, "retrieve_file_path", _retrieve_file_path)

    with pytest.raises(RuntimeError) as error:
        pretrained_checkpoint.get_published_pretrained_checkpoint("rsl_rl", "Isaac-Cartpole", "physx", "none")

    message = str(error.value)
    assert "Isaac-Cartpole_physx_none_rsl_rl.pt" in message
    assert "Permission denied" in message
    assert error.value.__cause__ is cause
