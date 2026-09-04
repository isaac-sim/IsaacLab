# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the published checkpoint bundle of a trained task variant."""

import re
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
from isaaclab_rl.utils.pretrained_checkpoint import WORKFLOWS, CheckpointBundle


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
    observation_params: dict = {}
    """Stands in for a manager term reaching the extractor config a second time through its params."""


_FE = Checkpoint(name="feature_extractor", run_glob="cnn_*.pth")


def test_bundle_filenames_follow_the_published_naming_pattern():
    """Backend-aware files are ``<task>_<physics>_<render>_<workflow><ext>``, companions add ``_<name>``."""
    bundle = CheckpointBundle("rsl_rl", "Isaac-Cartpole", "newtonmjwarp", "rtx", (_FE,))

    assert bundle.stem == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl"
    assert bundle.filename() == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl.pt"
    assert bundle.filename(_FE) == "Isaac-Cartpole_newtonmjwarp_rtx_rsl_rl_feature_extractor.pth"


def test_legacy_bundle_keeps_the_per_task_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Callers omitting both backends retain the legacy filename and task-specific paths."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        pretrained_checkpoint, "PRETRAINED_CHECKPOINT_PATH", "omniverse://IsaacLab/PretrainedCheckpoints"
    )
    bundle = CheckpointBundle("rl_games", "Isaac-Cartpole")

    assert bundle.filename() == "checkpoint.pth"
    assert bundle.cache_dir == str(Path(".pretrained_checkpoints") / "rl_games" / "Isaac-Cartpole")
    assert (
        bundle.published_path() == "omniverse://IsaacLab/PretrainedCheckpoints/rl_games/Isaac-Cartpole/checkpoint.pth"
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"workflow": "nope", "task_name": "T"}, "Unsupported workflow"),
        ({"workflow": "rsl_rl", "task_name": "T", "physics_backend": "newtonmjwarp"}, "must be provided together"),
        (
            {"workflow": "rsl_rl", "task_name": "T", "physics_backend": "", "render_backend": "none"},
            "physics backend",
        ),
    ],
)
def test_bundle_rejects_invalid_identity(kwargs, match):
    """An unsupported workflow or a partial or unknown backend pair is rejected on construction."""
    with pytest.raises(ValueError, match=match):
        CheckpointBundle(**kwargs)


@pytest.mark.parametrize(
    "env_cfg,expected",
    [
        (_EnvCfg(camera=None), ("physx", "none")),
        (_EnvCfg(camera=_CameraCfg(renderer_cfg=IsaacRtxRendererCfg())), ("physx", "rtx")),
        (_EnvCfg(camera=_CameraCfg(renderer_cfg=RendererCfg(renderer_type="auto_rtx"))), ("physx", "rtx")),
        (
            _EnvCfg(
                sim=SimulationCfg(physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())),
                camera=_CameraCfg(renderer_cfg=NewtonWarpRendererCfg()),
            ),
            ("newtonmjwarp", "newton"),
        ),
    ],
)
def test_backend_names_are_read_from_the_env_cfg(env_cfg, expected):
    """Backend discovery covers state-only PhysX, RTX (explicit and runtime-selected), and Newton."""
    assert CheckpointBundle.backend_names(env_cfg) == expected


def test_backend_names_reject_other_newton_solvers():
    """A non-MJWarp Newton solver is not mislabeled as MJWarp."""
    env_cfg = _EnvCfg(sim=SimulationCfg(physics=NewtonCfg(solver_cfg=KaminoPADMMSolverCfg())))

    with pytest.raises(ValueError, match="Unsupported Newton solver"):
        CheckpointBundle.backend_names(env_cfg)


def test_from_env_cfg_discovers_declared_run_artifacts_once():
    """A component's declaration is found without the task listing it, once, and URL weights are excluded."""
    assert CheckpointBundle.from_env_cfg("rsl_rl", "T", _EnvCfg()).companions == ()

    extractor = _ExtractorCfg()
    bundle = CheckpointBundle.from_env_cfg(
        "rsl_rl", "T", _EnvCfg(extractor=extractor, observation_params={"e": extractor})
    )

    assert (bundle.physics_backend, bundle.render_backend) == ("physx", "none")
    assert [(c.name, c.run_glob) for c in bundle.companions] == [("feature_extractor", "cnn_*.pth")]


def test_published_path_is_flat_within_the_workflow_directory(monkeypatch: pytest.MonkeyPatch):
    """Backend-aware bundles publish and fetch from the same flat layout, companions beside the policy."""
    monkeypatch.setattr(
        pretrained_checkpoint, "PRETRAINED_CHECKPOINT_PATH", "omniverse://IsaacLab/PretrainedCheckpoints"
    )
    bundle = CheckpointBundle("skrl", "Isaac-Shadow-Handover-Direct", "newtonmjwarp", "none", (_FE,))
    root = "omniverse://IsaacLab/PretrainedCheckpoints/skrl"

    assert bundle.published_path() == f"{root}/Isaac-Shadow-Handover-Direct_newtonmjwarp_none_skrl.pt"
    assert (
        bundle.published_path(_FE)
        == f"{root}/Isaac-Shadow-Handover-Direct_newtonmjwarp_none_skrl_feature_extractor.pth"
    )
    assert (
        bundle.published_path(root="s3://mirror/")
        == "s3://mirror/skrl/Isaac-Shadow-Handover-Direct_newtonmjwarp_none_skrl.pt"
    )


def test_collected_path_mirrors_the_published_layout(tmp_path: Path):
    """The collect step writes the same relative tree the publish step reads."""
    bundle = CheckpointBundle("rsl_rl", "Isaac-Cartpole", "physx", "none", (_FE,))

    assert bundle.collected_path(str(tmp_path)) == str(tmp_path / "rsl_rl" / "Isaac-Cartpole_physx_none_rsl_rl.pt")
    assert bundle.collected_path(str(tmp_path), _FE).endswith("_rsl_rl_feature_extractor.pth")
    assert (
        CheckpointBundle("rsl_rl", "Isaac-Cartpole")
        .collected_path(str(tmp_path))
        .endswith("rsl_rl/Isaac-Cartpole/checkpoint.pt")
    )


@pytest.mark.parametrize(
    "workflow,stem,written,not_written,other_dirs,preferred",
    [
        (
            "rl_games",
            "Isaac-Cartpole_physx_none_rl_games",
            ["last_Cartpole_ep_100_rew_5.pth"],
            ["config.yaml"],
            ["nn"],
            "Isaac-Cartpole_physx_none_rl_games.pth",
        ),
        ("rsl_rl", None, ["model_100.pt"], ["policy.pt"], None, None),
        ("sb3", None, ["model.zip", "model_1000_steps.zip"], ["events.out"], None, "model.zip"),
        ("skrl", None, ["agent_100.pt", "best_agent.pickle"], [], ["checkpoints"], "best_agent.pt"),
    ],
)
def test_workflow_selector_args_describe_the_files_each_library_writes(
    workflow, stem, written, not_written, other_dirs, preferred
):
    """``--checkpoint latest/best`` and the publish tooling select the policy from one description."""
    args = WORKFLOWS[workflow].selector_args(stem)

    assert all(re.fullmatch(args["checkpoint_pattern"], name) for name in written)
    assert not any(re.fullmatch(args["checkpoint_pattern"], name) for name in not_written)
    assert args.get("other_dirs") == other_dirs
    if preferred is None:
        assert "preferred_checkpoint_pattern" not in args
    else:
        assert re.fullmatch(args["preferred_checkpoint_pattern"], preferred)
        assert not re.fullmatch(args["preferred_checkpoint_pattern"], "x" + preferred)


def test_workflows_are_keyed_by_their_own_name():
    """The lookup key and the workflow's name are the same string."""
    assert all(name == workflow.name for name, workflow in WORKFLOWS.items())


def test_workflow_needing_the_experiment_name_demands_it():
    """rl_games names its best checkpoint after the experiment, so the stem cannot be omitted."""
    with pytest.raises(ValueError, match="stem"):
        WORKFLOWS["rl_games"].selector_args()


def _install_fake_retrieve(monkeypatch: pytest.MonkeyPatch, published_files: set[str]) -> list[tuple[str, str]]:
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

    monkeypatch.setattr(
        pretrained_checkpoint, "PRETRAINED_CHECKPOINT_PATH", "omniverse://IsaacLab/PretrainedCheckpoints"
    )
    monkeypatch.setattr(pretrained_checkpoint, "retrieve_file_path", _retrieve_file_path)
    return retrieved


def test_get_published_pretrained_checkpoint_downloads_to_the_bundle_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A bundle without companions downloads exactly its policy into a cache directory of its own."""
    monkeypatch.chdir(tmp_path)
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    retrieved = _install_fake_retrieve(monkeypatch, {remote_path})

    path = pretrained_checkpoint.get_published_pretrained_checkpoint("rsl_rl", "Isaac-Cartpole", "physx", "none")

    expected_download_dir = str(Path(".pretrained_checkpoints") / "rsl_rl" / "Isaac-Cartpole_physx_none_rsl_rl")
    assert retrieved == [(remote_path, expected_download_dir)]
    assert Path(path).is_relative_to(tmp_path / expected_download_dir)
    assert sorted(p.name for p in Path(path).parent.iterdir()) == ["Isaac-Cartpole_physx_none_rsl_rl.pt"]


def test_get_published_pretrained_checkpoint_downloads_the_declared_companions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """With an env cfg, the backends and the companion come from the config and land beside the policy."""
    monkeypatch.chdir(tmp_path)
    root = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl"
    stem = "Isaac-Reorient-Cube-Shadow-Camera_physx_rtx_rsl_rl"
    _install_fake_retrieve(monkeypatch, {f"{root}/{stem}.pt", f"{root}/{stem}_feature_extractor.pth"})
    env_cfg = _EnvCfg(camera=_CameraCfg(renderer_cfg=IsaacRtxRendererCfg()), extractor=_ExtractorCfg())

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl", "Isaac-Reorient-Cube-Shadow-Camera", env_cfg=env_cfg
    )

    assert path is not None and Path(path).name == f"{stem}.pt"
    companion = Path(path).parent / f"{stem}_feature_extractor.pth"
    assert companion.is_file()
    assert env_cfg.extractor.checkpoint.local_path == str(companion)
    assert env_cfg.extractor.frozen.local_path is None


def test_get_published_pretrained_checkpoint_tolerates_a_missing_companion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A policy whose declared companion is not published still resolves; the miss is not fatal."""
    monkeypatch.chdir(tmp_path)
    remote_path = "omniverse://IsaacLab/PretrainedCheckpoints/rsl_rl/Isaac-Cartpole_physx_none_rsl_rl.pt"
    _install_fake_retrieve(monkeypatch, {remote_path})

    env_cfg = _EnvCfg(extractor=_ExtractorCfg())

    path = pretrained_checkpoint.get_published_pretrained_checkpoint(
        "rsl_rl", "Isaac-Cartpole", "physx", "none", env_cfg=env_cfg
    )

    assert path is not None
    assert sorted(p.name for p in Path(path).parent.iterdir()) == ["Isaac-Cartpole_physx_none_rsl_rl.pt"]
    assert env_cfg.extractor.checkpoint.local_path is None
