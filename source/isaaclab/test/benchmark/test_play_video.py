# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Video recording on the play benchmark adapters.

``PlayBundle.video_path`` and ``build_play_bundle(video_path=...)`` existed long
before any adapter could populate them, so a camera task was benchmarked
headless. These tests pin the wiring that closed that gap.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from isaaclab_rl.entrypoints.common import add_video_args, latest_checkpoint_path, play_video_dir

_BACKENDS = ("rsl_rl", "rl_games", "skrl", "sb3")
# <repo>/source/isaaclab/test/benchmark/ -> <repo>/source/isaaclab/isaaclab/...
_ADAPTER_ROOT = Path(__file__).resolve().parents[2] / "isaaclab" / "benchmark" / "entrypoints" / "backends"


def _adapter_source(backend: str) -> str:
    return (_ADAPTER_ROOT / backend / f"benchmark_play_{backend}.py").read_text()


def test_play_video_args_omit_the_training_interval() -> None:
    # A play run is one bounded rollout, so there is no later interval to catch.
    parser = argparse.ArgumentParser()
    add_video_args(parser, include_interval=False)
    args = parser.parse_args([])

    assert args.video is False
    assert args.video_length == 200
    assert not hasattr(args, "video_interval")


def test_training_video_args_keep_the_interval() -> None:
    parser = argparse.ArgumentParser()
    add_video_args(parser, include_interval=True)
    assert parser.parse_args([]).video_interval == 2000


def test_video_dir_is_none_without_the_flag() -> None:
    args = argparse.Namespace(video=False)
    assert play_video_dir("/out", args) is None


def test_video_dir_is_reported_when_requested() -> None:
    args = argparse.Namespace(video=True)
    assert play_video_dir("/out", args) == "/out/videos/play"


def test_video_dir_tolerates_a_namespace_without_the_flag() -> None:
    # Callers that never added the argument must not raise.
    assert play_video_dir("/out", argparse.Namespace()) is None


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_wires_video(backend: str) -> None:
    source = _adapter_source(backend)
    assert "add_video_args(parser, include_interval=False)" in source
    # Without enabled cameras the env renders nothing and the recording is blank.
    assert "enable_cameras_for_video" in source
    # RecordVideo needs a render mode; gym.make defaults to None.
    assert 'render_mode="rgb_array"' in source
    assert "wrap_record_video_play" in source
    # The schema field is useless if nothing ever sets it.
    assert "video_path=_common.play_video_dir(" in source


@pytest.mark.parametrize("backend", ("rsl_rl", "rl_games", "skrl"))
def test_training_reports_the_checkpoint_it_wrote(backend: str) -> None:
    # checkpoint_path was hardcoded None in three of four train adapters, so the
    # schema field was dead and a chained play step had nothing to read.
    source = (_ADAPTER_ROOT / backend / f"benchmark_train_{backend}.py").read_text()
    assert "latest_checkpoint_path(" in source
    assert "checkpoint_path = None" not in source
    assert "checkpoint_path=None" not in source


def test_latest_checkpoint_path_finds_the_newest(tmp_path: Path) -> None:
    from isaaclab_rl.entrypoints.common import latest_checkpoint_path

    nested = tmp_path / "nn"
    nested.mkdir()
    (nested / "model_100.pt").write_text("x")
    older = nested / "model_100.pt"
    newer = nested / "model_200.pt"
    newer.write_text("x")
    import os

    os.utime(older, (1, 1))

    assert latest_checkpoint_path(str(tmp_path)) == str(newer.resolve())


def test_latest_checkpoint_path_is_none_when_nothing_was_written(tmp_path: Path) -> None:
    from isaaclab_rl.entrypoints.common import latest_checkpoint_path

    assert latest_checkpoint_path(str(tmp_path)) is None


@pytest.mark.parametrize(
    "filename",
    [
        "model_100.pt",  # rsl_rl
        "agent_4800.pt",  # skrl
        "best_agent.pt",  # skrl
        "Isaac-Cartpole-Direct_4800.pt",  # skrl, named after the experiment
        "last_Isaac_ep_50.pth",  # rl_games
        "model.zip",  # sb3
    ],
)
def test_latest_checkpoint_path_matches_every_library_naming(tmp_path: Path, filename: str) -> None:
    # A `model_*.pt` glob matched rsl_rl only, so every SKRL row reported no
    # checkpoint and a chained play step had nothing to roll out.
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / filename).write_text("x")

    assert latest_checkpoint_path(str(tmp_path)) == str((checkpoints / filename).resolve())
