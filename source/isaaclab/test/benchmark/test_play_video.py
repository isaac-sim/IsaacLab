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
import importlib.util
from pathlib import Path

import pytest

from isaaclab_rl.entrypoints.common import add_video_args, play_video_dir

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
def test_adapter_declares_video_arguments(backend: str) -> None:
    assert "add_video_args(parser, include_interval=False)" in _adapter_source(backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_enables_cameras_for_video(backend: str) -> None:
    # Without this the env renders nothing and the recording is blank.
    assert "enable_cameras_for_video" in _adapter_source(backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_requests_rgb_array_render_mode(backend: str) -> None:
    # RecordVideo needs a render mode; gym.make defaults to None.
    assert 'render_mode="rgb_array"' in _adapter_source(backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_wraps_the_env_for_recording(backend: str) -> None:
    assert "wrap_record_video_play" in _adapter_source(backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_records_the_video_path_on_the_bundle(backend: str) -> None:
    # The schema field is useless if nothing ever sets it.
    assert "video_path=_common.play_video_dir(" in _adapter_source(backend)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_adapter_module_is_importable_source(backend: str) -> None:
    spec = importlib.util.spec_from_file_location(
        f"_probe_{backend}", _ADAPTER_ROOT / backend / f"benchmark_play_{backend}.py"
    )
    assert spec is not None
