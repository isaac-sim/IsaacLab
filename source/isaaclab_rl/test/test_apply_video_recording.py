# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for video-recording entrypoint helpers."""

from __future__ import annotations

import argparse
import logging
import os

import pytest
from isaaclab_newton.physics import NewtonCfg
from isaaclab_ov.renderers import OVRTXRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.viser import ViserVisualizerCfg

import isaaclab.app.sim_launcher as sim_launcher
from isaaclab.app import SimulationLauncher, add_launcher_args, get_settings_manager
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

from isaaclab_rl.entrypoints.common import (
    add_common_train_args,
    apply_video_recording,
    pre_launch_video_config,
    video_playback_steps,
    wrap_record_video,
)

from isaaclab_tasks.utils import setup_preset_cli

_KIT_LAUNCHER = "isaaclab_physx.app:KitLauncher"


def _args(**kwargs: object) -> argparse.Namespace:
    defaults = dict(video="viz", video_length=None, video_interval=None, visualizer=None)
    return argparse.Namespace(**{**defaults, **kwargs})


def _parse(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Parse *argv* the way the training entry points do."""
    parser = argparse.ArgumentParser()
    add_common_train_args(parser, agent_default=None, agent_help="Agent.")
    add_launcher_args(parser)
    return setup_preset_cli(parser, argv)


@pytest.mark.parametrize(
    ("argv", "expected_video", "expected_hydra"),
    [
        (["--video", "presets=a", "env.b=1"], "viz", ["presets=a", "env.b=1"]),
        (["--video", "kit", "x=1"], "kit", ["x=1"]),
        (["--video=sensor:cam"], "sensor:cam", []),
        (["--video"], "viz", []),
        ([], None, []),
        (["--video", "foo"], SystemExit, None),
    ],
    ids=["hydra-override-after-flag", "alias-then-override", "attached-source", "bare", "absent", "invalid"],
)
def test_video_cli(argv, expected_video, expected_hydra):
    """``--video`` takes an optional, validated source; a following Hydra override is passed on, in order."""
    argv = ["--task", "Isaac-Task", *argv]
    if expected_video is SystemExit:
        with pytest.raises(SystemExit):
            _parse(argv)
        return
    args, hydra_args = _parse(argv)

    assert (args.video, hydra_args) == (expected_video, expected_hydra)


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Record the launcher args of each runtime a launch starts, without starting one."""
    started = {}

    def fake_launcher(launcher_type: str):
        def start(launcher_args: dict) -> SimulationLauncher:
            started[launcher_type] = launcher_args
            return SimulationLauncher()

        return start

    monkeypatch.setattr(sim_launcher, "string_to_callable", fake_launcher)
    yield started
    get_settings_manager().set("/isaaclab/visualizer/types", None)


@pytest.mark.parametrize(
    ("cli", "expected"),
    [
        # --video records from the first capture-capable visualizer --viz selects, in its window
        (
            dict(visualizer=["viser", "kit", "newton_gl"]),
            ("viz:kit", [("kit", False), ("viser", None), ("newton_gl", False)]),
        ),
        # ...else from a headless newton_gl, also next to streaming-only visualizers
        (dict(), ("viz:newton_gl", [("newton_gl", True)])),
        (dict(visualizer=["viser"]), ("viz:newton_gl", [("viser", None), ("newton_gl", True)])),
        # viz:<type> records from the selected visualizer, else from a headless one added for the recording
        (dict(video="viz:newton_rtx", visualizer=["newton_rtx"]), ("viz:newton_rtx", [("newton_rtx", False)])),
        (dict(video="viz:newton_rtx"), ("viz:newton_rtx", [("newton_rtx", True)])),
        (dict(video="kit"), ("viz:kit", [("kit", True)])),
        # a scene sensor needs no visualizer
        (dict(video="sensor:wrist_camera:depth"), ("sensor:wrist_camera:depth", [])),
        # the warp frontend records the same way
        (dict(frontend="warp"), ("viz:newton_gl", [("newton_gl", True)])),
        # streaming visualizers have no frame capture
        (dict(video="viser", visualizer=["viser"]), "has no frame capture"),
    ],
    ids=[
        "viz-selected",
        "viz-none",
        "viz-streaming-only",
        "type-selected",
        "type-added",
        "kit-added",
        "sensor",
        "warp-frontend",
        "streaming-rejected",
    ],
)
def test_video_source_resolves_against_the_visualizer_selection(launches, cli, expected):
    """The launch resolves the ``--video`` source and runs a visualizer ``--viz`` did not select headless."""
    kit_cfg = KitVisualizerCfg(eye=(1.0, 2.0, 3.0))
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.sim.physics = NewtonCfg()
    env_cfg.sim.visualizer_cfgs = [kit_cfg, ViserVisualizerCfg()]
    args = _args(**cli)
    if isinstance(expected, str):
        with pytest.raises(ValueError, match=expected):
            pre_launch_video_config(env_cfg, args)
            with sim_launcher.launch_simulation(env_cfg, args):
                pass
        return
    pre_launch_video_config(env_cfg, args)
    with sim_launcher.launch_simulation(env_cfg, args):
        pass

    source, visualizers = expected
    assert [recorder.source for recorder in env_cfg.video_recorders] == [source]
    assert [(cfg.visualizer_type, getattr(cfg, "headless", None)) for cfg in env_cfg.sim.visualizer_cfgs] == visualizers
    # a Kit visualizer, selected or added, keeps its configured camera; an added one is a copy, so the
    # configured visualizer stays as configured for a later launch
    assert all(cfg.eye == kit_cfg.eye for cfg in env_cfg.sim.visualizer_cfgs if cfg.visualizer_type == "kit")
    assert not kit_cfg.headless
    # Kit starts only for a Kit visualizer, with cameras, windowed only when --viz selects it
    kit_args = launches.get(_KIT_LAUNCHER)
    assert (kit_args is not None) == ("kit" in dict(visualizers))
    if kit_args is not None:
        assert ("kit" in kit_args["visualizer"], kit_args["enable_cameras"]) == (not dict(visualizers)["kit"], True)
    assert (OVRTXRendererCfg.launcher_type in launches) == ("newton_rtx" in source)
    # a launch with a SimulationCfg leaves no selection behind for configs built afterwards
    assert get_settings_manager().get("/isaaclab/visualizer/types") == ""


def test_apply_video_recording_noop_when_video_false():
    """Video recording remains disabled when the CLI flag is unset or absent."""
    env_cfg = ManagerBasedRLEnvCfg()
    pre_launch_video_config(env_cfg, _args(video=None))
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=None))
    assert env_cfg.video_recorders == []

    apply_video_recording(env_cfg, "/tmp/logs", argparse.Namespace())
    assert env_cfg.video_recorders == []


@pytest.mark.parametrize(
    ("declared", "cli", "expected"),
    [
        (None, dict(video_length=42, video_interval=500), ("viz", 42, 500, os.path.join("/my/log", "videos", "play"))),
        (None, dict(), ("viz", VideoRecorderCfg().video_length, 2000, os.path.join("/my/log", "videos", "play"))),
        # a recorder the env config declares takes precedence over the --video source
        (
            VideoRecorderCfg(source="viz:kit", output_dir="/my/custom/path"),
            dict(video="sensor:cam", video_length=10),
            ("viz:kit", 10, VideoRecorderCfg().video_interval, "/my/custom/path"),
        ),
    ],
    ids=["cli_overrides", "cfg_defaults", "declared_recorder_wins"],
)
def test_apply_video_recording_schedules_the_cli_recorder(declared, cli: dict, expected: tuple):
    """``--video`` records into the log directory, taking CLI values over defaults, unless the config declares one."""
    args = _args(**cli)
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [declared] if declared else []
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, "/my/log", args, subdir="play")

    assert [(r.source, r.video_length, r.video_interval, r.output_dir) for r in env_cfg.video_recorders] == [expected]


@pytest.mark.parametrize(
    ("subdir", "existing_prefix", "checkpoint_name", "expected_prefix"),
    [
        ("play", "clip", "model_1200.pt", "clip_model_1200"),
        ("play", "clip_model_1200", "model_120.pt", "clip_model_1200_model_120"),
        ("play", "clip", "custom_1200.pt", "clip"),
        ("train", "clip", "model_1200.pt", "clip"),
    ],
)
def test_apply_video_recording_labels_play_video_with_checkpoint_stem(
    subdir: str, existing_prefix: str, checkpoint_name: str, expected_prefix: str
):
    """Play videos append numeric model checkpoint stems as distinct tokens; training videos keep their prefix."""
    existing = VideoRecorderCfg()
    existing.output_filename_prefix = existing_prefix

    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/my/log", _args(), subdir=subdir, checkpoint_path=f"/my/log/{checkpoint_name}")

    assert env_cfg.video_recorders[0].output_filename_prefix == expected_prefix


def test_apply_video_recording_patches_existing_recorders():
    """CLI length and interval overrides preserve other recorder settings."""
    existing = VideoRecorderCfg()
    existing.source = "sensor:tiled_camera"
    existing.output_dir = "/my/custom/path"
    existing.fps = 60

    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10, video_interval=500))

    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "sensor:tiled_camera"
    assert recorder.output_dir == "/my/custom/path"
    assert recorder.fps == 60
    assert recorder.video_length == 10
    assert recorder.video_interval == 500


def test_wrap_record_video_is_noop_stub(caplog: pytest.LogCaptureFixture) -> None:
    """The compatibility stub warns only when video recording is requested."""
    env = object()

    result = wrap_record_video(env, "/tmp/logs", _args(video=None))
    assert result is env
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        result = wrap_record_video(env, "/tmp/logs", _args())
    assert result is env
    assert any("wrap_record_video" in r.message for r in caplog.records)


@pytest.mark.parametrize(
    ("recorders", "cli_args", "expected_steps"),
    [
        # (video_length, step_offset) per recorder; the second recorder's first clip ends last, at 80 + 50
        ([(100, 0), (50, 80), (120, 5)], dict(), 130),
        # --video_length replaces the length but the configured offset still delays the clip
        ([(200, 30)], dict(video_length=10), 40),
        ([(200, 0)], dict(video=None), None),
    ],
    ids=["waits_for_last_recorder", "cli_length_keeps_offset", "unbounded_without_video"],
)
def test_video_playback_steps(recorders: list[tuple[int, int]], cli_args: dict, expected_steps: int | None):
    """Playback runs until every recorder has finished its first clip, and is unbounded without --video."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [
        VideoRecorderCfg(source="viz:kit", video_length=length, step_offset=offset) for length, offset in recorders
    ]
    args = _args(**cli_args)
    apply_video_recording(env_cfg, "/tmp/logs", args)

    assert video_playback_steps(args, env_cfg) == expected_steps
