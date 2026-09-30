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
from isaaclab_physx.physics import PhysxCfg
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
        (["--video", "viz:kit", "x=1"], "viz:kit", ["x=1"]),
        (["--video=sensor:cam"], "sensor:cam", []),
        (["--video"], "viz", []),
        ([], None, []),
    ],
    ids=["hydra-override-after-flag", "source-then-override", "attached-source", "bare", "absent"],
)
def test_video_cli_takes_a_source_but_never_a_hydra_override(argv, expected_video, expected_hydra):
    """``--video`` takes an optional source; a following Hydra override is passed on, in order, instead."""
    args, hydra_args = _parse(["--task", "Isaac-Task", *argv])

    assert args.video == expected_video
    assert hydra_args == expected_hydra


def test_video_cli_rejects_an_invalid_source(capsys: pytest.CaptureFixture):
    """A ``--video`` value outside the source grammar is a CLI error."""
    with pytest.raises(SystemExit):
        _parse(["--video", "foo"])
    assert "Invalid video source 'foo'" in capsys.readouterr().err


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


def _launch(args: argparse.Namespace, visualizer_cfgs: list | None = None, physics=None) -> ManagerBasedRLEnvCfg:
    """Configure ``--video`` on an env config and launch it, the way the entry points do."""
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.sim.physics = physics or NewtonCfg()
    env_cfg.sim.visualizer_cfgs = visualizer_cfgs or []
    pre_launch_video_config(env_cfg, args)
    with sim_launcher.launch_simulation(env_cfg, args):
        apply_video_recording(env_cfg, "/my/log", args)
    return env_cfg


@pytest.mark.parametrize(
    ("video", "visualizer", "expected_source", "expected_visualizers"),
    [
        # --video records from the first capture-capable visualizer --viz selects, in its window
        ("viz", ["viser", "kit", "newton_gl"], "viz:kit", [("viser", None), ("kit", False), ("newton_gl", False)]),
        # ...else from a headless newton_gl, also next to streaming-only visualizers
        ("viz", None, "viz:newton_gl", [("newton_gl", True)]),
        ("viz", ["viser"], "viz:newton_gl", [("viser", None), ("newton_gl", True)]),
        # viz:<type> records from the selected visualizer, else from a headless one added for the recording
        ("viz:newton_rtx", None, "viz:newton_rtx", [("newton_rtx", True)]),
        ("viz:newton_rtx", ["newton_rtx"], "viz:newton_rtx", [("newton_rtx", False)]),
        ("viz:newton_gl", ["viser"], "viz:newton_gl", [("viser", None), ("newton_gl", True)]),
        # a scene sensor needs no visualizer
        ("sensor:wrist_camera:depth", None, "sensor:wrist_camera:depth", []),
    ],
    ids=[
        "viz-selected",
        "viz-none",
        "viz-streaming-only",
        "type-added",
        "type-selected",
        "type-next-to-viser",
        "sensor",
    ],
)
def test_video_source_resolves_against_the_visualizer_selection(
    launches, video, visualizer, expected_source, expected_visualizers
):
    """The launch resolves the ``--video`` source and runs a visualizer ``--viz`` did not select headless."""
    args = _args(video=video, visualizer=visualizer)
    env_cfg = _launch(args)

    assert [recorder.source for recorder in env_cfg.video_recorders] == [expected_source]
    assert [
        (cfg.visualizer_type, getattr(cfg, "headless", None)) for cfg in env_cfg.sim.visualizer_cfgs
    ] == expected_visualizers
    # only the Kit visualizer needs Kit, and only newton_rtx starts OVRTX
    assert (_KIT_LAUNCHER in launches) == ("kit" in (visualizer or []))
    assert (OVRTXRendererCfg.launcher_type in launches) == ("newton_rtx" in expected_source)
    # the SimulationContext keeps the visualizers the launch resolved
    assert get_settings_manager().get("/isaaclab/visualizer/types") == ",".join(
        name for name, _ in expected_visualizers
    )


def test_video_from_an_unselected_kit_visualizer_launches_headless_kit_with_its_configured_camera(launches):
    """``--video viz:kit`` without ``--viz`` records from the configured Kit visualizer, headless, in a headless Kit."""
    kit_cfg = KitVisualizerCfg(eye=(1.0, 2.0, 3.0))
    env_cfg = _launch(_args(video="viz:kit"), [kit_cfg, ViserVisualizerCfg()], physics=PhysxCfg())

    assert env_cfg.sim.visualizer_cfgs == [kit_cfg]
    assert kit_cfg.headless and kit_cfg.eye == (1.0, 2.0, 3.0)
    kit_args = launches[_KIT_LAUNCHER]
    # Kit opens a window only for a selected Kit visualizer; recording needs its rendering
    assert kit_args["visualizer"] == [] and kit_args["enable_cameras"] is True


@pytest.mark.parametrize("video", ["viz:viser", "viz:rerun"])
def test_video_from_a_streaming_visualizer_is_rejected(launches, video):
    """Streaming visualizers have no frame capture, so recording from one fails at launch."""
    with pytest.raises(ValueError, match="has no frame capture"):
        _launch(_args(video=video, visualizer=["viser"]))


def test_declared_video_recorders_take_precedence_over_the_cli_source(launches):
    """``--video`` keeps the recorders an env config declares, applying only the output directory and schedule."""
    declared = VideoRecorderCfg(source="viz:kit", output_dir="/my/custom/path")
    env_cfg = ManagerBasedRLEnvCfg()
    env_cfg.video_recorders = [declared]
    args = _args(video="sensor:cam", video_length=10)
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, "/my/log", args)

    assert env_cfg.video_recorders == [declared]
    assert (declared.source, declared.output_dir, declared.video_length) == ("viz:kit", "/my/custom/path", 10)


def test_video_cli_is_rejected_with_the_warp_frontend():
    """Video recording requires the torch frontend."""
    with pytest.raises(ValueError, match="--frontend 'warp'"):
        pre_launch_video_config(ManagerBasedRLEnvCfg(), _args(frontend="warp"))


def test_apply_video_recording_noop_when_video_false():
    """Video recording remains disabled when the CLI flag is unset or absent."""
    env_cfg = ManagerBasedRLEnvCfg()
    pre_launch_video_config(env_cfg, _args(video=None))
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=None))
    assert env_cfg.video_recorders == []

    apply_video_recording(env_cfg, "/tmp/logs", argparse.Namespace())
    assert env_cfg.video_recorders == []


@pytest.mark.parametrize(
    ("video_length", "video_interval", "expected_length", "expected_interval"),
    [(42, 500, 42, 500), (None, None, VideoRecorderCfg().video_length, 2000)],
    ids=["cli_overrides", "cfg_defaults"],
)
def test_apply_video_recording_schedules_the_cli_recorder(
    video_length: int | None, video_interval: int | None, expected_length: int, expected_interval: int
):
    """The ``--video`` recorder writes into the log directory, taking CLI values over defaults."""
    args = _args(video_length=video_length, video_interval=video_interval)
    env_cfg = ManagerBasedRLEnvCfg()
    pre_launch_video_config(env_cfg, args)
    apply_video_recording(env_cfg, "/my/log", args, subdir="play")

    assert len(env_cfg.video_recorders) == 1
    recorder = env_cfg.video_recorders[0]
    assert recorder.source == "viz"
    assert (recorder.video_length, recorder.video_interval) == (expected_length, expected_interval)
    assert recorder.output_dir == os.path.join("/my/log", "videos", "play")
    assert recorder.output_filename_prefix == "clip"


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
