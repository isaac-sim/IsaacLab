# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for apply_video_recording and wrap_record_video.

Pure Python — no simulation context or Kit app required.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

from isaaclab_rl.entrypoints.common import apply_video_recording, wrap_record_video


def _args(**kwargs) -> SimpleNamespace:
    # video_length and video_interval default to None (not passed at CLI)
    defaults = dict(video=True, video_length=None, video_interval=None)
    return SimpleNamespace(**{**defaults, **kwargs})


def _env_cfg():
    cfg = MagicMock()
    cfg.video_recorders = []
    # Simulate an env with no concrete visualizer configured so that apply_video_recording
    # uses the fallback "visualizer" source string rather than resolving a MagicMock type.
    cfg.sim.visualizer_cfgs = []
    cfg.sim.default_visualizer_cfg.visualizer_type = None
    return cfg


def test_apply_video_recording_noop_when_video_false():
    """video=False (or missing) leaves env_cfg.video_recorders untouched."""
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/tmp/logs", _args(video=False))
    assert env_cfg.video_recorders == []

    apply_video_recording(env_cfg, "/tmp/logs", SimpleNamespace())
    assert env_cfg.video_recorders == []


def test_apply_video_recording_injects_correct_recorder():
    """video=True with no pre-configured recorders creates a default with log_dir output_dir."""

    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args(video_length=42, video_interval=500), subdir="play")
    assert len(env_cfg.video_recorders) == 1
    rec = env_cfg.video_recorders[0]
    # Kit not running → Newton GL auto-created; source is the concrete backend type.
    assert rec.source == "visualizer:newton_gl"
    assert rec.video_length == 42  # CLI override applied
    assert rec.video_interval == 500  # CLI override applied
    assert rec.output_dir == os.path.join("/my/log", "videos", "play")


def test_apply_video_recording_uses_cfg_defaults_when_cli_not_passed():
    """video=True without --video_length/--video_interval uses historical CLI cadence."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    defaults = VideoRecorderCfg()
    env_cfg = _env_cfg()
    apply_video_recording(env_cfg, "/my/log", _args())  # no video_length/interval
    rec = env_cfg.video_recorders[0]
    assert rec.video_length == defaults.video_length  # default kept
    # CLI fallback uses video_interval=2000 to match historical --video cadence
    # (record immediately then every 2000 steps), not VideoRecorderCfg()'s default of 0.
    assert rec.video_interval == 2000


def test_apply_video_recording_patches_existing_recorders():
    """Existing recorders are kept; only video_length and video_interval are overwritten."""
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg

    existing = VideoRecorderCfg()
    existing.source = "sensor:tiled_camera"
    existing.output_dir = "/my/custom/path"
    existing.fps = 60

    env_cfg = _env_cfg()
    env_cfg.video_recorders = [existing]
    apply_video_recording(env_cfg, "/tmp/logs", _args(video_length=10, video_interval=500))

    # existing recorder is kept — not replaced
    assert len(env_cfg.video_recorders) == 1
    rec = env_cfg.video_recorders[0]
    assert rec.source == "sensor:tiled_camera"  # preserved
    assert rec.output_dir == "/my/custom/path"  # preserved
    assert rec.fps == 60  # preserved
    assert rec.video_length == 10  # CLI override applied
    assert rec.video_interval == 500  # CLI override applied


def test_apply_video_recording_injects_kit_visualizer_when_kit_is_running():
    """When Kit is running and --video given without --viz, KitVisualizerCfg is injected."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock, patch

    kit_cfg_instance = object()
    MockKitVisualizerCfg = MagicMock(return_value=kit_cfg_instance)

    fake_kit_module = ModuleType("isaaclab_visualizers.kit")
    fake_kit_module.KitVisualizerCfg = MockKitVisualizerCfg
    fake_visualizers_module = ModuleType("isaaclab_visualizers")

    sim_cfg = SimpleNamespace(visualizer_cfgs=[])
    env_cfg = SimpleNamespace(video_recorders=[], sim=sim_cfg)

    # Patch omni.kit.app into sys.modules so that has_kit() returns True,
    # simulating the Kit/Isaac Sim runtime being active.
    # has_kit() calls mod.get_app() and returns True only when it is not None.
    fake_omni_kit_app = ModuleType("omni.kit.app")
    fake_omni_kit_app.get_app = lambda: object()  # type: ignore[attr-defined]
    with patch.dict(
        sys.modules,
        {
            "isaaclab_visualizers": fake_visualizers_module,
            "isaaclab_visualizers.kit": fake_kit_module,
            "omni.kit.app": fake_omni_kit_app,
        },
    ):
        apply_video_recording(env_cfg, "/my/log", _args())

    # KitVisualizerCfg() must have been appended to sim_cfg.visualizer_cfgs
    assert len(sim_cfg.visualizer_cfgs) == 1
    assert sim_cfg.visualizer_cfgs[0] is kit_cfg_instance
    MockKitVisualizerCfg.assert_called_once_with(headless=True)
    # The default recorder must reference the injected kit visualizer
    assert len(env_cfg.video_recorders) == 1
    assert env_cfg.video_recorders[0].source == "visualizer:kit"


def test_apply_video_recording_injects_newton_gl_when_kit_not_running():
    """When Kit is NOT running and --video given without --viz, Newton GL is injected instead."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock, patch

    newton_gl_cfg_instance = object()
    MockNewtonGLVisualizerCfg = MagicMock(return_value=newton_gl_cfg_instance)

    fake_newton_module = ModuleType("isaaclab_visualizers.newton")
    fake_newton_module.NewtonGLVisualizerCfg = MockNewtonGLVisualizerCfg
    fake_visualizers_module = ModuleType("isaaclab_visualizers")

    sim_cfg = SimpleNamespace(visualizer_cfgs=[])
    env_cfg = SimpleNamespace(video_recorders=[], sim=sim_cfg)

    # Do NOT patch omni.kit.app so that has_kit() returns False (kitless path).
    with patch.dict(
        sys.modules,
        {
            "isaaclab_visualizers": fake_visualizers_module,
            "isaaclab_visualizers.newton": fake_newton_module,
        },
    ):
        # Ensure omni.kit.app is absent so has_kit() correctly returns False.
        sys.modules.pop("omni.kit.app", None)
        apply_video_recording(env_cfg, "/my/log", _args())

    # NewtonGLVisualizerCfg(headless=True) must have been appended
    assert len(sim_cfg.visualizer_cfgs) == 1
    assert sim_cfg.visualizer_cfgs[0] is newton_gl_cfg_instance
    MockNewtonGLVisualizerCfg.assert_called_once_with(headless=True)
    # The default recorder must reference the injected newton_gl visualizer
    assert len(env_cfg.video_recorders) == 1
    assert env_cfg.video_recorders[0].source == "visualizer:newton_gl"


def test_apply_video_recording_rejects_viz_none_with_video():
    """--viz none combined with --video raises ValueError with a clear message.

    AppLauncher._parse_visualizer_csv("none") returns None (not ["none"]), and
    ExplicitAction sets visualizer_explicit=True.  Simulate that parsed state.
    """
    sim_cfg = SimpleNamespace(visualizer_cfgs=[])
    env_cfg = SimpleNamespace(video_recorders=[], sim=sim_cfg)

    import pytest

    with pytest.raises(ValueError, match="--video is not compatible with --viz none"):
        apply_video_recording(env_cfg, "/my/log", _args(visualizer=None, visualizer_explicit=True))


def test_wrap_record_video_is_noop_stub(caplog):
    """wrap_record_video returns the env unchanged and warns only when video=True."""
    env = MagicMock()

    result = wrap_record_video(env, "/tmp/logs", _args(video=False))
    assert result is env
    assert not caplog.records

    with caplog.at_level(logging.WARNING, logger="isaaclab_rl.entrypoints.common"):
        result = wrap_record_video(env, "/tmp/logs", _args(video=True))
    assert result is env
    assert any("wrap_record_video" in r.message for r in caplog.records)
