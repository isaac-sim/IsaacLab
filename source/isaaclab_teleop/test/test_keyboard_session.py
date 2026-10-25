# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for keyboard teleop sessions fed by scripted key events in place of a visualizer window.

The ``se3_keyboard_teleop_cfg`` pipeline runs in a real Isaac Capture session. The in-process
keyboard never touches the OpenXR session handles, so a placeholder OpenXR session stands in for a
CloudXR runtime. The session lifecycle is checked without stepping, since its combined pipeline
also polls the controllers through OpenXR.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("isaacteleop")

from isaaclab_teleop import se3_keyboard_teleop_cfg
from isaaclab_teleop.session_lifecycle import TeleopSessionLifecycle
from isaacteleop.retargeting_engine.deviceio_source_nodes import KeyboardSource, find_sources

_SETTLE_STEPS = 3


def _keyboard_cfg(**kwargs):
    """The SE(3) keyboard config, retargeting synchronously so each step reflects the current keys."""
    from isaacteleop.teleop_session_manager import RetargetingExecutionConfig

    cfg = se3_keyboard_teleop_cfg(**kwargs)
    cfg.retargeting_execution = RetargetingExecutionConfig(mode="sync")
    return cfg


class _PlaceholderOpenXRSession:
    """Non-null handles that only OpenXR-backed tracker impls would dereference."""

    def __init__(self, *args, **kwargs):
        pass

    def get_handles(self):
        from isaacteleop.oxr import OpenXRSessionHandles

        return OpenXRSessionHandles(1, 1, 1, 1)

    def get_provider_snapshot(self):
        return SimpleNamespace(state="AVAILABLE", headset_state="CONNECTED", reason="NONE", result_code=None, error="")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def keyboard_session(visualizer, monkeypatch):
    """The keyboard config's pipeline in a real Isaac Capture session, fed by *visualizer*."""
    from isaacteleop import oxr
    from isaacteleop.teleop_session_manager import TeleopSession, TeleopSessionConfig

    monkeypatch.setattr(oxr, "OpenXRSession", _PlaceholderOpenXRSession)
    cfg = _keyboard_cfg(pos_sensitivity=0.5, rot_sensitivity=0.5)
    pipeline = cfg.pipeline_builder()
    source = visualizer.key_event_source
    attachments = [keyboard.attach(source, capture=False) for keyboard in find_sources(pipeline, KeyboardSource)]
    session_config = TeleopSessionConfig(
        app_name="KeyboardSessionTest", pipeline=pipeline, retargeting_execution=cfg.retargeting_execution
    )
    with TeleopSession(session_config) as session:
        yield session
    for attachment in attachments:
        attachment.close()


def _settle(session):
    result = None
    for _ in range(_SETTLE_STEPS):
        result = session.step()
    return result["action"][0]


def test_held_key_drives_the_action(keyboard_session, visualizer):
    visualizer.press("KeyW")
    action = _settle(keyboard_session)
    assert action[0] > 0.0

    visualizer.release("KeyW")
    action = _settle(keyboard_session)
    assert action[0] == 0.0


def test_focus_loss_releases_held_keys(keyboard_session, visualizer):
    visualizer.press("KeyW")
    _settle(keyboard_session)

    visualizer.blur()
    action = _settle(keyboard_session)

    assert action[0] == 0.0


class _Visualizer:
    """A running visualizer: an Isaac Lab keyboard source with key helpers, and its camera-keys toggle."""

    def __init__(self) -> None:
        from isaaclab.visualizers import KeyEventSource

        self.key_event_source = KeyEventSource()
        self.camera_keys: list[bool] = []

    def set_camera_keys_enabled(self, enabled: bool) -> None:
        self.camera_keys.append(enabled)

    @property
    def listener_count(self) -> int:
        return len(self.key_event_source._subscriptions)

    def press(self, code: str) -> None:
        self.key_event_source._emit_key(code, True)

    def release(self, code: str) -> None:
        self.key_event_source._emit_key(code, False)

    def tap(self, code: str) -> None:
        self.press(code)
        self.release(code)

    def blur(self) -> None:
        self.key_event_source._emit_focus_lost()


def _start(lifecycle):
    """Start a lifecycle reading *visualizers*, without launching CloudXR, a system check or OpenXR."""
    import isaacteleop.teleop_session_manager as tsm

    with (
        patch.object(lifecycle, "_ensure_cloudxr_runtime") as ensure_cloudxr,
        patch.object(lifecycle, "_run_system_check"),
        patch.object(tsm, "TeleopSession", MagicMock()),
    ):
        lifecycle.start()
    return ensure_cloudxr


def _started_lifecycle(cfg, *visualizers, **kwargs):
    lifecycle = TeleopSessionLifecycle(cfg, use_kit_xr_bridge=False, visualizers=list(visualizers), **kwargs)
    return lifecycle, _start(lifecycle)


@pytest.fixture
def visualizer():
    return _Visualizer()


def test_keyboard_session_takes_the_standard_path(visualizer):
    """A keyboard session launches the CloudXR profile and uses the control channel, like any other."""
    lifecycle, ensure_cloudxr = _started_lifecycle(
        _keyboard_cfg(), visualizer, cloudxr_env_file="/nonexistent/cloudxr.env"
    )
    try:
        ensure_cloudxr.assert_called_once()
        assert lifecycle._message_processor is not None
        outputs = lifecycle._pipeline.output_types()
        assert TeleopSessionLifecycle._CONTROLLER_RIGHT_KEY in outputs
    finally:
        lifecycle.stop()


def test_visualizer_feeds_the_keyboard_and_the_control_keys(visualizer):
    """The pipeline's keyboard and the control keys read the window; its camera keys are off until stop."""
    windows = [visualizer]
    lifecycle = TeleopSessionLifecycle(_keyboard_cfg(), use_kit_xr_bridge=False, visualizers=windows)
    _start(lifecycle)
    assert visualizer.listener_count == 2
    assert visualizer.camera_keys == [False]
    later = _Visualizer()
    windows.append(later)
    lifecycle._attach_keyboards()
    lifecycle._attach_keyboards()
    assert later.listener_count == 2 and later.camera_keys == [False]

    lifecycle.stop()
    assert later.listener_count == 0 and later.camera_keys == [False, True]
    assert visualizer.listener_count == 0
    assert visualizer.camera_keys == [False, True]


KEY_R, KEY_B = 19, 48


def test_each_control_key_press_is_drained_once(visualizer):
    """Presses reach the app once each, with no session step in between: repeats of a held key are dropped."""
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), visualizer)
    try:
        visualizer.press("KeyR")
        visualizer.press("KeyR")  # still held: a repeat, not a press
        visualizer.release("KeyR")
        visualizer.tap("KeyB")
        assert lifecycle.drain_pressed_keys() == [KEY_R, KEY_B]
        assert lifecycle.drain_pressed_keys() == []
    finally:
        lifecycle.stop()


def test_control_keys_work_without_a_pipeline_keyboard(visualizer):
    """A pipeline with no keyboard still gets B/P/R, and the window keeps its camera keys."""
    from isaaclab_teleop import se3_spacemouse_teleop_cfg

    lifecycle, _ = _started_lifecycle(se3_spacemouse_teleop_cfg(), visualizer)
    try:
        assert visualizer.camera_keys == []
        visualizer.tap("KeyB")
        assert lifecycle.drain_pressed_keys() == [KEY_B]
    finally:
        lifecycle.stop()


def test_presses_do_not_carry_into_the_next_session(visualizer):
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), visualizer)
    visualizer.tap("KeyR")
    lifecycle.stop()
    _start(lifecycle)
    try:
        visualizer.press("KeyR")  # the stale tap is gone; a new press counts
        assert lifecycle.drain_pressed_keys() == [KEY_R]
    finally:
        lifecycle.stop()


def test_another_window_losing_focus_does_not_repeat_a_held_key():
    """Each window tracks its own held keys, so one window's blur leaves another's repeat suppression intact."""
    first, second = _Visualizer(), _Visualizer()
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), first, second)
    try:
        first.press("KeyR")
        second.blur()
        first.press("KeyR")  # an autorepeat of the key the first window still holds
        assert lifecycle.drain_pressed_keys() == [KEY_R]
    finally:
        lifecycle.stop()


def test_a_failing_window_does_not_stop_the_session(visualizer):
    """A window whose keyboard cannot be attached is skipped with a warning; the others still work."""
    broken = _Visualizer()

    def set_camera_keys(enabled):
        if not enabled:
            raise RuntimeError("no window")

    broken.set_camera_keys_enabled = set_camera_keys
    lifecycle, _ = _started_lifecycle(_keyboard_cfg(), broken, visualizer)
    try:
        assert broken.listener_count == 0
        visualizer.tap("KeyB")
        assert lifecycle.drain_pressed_keys() == [KEY_B]
    finally:
        lifecycle.stop()
