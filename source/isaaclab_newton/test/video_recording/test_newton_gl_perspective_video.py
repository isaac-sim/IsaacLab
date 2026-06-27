# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton GL perspective video visibility."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, call

import pytest
from isaaclab_newton.video_recording.newton_gl_perspective_video import NewtonGlPerspectiveVideo


def _make_capture() -> NewtonGlPerspectiveVideo:
    """Create a capture without importing Newton viewer dependencies."""
    cfg = SimpleNamespace(
        window_width=1280,
        window_height=720,
        eye=(7.5, 7.5, 7.5),
        lookat=(0.0, 0.0, 0.0),
        horiz_fov_deg=60.0,
    )
    return NewtonGlPerspectiveVideo(cfg)


def _initialize_with_fake_viewer(
    capture: NewtonGlPerspectiveVideo,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MagicMock, object]:
    """Initialize a capture with lightweight fake lazy-import dependencies."""
    model = object()
    viewer = MagicMock()
    viewer.camera = SimpleNamespace(fov=None)
    viewer_factory = MagicMock(return_value=viewer)

    physics_module = ModuleType("isaaclab_newton.physics")
    physics_module.NewtonManager = SimpleNamespace(get_model=MagicMock(return_value=model))

    pyglet_module = ModuleType("pyglet")
    pyglet_module.options = {}

    newton_module = ModuleType("newton")
    newton_module.__path__ = []
    newton_viewer_module = ModuleType("newton.viewer")
    newton_viewer_module.ViewerGL = viewer_factory
    newton_module.viewer = newton_viewer_module

    monkeypatch.setitem(sys.modules, "isaaclab_newton.physics", physics_module)
    monkeypatch.setitem(sys.modules, "pyglet", pyglet_module)
    monkeypatch.setitem(sys.modules, "newton", newton_module)
    monkeypatch.setitem(sys.modules, "newton.viewer", newton_viewer_module)
    monkeypatch.setattr(capture, "_apply_camera", MagicMock())

    capture._ensure_viewer()
    viewer_factory.assert_called_once_with(width=1280, height=720, headless=True)
    return viewer, model


def test_set_visible_worlds_defers_until_viewer_initialization(monkeypatch: pytest.MonkeyPatch):
    """A visibility request is pending until ViewerGL exists, then applies after set_model."""
    capture = _make_capture()

    capture.set_visible_worlds([0, 1, 2, 3])

    assert capture._viewer is None
    assert capture._init_attempted is False

    viewer, model = _initialize_with_fake_viewer(capture, monkeypatch)

    viewer.set_visible_worlds.assert_called_once_with([0, 1, 2, 3])
    assert viewer.method_calls.index(call.set_model(model)) < viewer.method_calls.index(
        call.set_visible_worlds([0, 1, 2, 3])
    )


def test_set_visible_worlds_preinit_uses_last_request(monkeypatch: pytest.MonkeyPatch):
    """Only the last of multiple pre-initialization selections is applied."""
    capture = _make_capture()

    capture.set_visible_worlds([0, 1])
    capture.set_visible_worlds([2, 3])

    viewer, _ = _initialize_with_fake_viewer(capture, monkeypatch)

    viewer.set_visible_worlds.assert_called_once_with([2, 3])


def test_set_visible_worlds_copies_preinit_input(monkeypatch: pytest.MonkeyPatch):
    """Mutating the caller's list does not alter the pending viewer selection."""
    capture = _make_capture()
    world_indices = [0, 1]

    capture.set_visible_worlds(world_indices)
    world_indices.append(2)

    viewer, _ = _initialize_with_fake_viewer(capture, monkeypatch)

    viewer.set_visible_worlds.assert_called_once_with([0, 1])


def test_set_visible_worlds_deduplicates_live_viewer_calls():
    """Equal selections do not rebuild ViewerGL visibility caches."""
    capture = _make_capture()
    viewer = MagicMock()
    capture._viewer = viewer

    capture.set_visible_worlds([0, 1, 2, 3])
    capture.set_visible_worlds([0, 1, 2, 3])
    capture.set_visible_worlds([1, 3])
    capture.set_visible_worlds([1, 3])

    assert viewer.set_visible_worlds.call_args_list == [
        call([0, 1, 2, 3]),
        call([1, 3]),
    ]


def test_set_visible_worlds_distinguishes_all_from_empty_selection():
    """None clears the filter while an empty list selects no worlds; both requests deduplicate."""
    capture = _make_capture()
    viewer = MagicMock()
    capture._viewer = viewer

    capture.set_visible_worlds([0])
    capture.set_visible_worlds(None)
    capture.set_visible_worlds(None)
    capture.set_visible_worlds([])
    capture.set_visible_worlds([])

    assert viewer.set_visible_worlds.call_args_list == [
        call([0]),
        call(None),
        call([]),
    ]


def test_set_world_offsets_defers_until_viewer_initialization(monkeypatch: pytest.MonkeyPatch):
    """World spacing is stored before ViewerGL exists and applied after set_model."""
    capture = _make_capture()

    capture.set_world_offsets((2.0, 2.0, 0.0))

    viewer, model = _initialize_with_fake_viewer(capture, monkeypatch)

    viewer.set_world_offsets.assert_called_once_with((2.0, 2.0, 0.0))
    assert viewer.method_calls.index(call.set_model(model)) < viewer.method_calls.index(
        call.set_world_offsets((2.0, 2.0, 0.0))
    )


def test_set_world_offsets_deduplicates_live_viewer_calls():
    """Equal world spacing does not trigger redundant ViewerGL updates."""
    capture = _make_capture()
    viewer = MagicMock()
    capture._viewer = viewer

    capture.set_world_offsets((2.0, 2.0, 0.0))
    capture.set_world_offsets((2.0, 2.0, 0.0))
    capture.set_world_offsets((1.0, 1.0, 0.0))
    capture.set_world_offsets((1.0, 1.0, 0.0))

    assert viewer.set_world_offsets.call_args_list == [
        call((2.0, 2.0, 0.0)),
        call((1.0, 1.0, 0.0)),
    ]


def test_set_world_offsets_rejects_wrong_length():
    """World spacing must provide exactly three axis values."""
    capture = _make_capture()

    with pytest.raises(ValueError, match="three values"):
        capture.set_world_offsets((1.0, 2.0))


def test_frame_overlay_callback_runs_between_state_and_end_frame(monkeypatch: pytest.MonkeyPatch):
    """Viewer-side overlays are logged after state and before the captured frame ends."""
    capture = _make_capture()
    viewer, _ = _initialize_with_fake_viewer(capture, monkeypatch)
    state = object()
    sys.modules["isaaclab_newton.physics"].NewtonManager.get_state = MagicMock(return_value=state)

    sim_module = ModuleType("isaaclab.sim")
    sim_module.SimulationContext = SimpleNamespace(
        instance=MagicMock(return_value=SimpleNamespace(get_physics_dt=MagicMock(return_value=0.01)))
    )
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)
    overlay_calls = []

    def _render_overlay(target_viewer):
        assert target_viewer.log_state.called
        assert not target_viewer.end_frame.called
        overlay_calls.append(target_viewer)

    capture.set_frame_overlay_callback(_render_overlay)
    capture.render_rgb_array()

    assert overlay_calls == [viewer]
    viewer.log_state.assert_called_once_with(state)
    viewer.end_frame.assert_called_once_with()


def test_frame_overlay_callback_failure_still_ends_frame(monkeypatch: pytest.MonkeyPatch):
    """A failing overlay callback propagates its error after closing the viewer frame."""
    capture = _make_capture()
    viewer, _ = _initialize_with_fake_viewer(capture, monkeypatch)
    state = object()
    sys.modules["isaaclab_newton.physics"].NewtonManager.get_state = MagicMock(return_value=state)

    sim_module = ModuleType("isaaclab.sim")
    sim_module.SimulationContext = SimpleNamespace(
        instance=MagicMock(return_value=SimpleNamespace(get_physics_dt=MagicMock(return_value=0.01)))
    )
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim_module)

    callback = MagicMock(side_effect=RuntimeError("overlay failed"))
    capture.set_frame_overlay_callback(callback)

    with pytest.raises(RuntimeError, match="overlay failed"):
        capture.render_rgb_array()

    callback.assert_called_once_with(viewer)
    viewer.end_frame.assert_called_once_with()
    viewer.get_frame.assert_not_called()
