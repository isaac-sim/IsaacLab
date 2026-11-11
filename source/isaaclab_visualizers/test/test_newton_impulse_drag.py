# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Impulse Drag, which aims pushes while the simulation is paused."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("newton")
import warp as wp
from isaaclab_visualizers.newton.newton_impulse_drag import (
    IMPULSE_GAIN,
    MAX_IMPULSE_SPEED,
    MAX_IMPULSES,
    ImpulseDrag,
    impulse_color,
    make_impulse,
)


def test_the_arrow_color_runs_from_green_through_yellow_to_red():
    assert impulse_color(0.0)[1] == 1.0 and impulse_color(0.0)[0] == 0.0
    assert impulse_color(0.5)[:2] == (1.0, 1.0)
    assert impulse_color(1.0)[0] == 1.0 and impulse_color(1.0)[1] == 0.0
    assert impulse_color(7.0) == impulse_color(1.0) and impulse_color(-1.0) == impulse_color(0.0)


def test_an_impulse_gives_the_body_a_speed_in_proportion_to_the_arrow_up_to_a_limit():
    short = make_impulse(3, np.zeros(3), np.array([0.0, 0.0, 0.2]), mass=2.0)
    assert short.vector == pytest.approx([0.0, 0.0, IMPULSE_GAIN * 0.2 * 2.0])
    assert short.fraction == pytest.approx(IMPULSE_GAIN * 0.2 / MAX_IMPULSE_SPEED)

    long = make_impulse(3, np.zeros(3), np.array([10.0, 0.0, 0.0]), mass=2.0)
    assert long.vector == pytest.approx([MAX_IMPULSE_SPEED * 2.0, 0.0, 0.0]) and long.fraction == 1.0


def test_an_arrow_of_no_length_makes_no_impulse():
    drag = ImpulseDrag("cpu", 1.0 / 120.0)

    assert not drag.add(make_impulse(0, np.zeros(3), np.zeros(3), mass=1.0))
    assert drag.entries == []


def _state(count=2):
    transforms = np.zeros((count, 7), dtype=np.float32)
    transforms[:, 6] = 1.0
    return SimpleNamespace(
        body_q=wp.array(transforms, dtype=wp.transform, device="cpu"),
        body_f=wp.zeros(count, dtype=wp.spatial_vector, device="cpu"),
    )


def test_queued_impulses_are_applied_once_over_one_substep_and_may_be_several():
    dt = 1.0 / 120.0
    drag = ImpulseDrag("cpu", dt)
    com = wp.zeros(3, dtype=wp.vec3, device="cpu")
    state = _state(3)
    drag.add(make_impulse(0, np.zeros(3), np.array([0.1, 0.0, 0.0]), mass=1.0))
    drag.add(make_impulse(1, np.array([0.0, 0.5, 0.0]), np.array([0.0, 0.0, 0.1]), mass=1.0))
    drag.add(make_impulse(2, np.zeros(3), np.array([0.0, 0.1, 0.0]), mass=3.0))

    drag.apply(state, com)
    forces = state.body_f.numpy()

    assert forces[0][:3] * dt == pytest.approx([IMPULSE_GAIN * 0.1, 0.0, 0.0])
    assert forces[1][:3] * dt == pytest.approx([0.0, 0.0, IMPULSE_GAIN * 0.1])
    assert forces[1][3:] * dt == pytest.approx([IMPULSE_GAIN * 0.1 * 0.5, 0.0, 0.0])  # It acts off center.
    assert forces[2][:3] * dt == pytest.approx([0.0, IMPULSE_GAIN * 0.1 * 3.0, 0.0])

    second = _state(3)
    drag.apply(second, com)
    assert not second.body_f.numpy().any()  # The next substep finds nothing to apply.
    drag.forget_applied()
    assert drag.entries == []


def test_impulses_wait_for_the_simulation_and_the_queue_has_a_limit():
    drag = ImpulseDrag("cpu", 1.0 / 120.0)
    for body in range(MAX_IMPULSES):
        assert drag.add(make_impulse(body, np.zeros(3), np.array([0.1, 0.0, 0.0]), mass=1.0))

    drag.forget_applied()  # Nothing ran yet, so nothing is dropped.
    assert len(drag.entries) == MAX_IMPULSES and drag.full
    assert not drag.add(make_impulse(MAX_IMPULSES, np.zeros(3), np.array([0.1, 0.0, 0.0]), mass=1.0))

    drag.clear()
    assert drag.entries == [] and not drag.full


def test_a_new_impulse_on_a_body_replaces_the_one_waiting_on_it():
    drag = ImpulseDrag("cpu", 1.0 / 120.0)
    com = wp.zeros(2, dtype=wp.vec3, device="cpu")
    drag.add(make_impulse(0, np.zeros(3), np.array([0.1, 0.0, 0.0]), mass=1.0))
    drag.add(make_impulse(1, np.zeros(3), np.array([0.0, 0.1, 0.0]), mass=1.0))
    drag.add(make_impulse(0, np.zeros(3), np.array([0.0, 0.0, 0.2]), mass=1.0))

    assert [entry.body for entry in drag.entries] == [0, 1]
    state = _state()
    drag.apply(state, com)
    forces = state.body_f.numpy()

    assert forces[0][0] == pytest.approx(0.0) and forces[0][2] > 0.0  # Only the new one acts on body 0.
    assert forces[1][1] > 0.0


def test_space_pauses_the_simulation_and_shift_space_pauses_rendering():
    from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
    from pyglet.window import key

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.gui = SimpleNamespace(is_keyboard_capturing=lambda: False)
    viewer.renderer = SimpleNamespace(close=lambda: None)
    viewer._paused, viewer._paused_training = False, False

    viewer.on_key_press(key.SPACE, 0)
    assert viewer._paused_training and not viewer._paused

    viewer.on_key_press(key.SPACE, key.MOD_SHIFT)
    assert viewer._paused and viewer._paused_training

    viewer.on_key_press(key.SPACE, 0)
    assert viewer._paused and not viewer._paused_training

    viewer.gui = SimpleNamespace(
        is_keyboard_capturing=lambda: True, handle_key_press=lambda *args, **kwargs: None
    )  # A text field has the keyboard.
    viewer.on_key_press(key.SPACE, 0)
    assert not viewer._paused_training


def test_the_drag_arrow_reaches_the_ovrtx_view_as_a_wide_colored_arrow(monkeypatch):
    from isaaclab_visualizers.newton import newton_visualizer
    from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL

    monkeypatch.setattr(newton_visualizer.ViewerGL, "log_arrows", lambda self, *args, **kwargs: None)
    recorded = []
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._overlay_capture = SimpleNamespace(record=lambda *args, **kwargs: recorded.append((args, kwargs)))
    starts = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device="cpu")
    ends = wp.array([[0.5, 0.0, 0.0]], dtype=wp.vec3, device="cpu")

    viewer.log_lines("picking_line", starts, ends, (0.0, 1.0, 1.0))

    (name, _, _, colors, width, _), kwargs = recorded[0]
    assert name == "picking_arrow" and kwargs == {"arrow": True} and width == NewtonViewerGL.PICKING_ARROW_WIDTH
    assert colors == impulse_color(IMPULSE_GAIN * 0.5 / MAX_IMPULSE_SPEED)


def _rtx_visualizer(picking: bool, queued: bool = False):
    from isaaclab_visualizers.newton.newton_visualizer import NewtonGLVisualizer

    viz = NewtonGLVisualizer.__new__(NewtonGLVisualizer)
    camera = SimpleNamespace(cfg=SimpleNamespace(prim_path="/World/viewer_camera_0", renderer_cfg="ovrtx"))
    view = SimpleNamespace(camera=camera)
    viz._image_views, viz.current_image_view = [view], view
    viz._sim = SimpleNamespace(get_or_create_backend=lambda cfg: cfg)
    calls = []
    viz._overlay_publisher = SimpleNamespace(
        hide_all=lambda renderers: calls.append("hide_all"), publish=lambda renderers: calls.append("publish")
    )
    viz._update_state = lambda: SimpleNamespace(body_q=None)
    viz._viewer = SimpleNamespace(
        show_joints=False,
        show_contacts=False,
        show_springs=False,
        show_com=False,
        show_inertia_boxes=False,
        show_contact_disks=False,
        picking_enabled=True,
        picking=SimpleNamespace(is_picking=lambda: picking),
        impulse_drag=SimpleNamespace(entries=[object()] if queued else []),
        _impulse_arrows_shown=False,
        is_paused=lambda: False,
        _render_picking_line=lambda state: calls.append("picking_line"),
    )
    return viz, calls


def test_the_drag_arrow_is_published_to_ovrtx_even_when_no_overlay_is_enabled():
    viz, calls = _rtx_visualizer(picking=True)

    viz._log_rtx_overlays()

    assert calls == ["picking_line", "publish"]


def test_queued_impulse_arrows_are_published_while_the_mouse_is_up():
    viz, calls = _rtx_visualizer(picking=False, queued=True)

    viz._log_rtx_overlays()

    assert calls == ["picking_line", "publish"]


def test_nothing_is_published_to_ovrtx_when_nothing_is_being_dragged():
    viz, calls = _rtx_visualizer(picking=False)

    viz._log_rtx_overlays()

    assert calls == ["hide_all"]


def _paused_rtx(paused: bool, timestamp: float = 5.0):
    viz, _ = _rtx_visualizer(picking=False)
    renders = []
    camera = viz.current_image_view.camera
    camera._is_outdated = SimpleNamespace(fill_=lambda value: renders.append("outdated"))
    camera.update = lambda dt, force_recompute=False: renders.append(("update", dt, force_recompute))
    viz.current_image_view.frame = SimpleNamespace(timestamp=timestamp)
    viz.current_image_view.invalidate = lambda: renders.append("invalidate")
    viz._viewer.is_training_paused = lambda: paused
    return viz, renders


def test_a_paused_ovrtx_view_is_rendered_again_while_an_arrow_changes_and_once_more_after():
    viz, renders = _paused_rtx(paused=True)

    viz._refresh_paused_rtx_view(True)
    assert renders == ["outdated", ("update", 0.0, True), "invalidate"]

    renders.clear()
    viz._refresh_paused_rtx_view(False)  # The arrows are gone; this render removes them from the image.
    assert len(renders) == 3

    renders.clear()
    viz._refresh_paused_rtx_view(False)
    assert renders == []


def test_a_paused_ovrtx_view_follows_the_camera_and_a_running_one_is_left_alone():
    viz, renders = _paused_rtx(paused=True, timestamp=-1.0)
    viz._refresh_paused_rtx_view(False)
    assert len(renders) == 3  # Navigating invalidated the image.

    viz, renders = _paused_rtx(paused=False, timestamp=-1.0)
    viz._refresh_paused_rtx_view(True)
    assert renders == []  # The simulation renders it.


def test_the_solver_substep_survives_a_model_reset(monkeypatch):
    from isaaclab_visualizers.newton import newton_visualizer
    from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL

    monkeypatch.setattr(newton_visualizer.ViewerGL, "set_model", lambda self, model, *args, **kwargs: None)
    viewer = object.__new__(NewtonViewerGL)
    model = SimpleNamespace(device="cpu")

    viewer.set_model(model)
    viewer.set_impulse_substep_dt(1.0 / 240.0)
    viewer.set_model(model)  # A reset replaces the Impulse Drag, which must keep the solver's substep.

    assert viewer.impulse_drag._inverse_dt.numpy()[0] == pytest.approx(240.0)
