# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton viewer adapter helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import warp as wp
from isaaclab_visualizers.newton import NewtonVisualizer, NewtonVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
from isaaclab_visualizers.newton_adapter import (
    VISUALIZER_INFINITE_PLANE_SIZE,
    apply_viewer_visible_worlds,
    expand_infinite_plane_scale,
    log_geo_with_expanded_plane_scale,
    resolve_visible_env_indices,
)


def test_expand_infinite_plane_scale_expands_non_positive_extents():
    assert expand_infinite_plane_scale((0.0, 0.0, 1.0, 0.0)) == (
        VISUALIZER_INFINITE_PLANE_SIZE,
        VISUALIZER_INFINITE_PLANE_SIZE,
        1.0,
        0.0,
    )
    assert expand_infinite_plane_scale((-1.0, 25.0)) == (
        VISUALIZER_INFINITE_PLANE_SIZE,
        25.0,
    )
    assert expand_infinite_plane_scale((25.0, 0.0)) == (
        25.0,
        VISUALIZER_INFINITE_PLANE_SIZE,
    )


def test_expand_infinite_plane_scale_preserves_finite_extents():
    assert expand_infinite_plane_scale((100.0, 50.0, 1.0)) == (100.0, 50.0, 1.0)


def test_log_geo_with_expanded_plane_scale_delegates_with_adjusted_plane_scale():
    calls = []

    def _log_geo(*args):
        calls.append(args)
        return "logged"

    assert log_geo_with_expanded_plane_scale(_log_geo, 1, "ground", 1, (0.0, 25.0), 0.0, True) == "logged"
    assert calls == [("ground", 1, (VISUALIZER_INFINITE_PLANE_SIZE, 25.0), 0.0, True, None, False)]


def test_log_geo_with_expanded_plane_scale_preserves_non_plane_scale():
    calls = []

    def _log_geo(*args):
        calls.append(args)

    log_geo_with_expanded_plane_scale(_log_geo, 1, "box", 2, (0.0, 25.0), 0.0, True, hidden=True)
    assert calls == [("box", 2, (0.0, 25.0), 0.0, True, None, True)]


def test_resolve_visible_env_indices_truncates_explicit_list():
    assert resolve_visible_env_indices([1, 3, 5], 2, 10) == [1, 3]
    assert resolve_visible_env_indices([1, 3], 1, 10) == [1]


def test_resolve_visible_env_indices_explicit_full_list_when_no_cap():
    assert resolve_visible_env_indices([1, 3], None, 10) == [1, 3]


def test_resolve_visible_env_indices_cap_when_no_filter():
    # When _compute_visualized_env_ids is None, cap is max_visible_envs.
    assert resolve_visible_env_indices(None, 3, 10) == [0, 1, 2]


def test_resolve_visible_env_indices_all_when_no_cap():
    assert resolve_visible_env_indices(None, None, 10) is None


def test_resolve_visible_env_indices_num_envs_zero_falls_through_like_newton():
    assert resolve_visible_env_indices(None, 5, 0) is None


def test_apply_viewer_visible_worlds_delegates_to_resolved():
    calls: list = []

    class _V:
        def set_visible_worlds(self, worlds):
            calls.append(worlds)

    apply_viewer_visible_worlds(_V(), env_ids=None, max_visible_envs=2, num_envs=5)
    assert calls == [[0, 1]]

    apply_viewer_visible_worlds(_V(), env_ids=[2], max_visible_envs=99, num_envs=5)
    assert calls[-1] == [2]

    apply_viewer_visible_worlds(_V(), env_ids=None, max_visible_envs=None, num_envs=3)
    assert calls[-1] is None


def test_newton_visualizer_cfg_exposes_particle_options():
    cfg = NewtonVisualizerCfg(show_particles=True, particle_color=(0.1, 0.2, 0.3))

    assert cfg.show_particles is True
    assert cfg.particle_color == (0.1, 0.2, 0.3)


def test_newton_visualizer_set_camera_view_updates_cfg_without_viewer():
    visualizer = NewtonVisualizer(NewtonVisualizerCfg())

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)
    assert visualizer._resolve_initial_camera_pose() == ((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))


def test_newton_visualizer_set_camera_view_updates_active_viewer():
    """NewtonVisualizer should honor SimulationContext camera updates."""

    class _FakeCamera:
        def __init__(self):
            self.pos = None
            self.look_at_calls = []

        def look_at(self, target):
            self.look_at_calls.append(tuple(target))

    class _FakeViewer:
        def __init__(self):
            self.camera = _FakeCamera()

    viewer = _FakeViewer()
    visualizer = NewtonVisualizer(NewtonVisualizerCfg())
    visualizer._viewer = viewer

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert (viewer.camera.pos.x, viewer.camera.pos.y, viewer.camera.pos.z) == (1.0, 2.0, 3.0)
    assert viewer.camera.look_at_calls == [(0.0, 0.0, 1.0)]
    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)


def test_newton_viewer_particle_color_override(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.particle_color = (0.1, 0.2, 0.3)
    viewer._particle_color_buffer = None
    viewer._particle_color_buffer_count = 0
    viewer._particle_color_buffer_value = None
    points = wp.zeros(4, dtype=wp.vec3, device="cpu")
    calls = []

    def _log_points(self, name, points, radii=None, colors=None, hidden=False):
        calls.append((name, points, radii, colors, hidden))

    monkeypatch.setattr(ViewerGL, "log_points", _log_points)

    viewer.log_points("/model/particles", points, colors=None)

    name, _, _, colors, hidden = calls[-1]
    assert name == "/model/particles"
    assert hidden is False
    assert isinstance(colors, wp.array)
    assert colors.shape[0] == 4
    np.testing.assert_allclose(colors.numpy()[0], np.array([0.1, 0.2, 0.3], dtype=np.float32), rtol=1.0e-6)


def test_newton_viewer_particle_color_override_leaves_other_points_unchanged(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.particle_color = (0.1, 0.2, 0.3)
    points = wp.zeros(1, dtype=wp.vec3, device="cpu")
    original_colors = object()
    calls = []

    def _log_points(self, name, points, radii=None, colors=None, hidden=False):
        calls.append((name, colors))

    monkeypatch.setattr(ViewerGL, "log_points", _log_points)

    viewer.log_points("/debug/points", points, colors=original_colors)

    assert calls[-1] == ("/debug/points", original_colors)


class _BodyQ:
    shape = (1,)


class _Viewer:
    _update_frequency = 1

    def __init__(self):
        self.device = "cpu"
        self.show_contacts = False
        self.logged_state = None
        self.logged_contacts = None
        self.logged_arrows = None

    def is_paused(self):
        return False

    def begin_frame(self, _time):
        pass

    def log_state(self, state):
        self.logged_state = state

    def log_contacts(self, contacts, state):
        self.logged_contacts = (contacts, state)

    def log_arrows(self, name, starts, ends, colors):
        self.logged_arrows = (name, starts, ends, colors)

    def end_frame(self):
        pass


class _Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class _ContactSensorData:
    def __init__(self, net_forces_w, pos_w):
        self.net_forces_w = _Proxy(net_forces_w)
        self.pos_w = _Proxy(pos_w)
        self.contact_pos_w = None
        self.force_matrix_w = None


class _ContactSensor:
    def __init__(self, net_forces_w, pos_w, force_threshold=1.0):
        self.cfg = SimpleNamespace(force_threshold=force_threshold)
        self.data = _ContactSensorData(net_forces_w, pos_w)


class _SceneDataProvider:
    def __init__(self, contact_sensors=None):
        self._contact_sensors = contact_sensors or {}

    def get_contact_sensors(self):
        return self._contact_sensors


def _make_newton_visualizer(viewer, scene_data_provider=None):
    visualizer = NewtonVisualizer.__new__(NewtonVisualizer)
    visualizer.cfg = NewtonVisualizerCfg(enable_markers=False)
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._sim_time = 0.0
    visualizer._step_counter = 0
    visualizer._viewer = viewer
    visualizer._state = None
    visualizer._scene_data_provider = scene_data_provider
    visualizer._resolved_visible_env_ids = None
    visualizer._log_camera_sensor_image = lambda: None
    return visualizer


def test_newton_visualizer_logs_native_contacts_when_available(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    state = SimpleNamespace(body_q=_BodyQ())
    contacts = object()
    viewer = _Viewer()

    monkeypatch.setattr(NewtonManager, "get_state", lambda _scene_data_provider=None: state)
    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: contacts)
    monkeypatch.setattr(NewtonManager, "get_num_envs", lambda: 1)

    _make_newton_visualizer(viewer).step(0.1)

    assert viewer.logged_state is state
    assert viewer.logged_contacts == (contacts, state)


def test_newton_visualizer_contact_sensor_fallback_obeys_show_contacts(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    state = SimpleNamespace(body_q=_BodyQ())
    viewer = _Viewer()
    sensor = _ContactSensor(
        net_forces_w=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]]], dtype=torch.float32),
        pos_w=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype=torch.float32),
        force_threshold=1.0,
    )
    scene_data_provider = _SceneDataProvider({"contact_forces": sensor})

    monkeypatch.setattr(NewtonManager, "get_state", lambda _scene_data_provider=None: state)
    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: None)
    monkeypatch.setattr(NewtonManager, "get_num_envs", lambda: 1)

    visualizer = _make_newton_visualizer(viewer, scene_data_provider)
    visualizer.step(0.1)
    assert viewer.logged_arrows == ("/contacts", None, None, None)

    viewer.show_contacts = True
    visualizer.step(0.1)

    name, starts, ends, colors = viewer.logged_arrows
    assert name == "/contacts"
    assert len(starts) == 1
    assert len(ends) == 1
    assert colors == (0.0, 1.0, 0.0)
    assert torch.allclose(torch.tensor(starts.numpy()[0]), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(torch.tensor(ends.numpy()[0]), torch.tensor([1.0, 2.0, 3.1]))
