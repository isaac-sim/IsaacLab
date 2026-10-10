# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for Newton viewer adapter helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.newton import (
    NewtonGLVisualizer,
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizer,
    NewtonRTXVisualizerCfg,
)
from isaaclab_visualizers.newton import newton_visualization_markers as newton_markers
from isaaclab_visualizers.newton import newton_visualizer as newton_visualizer_module
from isaaclab_visualizers.newton.newton_visualizer import NewtonViewerGL
from isaaclab_visualizers.newton_adapter import (
    VISUALIZER_INFINITE_PLANE_SIZE,
    expand_infinite_plane_scale,
    log_geo_with_expanded_plane_scale,
)

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.utils.camera_colorizer import CameraFrameColorizer
from isaaclab.envs.utils.camera_view import resolve_camera_sources
from isaaclab.sim import SimulationContext
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import instantiate, validate
from isaaclab.utils.warp import ProxyArray
from isaaclab.visualizers import PerspectiveCameraCfg, SceneCameraCfg, WindowCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (
            (0.0, 0.0, 1.0, 0.0),
            (VISUALIZER_INFINITE_PLANE_SIZE, VISUALIZER_INFINITE_PLANE_SIZE, 1.0, 0.0),
        ),
        ((-1.0, 25.0), (VISUALIZER_INFINITE_PLANE_SIZE, 25.0)),
        ((25.0, 0.0), (25.0, VISUALIZER_INFINITE_PLANE_SIZE)),
        ((100.0, 50.0, 1.0), (100.0, 50.0, 1.0)),
    ],
)
def test_expand_infinite_plane_scale(scale, expected):
    assert expand_infinite_plane_scale(scale) == expected


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


def test_newton_visualizer_log_mesh_keeps_latest_submission_per_name():
    viewer = Mock()
    visualizer = _make_newton_visualizer(None)
    visualizer._viewer = viewer
    points_0 = wp.zeros(3, dtype=wp.vec3)
    points_1 = wp.zeros(6, dtype=wp.vec3)
    indices = wp.zeros(3, dtype=wp.int32)

    visualizer.log_mesh("/surface", points_0, indices, dynamic=True)
    visualizer.log_mesh("/surface", points_1, indices, dynamic=True)
    visualizer._log_pending_meshes()

    viewer.log_mesh.assert_called_once()
    assert viewer.log_mesh.call_args.args[:3] == ("/surface", points_1, indices)


def test_newton_visualizer_log_mesh_requires_initialized_viewer():
    visualizer = _make_newton_visualizer(None)
    points = wp.zeros(3, dtype=wp.vec3)
    indices = wp.zeros(3, dtype=wp.int32)

    with pytest.raises(RuntimeError, match="must be initialized"):
        visualizer.log_mesh("/surface", points, indices)


class _MarkerRegistry:
    def __init__(self) -> None:
        self.groups: dict[str, object] = {}

    def set_group(self, group_id: str, marker) -> None:
        self.groups[group_id] = marker

    def remove_group(self, group_id: str) -> None:
        self.groups.pop(group_id)

    def get_groups(self) -> dict[str, object]:
        return self.groups


class _FakeSimulationContext:
    current: object | None = None

    @classmethod
    def instance(cls):
        return cls.current


@pytest.fixture
def marker_registry(monkeypatch: pytest.MonkeyPatch):
    """Marker registry that ``NewtonVisualizationMarkers`` finds through a fake simulation context."""
    registry = _MarkerRegistry()
    monkeypatch.setattr(newton_markers.sim_utils, "SimulationContext", _FakeSimulationContext)
    _FakeSimulationContext.current = SimpleNamespace(vis_marker_registry=registry)
    yield registry
    _FakeSimulationContext.current = None


def test_newton_marker_registry_lifecycle(marker_registry: _MarkerRegistry):
    """Construction caches the registry; close survives context teardown and is idempotent."""
    marker = newton_markers.NewtonVisualizationMarkers(
        newton_markers.VisualizationMarkersCfg(prim_path="/Visuals/test", markers={}), visible=False
    )
    assert marker_registry.groups == {marker.group_id: marker}

    # the context is torn down before markers close during interpreter shutdown
    _FakeSimulationContext.current = None

    marker.close()
    marker.close()

    assert marker._registry is None
    assert marker_registry.groups == {}


def test_importing_newton_visualizer_lets_pyglet_resolve_a_screen_without_monitors():
    """On a monitor-free X server, importing the module must still let pyglet resolve a default screen."""
    pytest.importorskip("pyglet.display.xlib")
    code = textwrap.dedent(
        """
        from types import SimpleNamespace

        from pyglet.display import xlib

        xlib._have_xrandr = True
        import isaaclab_visualizers.newton.newton_visualizer  # noqa: F401

        screen = object.__new__(xlib.XlibScreen)
        display = SimpleNamespace(get_screens=lambda: [screen], _screens=[screen])
        assert xlib.XlibDisplay.get_default_screen(display) is screen
        """
    )
    # a fresh interpreter: the workaround runs at import, and only when DISPLAY is set
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "DISPLAY": ":99"},
        capture_output=True,
        text=True,
        timeout=300,
    )

    assert result.returncode == 0, result.stderr[-2000:]


def test_newton_visualizer_set_camera_view_updates_cfg_without_viewer():
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)


def test_newton_visualizer_set_camera_view_updates_active_viewer():
    """NewtonGLVisualizer should honor SimulationContext camera updates."""

    from newton._src.viewer.camera import Camera

    viewer = object.__new__(NewtonViewerGL)
    viewer.camera = Camera(width=64, height=64, up_axis="Z")
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    visualizer._viewer = viewer

    visualizer.set_camera_view((1, 2, 3), (0, 0, 1))

    assert (viewer.camera.pos.x, viewer.camera.pos.y, viewer.camera.pos.z) == (1.0, 2.0, 3.0)
    expected = np.array([-1.0, -2.0, -2.0]) / 3.0
    np.testing.assert_allclose(viewer.camera.get_front(), expected, atol=1e-6)
    assert visualizer.cfg.eye == (1.0, 2.0, 3.0)
    assert visualizer.cfg.lookat == (0.0, 0.0, 1.0)


def test_visualizers_borrow_scene_camera_outputs(monkeypatch):
    """Resolved sources preserve selection and sensor ownership without visualizer scene discovery."""
    monkeypatch.setattr(SimulationContext, "instance", Mock(side_effect=AssertionError("No global scene lookup")))
    pixels = torch.arange(4, dtype=torch.uint8).reshape(4, 1, 1, 1).expand(4, 2, 3, 3).clone()
    camera = SimpleNamespace(
        _view=None,
        cfg=SimpleNamespace(prim_path="/Scenes/world_[^/]+/Camera", data_types=["rgba"]),
        data=SimpleNamespace(
            output={
                "rgba": ProxyArray(wp.from_torch(torch.cat((pixels, torch.full_like(pixels[..., :1], 255)), dim=-1)))
            }
        ),
        close=Mock(),
        update=Mock(),
    )
    provider = SimpleNamespace(
        num_envs=4, get_camera_sensors=Mock(side_effect=AssertionError("Sources must already be bound"))
    )
    sim = Mock(stage=None, get_scene_data_provider=Mock(return_value=provider))
    camera_sensors = {"camera": camera}
    viewers = []
    for ids in ([0, 2], [1, 3]):
        cfg = NewtonGLVisualizerCfg(streaming_envs=ids, cameras=[SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera")])
        visualizer = instantiate(cfg)
        cameras = resolve_camera_sources(cfg, camera_sensors, env_template="/Scenes/world_{}")
        BaseVisualizer.initialize(visualizer, sim, cameras=cameras)
        assert not hasattr(visualizer, "_clone_plan")
        assert not hasattr(visualizer, "_resolve_camera_pose_from_usd_path")
        assert not hasattr(visualizer, "_streaming_params")
        assert not hasattr(visualizer, "_resolved_visible_env_ids")
        assert not hasattr(visualizer, "_resolve_initial_camera_pose")
        visualizer._setup_streaming_view(4)
        assert visualizer._camera_sensor is camera
        image = visualizer.render_tiled_rgb_array()
        np.testing.assert_array_equal(np.unique(image), ids)
        assert visualizer.render_tiled_rgb_array() is image
        viewers.append(visualizer)

    # A new step or reset refreshes the composite, not the sensor's lifetime.
    frame = viewers[0].render_tiled_rgba_array()
    camera.data.output["rgba"].torch[..., :3].add_(10)
    viewers[0]._sim_time += 0.1
    np.testing.assert_array_equal(np.unique(viewers[0].render_tiled_rgb_array()), [10, 12])
    assert viewers[0].render_tiled_rgba_array() is frame
    viewers[1].reset(soft=True)
    np.testing.assert_array_equal(np.unique(viewers[1].render_tiled_rgb_array()), [11, 13])
    camera.update.assert_not_called()
    camera.close.assert_not_called()
    SimulationContext.instance.assert_not_called()

    visualizer = viewers[0]
    cfg = visualizer.cfg
    rgba = camera.data.output["rgba"]
    camera.data.output["rgba"] = ProxyArray(rgba.warp[:, :, :, :1])
    visualizer._sim_time += 0.1
    with pytest.raises(ValueError, match="channels"):
        visualizer.render_tiled_rgba_array()
    camera.data.output["rgba"] = rgba

    # Channel and color-range changes rebuild the display buffers at their owner.
    camera.cfg.data_types.append("depth")
    camera.data.output["depth"] = ProxyArray(wp.full((4, 2, 3, 1), 2.0, dtype=wp.float32, device=rgba.warp.device))
    cfg.streaming_gt_types = ("depth",)
    cfg.streaming_depth_min = 1.0
    for depth_max in (5.0, 3.0):
        cfg.streaming_depth_max = depth_max
        image = visualizer.render_tiled_rgb_array()
        expected = CameraFrameColorizer.colorize(np.array([[[2.0]]]), "depth", depth_min=1.0, depth_max=depth_max)
        np.testing.assert_array_equal(image, np.broadcast_to(expected, image.shape))
    camera.cfg.data_types.remove("depth")
    del camera.data.output["depth"]
    cfg.streaming_gt_types = ("rgb",)

    cfg.cameras = [SceneCameraCfg(prim_path="/Missing/Camera")]
    with pytest.raises(ValueError, match="No scene Camera matches"):
        resolve_camera_sources(cfg, camera_sensors)
    cfg.cameras = None
    assert resolve_camera_sources(cfg, camera_sensors)[1:] == [camera]
    cfg.streaming_gt_types = ("depth",)
    assert len(resolve_camera_sources(cfg, camera_sensors)) == 1
    cfg.cameras = [SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera")]
    with pytest.raises(KeyError, match="No sensor output"):
        resolve_camera_sources(cfg, camera_sensors, env_template="/Scenes/world_{}")
    cfg.streaming_gt_types = ("optical_flow",)
    with pytest.raises(ValueError, match="optical_flow"):
        resolve_camera_sources(cfg, camera_sensors)

    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    camera._view = SimpleNamespace(
        prims=[UsdGeom.Camera.Define(stage, f"/Scenes/world_{i}/Camera").GetPrim() for i in range(4)]
    )
    cfg.streaming_gt_types = ("rgb",)
    for path in ("/Scenes/world_2/Camera", "/Scenes/world_.*/Camera"):
        cfg.cameras = [SceneCameraCfg(prim_path=path)]
        assert resolve_camera_sources(cfg, camera_sensors) == [camera]
    cfg.cameras = None
    cfg.streaming_sensor_prim_path = "/Scenes/world_2/Camera"
    assert resolve_camera_sources(cfg, camera_sensors)[0] is camera


def test_newton_visualizer_render_rgb_array_returns_viewer_frame():
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    viewer = SimpleNamespace(get_frame=lambda: SimpleNamespace(numpy=lambda: frame))
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())
    visualizer._viewer = viewer

    assert visualizer.render_rgb_array() is frame


def test_newton_visualizer_render_rgb_array_requires_initialized_viewer():
    visualizer = NewtonGLVisualizer(NewtonGLVisualizerCfg())

    with pytest.raises(RuntimeError, match="must be initialized"):
        visualizer.render_rgb_array()


def test_newton_viewer_camera_speed_boost_when_shift_held(monkeypatch):
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._camera_speed = 4.0

    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: True)
    assert viewer.camera_speed == pytest.approx(8.0)

    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: False)
    assert viewer.camera_speed == pytest.approx(4.0)


def test_newton_viewer_camera_speed_setter_validates(monkeypatch):
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    monkeypatch.setattr(NewtonViewerGL, "is_key_down", lambda self, key: False)

    viewer.camera_speed = 6.0
    assert viewer._camera_speed == pytest.approx(6.0)

    with pytest.raises(ValueError, match="camera_speed must be finite and nonnegative"):
        viewer.camera_speed = -1.0


class _FakeTrainingControlsImgui:
    """Minimal imgui double that drives ``_render_training_controls`` by label."""

    def __init__(self, clicked_label: str | None = None):
        self._clicked_label = clicked_label

    def button(self, label):
        return label == self._clicked_label

    def text(self, _text):
        pass

    def slider_float(self, _label, value, _min_value, _max_value, _format):
        return False, value

    def is_item_hovered(self):
        return False

    def set_tooltip(self, _text):
        pass


def test_newton_gl_viewer_rendering_pause_state_stays_in_sync_with_space_key():
    """Space toggles ``_paused`` directly (Newton's own key handler); the "Pause Rendering"
    button and ``is_rendering_paused()`` must reflect that instead of a separately tracked flag,
    or the UI desyncs from the actual paused state that gates rendering.
    """
    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._paused = False
    viewer._paused_training = False
    viewer._reset_requested = False
    viewer.window_cfg = WindowCfg()

    assert viewer.is_rendering_paused() is False

    # Simulate Newton's own Space key handler (newton/_src/viewer/viewer_gui.py), which
    # toggles ``_paused`` directly and bypasses the Isaac Lab "Pause Rendering" button.
    viewer._paused = not viewer._paused

    assert viewer.is_rendering_paused() is True
    viewer._render_training_controls(_FakeTrainingControlsImgui())  # must not raise: no click

    # The button must read the post-Space state and toggle it back correctly.
    resume_click = _FakeTrainingControlsImgui(clicked_label="Resume Rendering")
    viewer._render_training_controls(resume_click)
    assert viewer.is_rendering_paused() is False


def test_newton_viewer_particle_color_override(monkeypatch):
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer.device = "cpu"
    viewer.objects = {}
    viewer.particle_color = (0.1, 0.2, 0.3)
    viewer._particle_color_buffer = None
    viewer._particle_color_buffer_value = None
    points = wp.zeros(4, dtype=wp.vec3, device="cpu")
    calls = []
    monkeypatch.setattr(ViewerGL, "log_points", lambda self, *args: calls.append(args))

    viewer.log_points("/model/particles", points)
    name, _, _, colors, hidden = calls[-1]
    assert name == "/model/particles" and hidden is False
    assert colors.shape == (4,)
    np.testing.assert_allclose(colors.numpy(), np.tile([0.1, 0.2, 0.3], (4, 1)), rtol=1.0e-6)

    # Newton retains uploaded colors until the batch grows or the color changes.
    viewer.objects[name] = SimpleNamespace(num_instances=4)
    viewer.log_points(name, points)
    assert calls[-1][3] is None
    points = wp.zeros(6, dtype=wp.vec3, device="cpu")
    viewer.log_points(name, points)
    np.testing.assert_allclose(calls[-1][3].numpy(), np.tile([0.1, 0.2, 0.3], (6, 1)), rtol=1.0e-6)
    viewer.objects[name].num_instances = 6
    viewer.particle_color = (0.3, 0.2, 0.1)
    viewer.log_points(name, points)
    np.testing.assert_allclose(calls[-1][3].numpy(), np.tile([0.3, 0.2, 0.1], (6, 1)), rtol=1.0e-6)

    custom_colors = wp.zeros(6, dtype=wp.vec3, device="cpu")
    viewer.log_points("/user/custom_points", points, colors=custom_colors)
    assert calls[-1][3] is custom_colors


def test_newton_viewer_fast_paths_all_active_mpm_particles(monkeypatch):
    import newton as nt

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._mpm_particle_flags_cache_key = None
    viewer._mpm_particles_all_active = False
    viewer.model_changed = False
    viewer.particle_color = None
    viewer.show_particles = True
    viewer.model = SimpleNamespace(
        mpm=object(),
        particle_count=3,
        particle_radius=0.01,
        particle_flags=wp.array(
            [int(nt.ParticleFlags.ACTIVE)] * 3,
            dtype=wp.int32,
            device="cpu",
        ),
    )
    state = SimpleNamespace(particle_q=wp.zeros(3, dtype=wp.vec3, device="cpu"))
    log_points_calls = []

    monkeypatch.setattr(NewtonViewerGL, "log_points", lambda self, *a, **kw: log_points_calls.append((a, kw)))

    viewer._log_particles(state)

    # _log_particles calls self.log_points with all keyword args, so positional tuple is empty.
    assert len(log_points_calls) == 1
    assert log_points_calls[0][1]["name"] == "/model/particles"


def test_newton_viewer_inactive_mpm_particles_use_newton_filter(monkeypatch):
    import newton as nt
    from newton.viewer import ViewerGL

    viewer = NewtonViewerGL.__new__(NewtonViewerGL)
    viewer._mpm_particle_flags_cache_key = None
    viewer._mpm_particles_all_active = False
    viewer.model = SimpleNamespace(
        mpm=object(),
        particle_count=2,
        particle_flags=wp.array([int(nt.ParticleFlags.ACTIVE), 0], dtype=wp.int32, device="cpu"),
    )
    state = object()
    fallback_calls = []

    monkeypatch.setattr(ViewerGL, "_log_particles", lambda self, state: fallback_calls.append(state))

    viewer._log_particles(state)

    assert fallback_calls == [state]


class _Viewer:
    def __init__(self):
        self.device = "cpu"
        self.show_contacts = False
        self.paused = False
        self.logged_state = None
        self.logged_contacts = None
        self.logged_arrows = None
        self.logged_mesh = None
        self.events = []
        self.closed = False
        self.camera = SimpleNamespace(get_view_matrix=lambda: np.eye(4, dtype=np.float32).ravel())
        self.picking = None
        self.renderer = SimpleNamespace(window=SimpleNamespace(get_framebuffer_size=lambda: (640, 480)))

    def is_paused(self):
        return self.paused

    def is_running(self):
        return True

    def begin_frame(self, _time):
        self.events.append("begin_frame")

    def log_state(self, state):
        self.events.append("log_state")
        self.logged_state = state

    def log_image(self, name, image, *, fullscreen=False):
        self.events.append("log_image")
        self.logged_image = (name, image, fullscreen)

    def log_mesh(self, name, points, indices, **kwargs):
        self.events.append("log_mesh")
        self.logged_mesh = (name, points, indices, kwargs)

    def log_contacts(self, contacts, state):
        self.logged_contacts = (contacts, state)

    def log_arrows(self, name, starts, ends, colors):
        self.logged_arrows = (name, starts, ends, colors)

    def end_frame(self):
        self.events.append("end_frame")

    def close(self):
        # Mirrors ViewerBase.close(), which every real viewer inherits.
        self.closed = True

    def get_frame(self):
        return SimpleNamespace(numpy=lambda: np.zeros((4, 6, 3), dtype=np.uint8))


class _Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class _ContactSensorData:
    def __init__(self, net_normal_forces_w, pos_w):
        self.net_normal_forces_w = _Proxy(net_normal_forces_w)
        self.pos_w = _Proxy(pos_w)
        self.contact_pos_w = None
        self.normal_force_matrix_w = None


class _ContactSensor:
    def __init__(self, net_normal_forces_w, pos_w, force_threshold=1.0):
        self.cfg = SimpleNamespace(force_threshold=force_threshold)
        self.data = _ContactSensorData(net_normal_forces_w, pos_w)


class _SceneDataProvider:
    def __init__(self, contact_sensors=None):
        self._contact_sensors = contact_sensors or {}

    def get_contact_sensors(self):
        return self._contact_sensors

    def create_mapping(self, paths):
        return None

    def get_transforms(self, output, **kwargs):
        output.transforms = self.poses
        return True


def _make_newton_visualizer(viewer, scene_data_provider=None, state=None, *, cfg=None):
    cfg = cfg or NewtonGLVisualizerCfg(enable_markers=False)
    visualizer = instantiate(cfg)
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._sim_time = 0.0
    visualizer.cfg.window.fps = 1e9
    visualizer._runtime_headless = False
    visualizer._viewer = viewer
    state = state or SimpleNamespace(body_q=wp.empty(1, dtype=wp.transform, device="cpu"))
    visualizer.backend = SimpleNamespace(
        model=SimpleNamespace(num_envs=1, body_count=len(state.body_q)), state_0=state, geometry_offsets={}
    )
    provider = scene_data_provider or _SceneDataProvider()
    provider.poses = state.body_q
    visualizer._sim = SimpleNamespace(get_scene_data_provider=lambda: provider)
    visualizer._transform_mapping = None
    visualizer._live_plot_sources = []
    if viewer is not None:
        visualizer._viewer_picking_binding.bind(viewer)
    return visualizer


def test_newton_visualizer_forwards_and_neutralizes_picking():
    viewer = _Viewer()
    viewer.picking_enabled = True
    viewer.picking = SimpleNamespace(release=Mock())
    viewer.apply_forces = Mock()
    visualizer = _make_newton_visualizer(viewer)
    visualizer._picking_enabled = True
    callback = visualizer._viewer_picking_binding.apply

    state = object()
    callback(state)
    viewer.apply_forces.assert_called_once_with(state)

    visualizer.close()

    assert viewer.picking_enabled is False
    viewer.picking.release.assert_called_once_with()
    assert visualizer._viewer is None
    assert visualizer._viewer_picking_binding._viewer is None
    assert visualizer._viewer_picking_binding._retained_picking is viewer.picking

    callback(object())
    assert visualizer._viewer_picking_binding._retained_picking is None


@pytest.mark.parametrize("picking", [False, True])
def test_newton_visualizer_hard_reset_rebinds_viewer_model(monkeypatch, picking):
    from isaaclab_newton.physics import NewtonBackendCfg

    new_model = SimpleNamespace(body_label=["/Object"])
    new_state = object()
    backend = SimpleNamespace(model=new_model, state_0=new_state)
    sim = SimpleNamespace(get_or_create_backend=Mock(return_value=backend))
    monkeypatch.setattr(SimulationContext, "instance", Mock(side_effect=AssertionError("No global resource lookup")))

    viewer = _Viewer()
    viewer.picking_enabled = False
    viewer.set_model = Mock()
    viewer.renderer = SimpleNamespace()
    viewer.register_ui_callback = Mock()
    viewer._render_training_controls = Mock()
    viewer.set_visible_worlds = Mock()
    viewer.set_world_offsets = Mock()
    visualizer = _make_newton_visualizer(viewer)
    sim.get_scene_data_provider = visualizer._sim.get_scene_data_provider
    visualizer._sim = sim
    cfg = visualizer.newton_cfg = NewtonBackendCfg(physics_cfg=object(), device="cpu")
    visualizer._env_ids = [1, 3]
    visualizer._picking_enabled = picking
    visualizer.cfg.world_spacing = (2.0, 0.0, 0.0)
    visualizer.cfg.show_contacts = True

    visualizer.reset(soft=False)
    visualizer.reset(soft=False)

    assert visualizer.backend is backend
    sim.get_or_create_backend.assert_called_with(cfg)
    viewer.set_model.assert_called_once_with(new_model)
    assert viewer.register_ui_callback.call_args_list == [
        call(viewer._render_training_controls, position="side"),
        call(visualizer._draw_streaming_view_controls, position="side"),
    ]
    viewer.set_visible_worlds.assert_called_once_with([1, 3])
    viewer.set_world_offsets.assert_called_once_with((2.0, 0.0, 0.0))
    assert viewer.show_contacts is True
    assert viewer.picking_enabled is picking
    if picking:
        assert viewer.wind is None
    assert visualizer._viewer_picking_binding._viewer is viewer


def test_newton_visualizer_logs_native_contacts_when_available(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    state = SimpleNamespace(body_q=wp.empty(1, dtype=wp.transform, device="cpu"))
    contacts = object()
    viewer = _Viewer()

    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: contacts)

    _make_newton_visualizer(viewer, state=state).step(0.1)

    assert viewer.logged_state is state
    assert viewer.logged_contacts == (contacts, state)


def test_newton_visualizer_logs_staged_mesh_inside_frame(monkeypatch):
    from isaaclab_newton.physics import NewtonManager

    state = SimpleNamespace(body_q=wp.empty(1, dtype=wp.transform, device="cpu"))
    viewer = _Viewer()
    visualizer = _make_newton_visualizer(viewer, state=state)
    points = wp.zeros(3, dtype=wp.vec3)
    indices = wp.zeros(3, dtype=wp.int32)

    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: None)

    normals = wp.zeros(3, dtype=wp.vec3)

    visualizer.log_mesh(
        "/surface",
        points,
        indices,
        normals=normals,
        color=(0.1, 0.2, 0.3),
        roughness=0.2,
        dynamic=True,
        opacity=0.35,
    )
    assert viewer.events == []
    visualizer.step(0.1)

    assert viewer.events == ["begin_frame", "log_state", "log_mesh", "end_frame"]
    assert viewer.logged_mesh == (
        "/surface",
        points,
        indices,
        {
            "normals": normals,
            "uvs": None,
            "texture": None,
            "hidden": False,
            "backface_culling": True,
            "color": (0.1, 0.2, 0.3),
            "roughness": 0.2,
            "metallic": None,
            "dynamic": True,
            "opacity": 0.35,
        },
    )
    assert visualizer._pending_mesh_submissions == {}


def test_newton_visualizer_logs_staged_mesh_for_bodyless_state(monkeypatch):
    state = SimpleNamespace(body_q=wp.empty(0, dtype=wp.transform, device="cpu"))
    viewer = _Viewer()
    visualizer = _make_newton_visualizer(viewer, state=state)
    points = wp.zeros(3, dtype=wp.vec3)
    indices = wp.zeros(3, dtype=wp.int32)

    visualizer.log_mesh("/surface", points, indices, dynamic=True)
    visualizer.step(0.1)

    assert viewer.events == ["begin_frame", "log_mesh", "end_frame"]
    assert viewer.logged_state is None


def test_newton_gl_visualizer_logs_staged_mesh_while_paused(monkeypatch):
    viewer = _Viewer()
    viewer.paused = True
    visualizer = _make_newton_visualizer(viewer)
    points = wp.zeros(3, dtype=wp.vec3)
    indices = wp.zeros(3, dtype=wp.int32)

    visualizer.log_mesh("/surface", points, indices, dynamic=True)
    visualizer.step(0.1)

    assert viewer.events == ["begin_frame", "log_mesh", "end_frame"]
    assert viewer.logged_state is None


def test_newton_scene_camera_replaces_perspective_rendering(monkeypatch):
    """Image mode bypasses scene uploads, preserves pause, and can return to perspective."""
    viewer = _Viewer()
    cameras = [SceneCameraCfg(prim_path="/Camera"), PerspectiveCameraCfg()]
    visualizer = _make_newton_visualizer(viewer, cfg=NewtonGLVisualizerCfg(cameras=cameras, enable_markers=False))
    image = wp.full((4, 6, 4), 127, dtype=wp.uint8, device="cpu")
    visualizer._camera_sensor = Mock()
    visualizer._streaming_frame.data = image
    visualizer._streaming_frame.timestamp = 0.0
    visualizer.render_tiled_rgba_array = Mock(return_value=image)
    provider = visualizer._sim.get_scene_data_provider()
    provider.get_transforms = Mock(wraps=provider.get_transforms)
    contacts, markers = Mock(return_value=None), Mock()
    monkeypatch.setattr(newton_visualizer_module.NewtonManager, "get_contacts", contacts)
    monkeypatch.setattr(newton_visualizer_module, "render_newton_visualization_markers", markers)

    visualizer.step(0.1)
    viewer.paused = True
    visualizer.step(0.1)
    assert viewer.events == ["begin_frame", "log_image", "end_frame"] * 2
    name, displayed, fullscreen = viewer.logged_image
    assert name == "Streaming View" and displayed is image and fullscreen
    visualizer.render_tiled_rgba_array.assert_called_once()
    provider.get_transforms.assert_not_called()
    contacts.assert_not_called()
    markers.assert_not_called()

    viewer.paused = False
    visualizer._camera_sensor = None
    visualizer.step(0.1)
    assert viewer.events[-3:] == ["begin_frame", "log_state", "end_frame"]
    provider.get_transforms.assert_called_once()
    contacts.assert_called_once()


@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_newton_scene_camera_controls_apply_uniformly_to_active_copies(monkeypatch, device):
    """Only compatible cameras are offered; selection is lazy and navigation follows live parent poses."""
    import gymnasium as gym
    from isaaclab_newton.renderers import NewtonWarpRendererCfg

    from isaaclab.app import launch_simulation
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.scene.depth_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/cart/DepthCamera",
        width=64,
        height=48,
        data_types=["depth"],
        renderer_cfg=NewtonWarpRendererCfg(),
        spawn=PinholeCameraCfg(focal_length=24.0),
        offset=CameraCfg.OffsetCfg(pos=(-5.0, 0.0, 0.0), convention="world"),
    )
    cfg.scene.color_camera = cfg.scene.depth_camera.copy()
    cfg.scene.color_camera.prim_path = "{ENV_REGEX_NS}/Robot/cart/ColorCamera"
    cfg.scene.color_camera.data_types = ["rgba"]
    cfg.scene.back_camera = cfg.scene.color_camera.copy()
    cfg.scene.back_camera.prim_path = "{ENV_REGEX_NS}/Robot/cart/BackCamera"
    cfg.sim.visualizer_cfgs = [
        NewtonGLVisualizerCfg(headless=True, window=WindowCfg(size=(128, 128)), streaming_envs=[0])
    ]
    with launch_simulation(cfg, {"visualizer": ["newton_gl"], "device": device}):
        env = gym.make("Isaac-Cartpole", cfg=cfg)
        try:
            env.reset()
            depth, color = env.unwrapped.scene["depth_camera"], env.unwrapped.scene["color_camera"]
            back = env.unwrapped.scene["back_camera"]
            visualizer = env.unwrapped.sim.visualizers[0]
            depth_frame = depth.frame.torch.clone()
            color_frame = color.frame.torch.clone()
            back_frame = back.frame.torch.clone()
            assert visualizer._camera_sensor is None
            assert isinstance(visualizer._cameras[0], PerspectiveCameraCfg)
            assert [camera.cfg.prim_path for camera in visualizer._cameras[1:]] == [
                color.cfg.prim_path,
                back.cfg.prim_path,
            ]

            visualizer._select_camera(1)
            torch.testing.assert_close(color.frame.torch, color_frame)
            image = visualizer.render_tiled_rgb_array()
            np.testing.assert_array_equal(image, color.data.output["rgb"].torch[0].cpu().numpy())
            assert np.ptp(image) > 0
            torch.testing.assert_close(depth.frame.torch, depth_frame)

            # Exercise the real GL image sink without downloading the sensor pixels.
            with monkeypatch.context() as gpu_display:
                gpu_display.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("Display downloaded pixels")))
                visualizer._sim_time += cfg.sim.dt
                visualizer._log_streaming_image()

            # Give the copies different orientations, then move their parent without refreshing measurements.
            orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2**-0.5, 2**-0.5]], device=device)
            color.set_world_poses(orientations=orientations, convention="opengl")
            cached = color.data.pos_w.torch.clone()
            robot = env.unwrapped.scene["robot"]
            joint_positions = robot.data.joint_pos.torch.clone()
            joint_positions[:, robot.find_joints("slider_to_cart")[0]] += 1.0
            robot.write_joint_state_to_sim_index(position=joint_positions, velocity=torch.zeros_like(joint_positions))
            env.unwrapped.sim.forward()
            env.unwrapped.sim.step(render=False)
            live = color._view.get_world_poses()[0].torch.clone()
            assert torch.all(torch.linalg.vector_norm(live - cached, dim=-1) > 0.5)
            torch.testing.assert_close(color.data.pos_w.torch, cached)
            inactive = back._view.get_world_poses()[0].torch.clone()

            # Supply a known mouse/keyboard delta; sensors, body motion, and camera writes remain real.
            viewer = _Viewer()
            delta = np.asarray([[0, -1, 0, 0.25], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
            after = np.linalg.inv(delta).T.ravel()
            viewer.camera.get_view_matrix = Mock(return_value=after)
            visualizer._navigation_view = np.eye(4, dtype=np.float32)
            with monkeypatch.context() as controls:
                controls.setattr(visualizer, "_viewer", viewer)
                controls.setattr(visualizer, "_runtime_headless", False)
                visualizer.step(0.0)
                pos, quat = color._view.get_world_poses()
                expected_delta = torch.tensor([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0]], device=device)
                expected_quat = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5], [0.0, 0.0, 1.0, 0.0]], device=device)
                torch.testing.assert_close(pos.torch, live + expected_delta)
                torch.testing.assert_close(quat.torch, expected_quat, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(back._view.get_world_poses()[0].torch, inactive)
                torch.testing.assert_close(back.frame.torch, back_frame)

                # Switching only binds; the next display captures the newly selected sensor alone.
                color_frame = color.frame.torch.clone()
                back.update(cfg.sim.dt)
                viewer.paused = True
                visualizer._select_camera(2)
                torch.testing.assert_close(back.frame.torch, back_frame)
                viewer.camera.get_view_matrix = Mock(return_value=after)
                visualizer.step(0.0)
                assert torch.all(back.frame.torch > back_frame)
                torch.testing.assert_close(color.frame.torch, color_frame)
                torch.testing.assert_close(back._view.get_world_poses()[0].torch, inactive)
                torch.testing.assert_close(depth.frame.torch, depth_frame)
        finally:
            env.close()


def test_newton_visualizer_headless_renders_frame_on_demand(monkeypatch):
    """Headless viewers share on-demand binding, preserve pause, and close frames even on errors."""

    state = SimpleNamespace(
        body_q=wp.empty(1, dtype=wp.transform, device="cpu"), particle_q=wp.empty(3, dtype=wp.vec3, device="cpu")
    )
    viewer = _Viewer()
    markers = Mock()
    monkeypatch.setattr(newton_visualizer_module, "render_newton_visualization_markers", markers)
    visualizer = _make_newton_visualizer(viewer, state=state, cfg=NewtonGLVisualizerCfg(enable_markers=True))
    provider = visualizer._sim.get_scene_data_provider()
    provider.get_transforms = Mock(wraps=provider.get_transforms)
    provider.get_geometry_points = Mock()
    visualizer.backend.geometry_offsets = {"/Cloth": 0}
    visualizer._runtime_headless = True
    visualizer.step(0.1)
    assert viewer.logged_state is None
    provider.get_transforms.assert_not_called()

    assert visualizer.render_rgb_array().shape == (4, 6, 3)
    assert viewer.logged_state is state
    assert viewer.events == ["begin_frame", "log_state", "end_frame"]
    assert markers.call_count == 1
    provider.get_geometry_points.assert_called_once_with(output=state.particle_q, offsets={"/Cloth": 0})

    viewer.paused = True
    visualizer.render_rgb_array()
    assert len(viewer.events) == 3
    provider.get_transforms.assert_called_once()

    viewer.paused = False
    visualizer._runtime_headless = False
    visualizer.cfg.window.fps = 20.0
    clock = iter((0.0, 0.01, 0.025, 0.05, 0.1))
    monkeypatch.setattr(newton_visualizer_module.time, "monotonic", lambda: next(clock))
    viewer.events.clear()
    for dt in (0.0, 10.0, 0.1, 0.0):
        visualizer.step(dt)
    assert viewer.events == ["begin_frame", "log_state", "end_frame"] * 2

    visualizer._runtime_headless = True
    provider.poses = wp.empty(1, dtype=wp.transform, device="cpu")
    viewer.log_state = Mock(side_effect=RuntimeError("render failed"))
    with pytest.raises(RuntimeError, match="render failed"):
        visualizer.render_rgb_array()
    assert state.body_q is provider.poses
    assert viewer.events[-2:] == ["begin_frame", "end_frame"]

    visualizer._runtime_headless = False
    viewer.events.clear()
    with pytest.raises(RuntimeError, match="render failed"):
        visualizer.step(0.1)
    assert viewer.events == ["begin_frame", "end_frame"]


def test_newton_live_plots_read_updated_scalar_and_array_history():
    from newton._src.viewer.plot_logger import PlotLogger

    viewer = _Viewer()
    viewer._plot_logger = plots = PlotLogger(4, get_window=lambda: None)
    viewer.log_scalar = plots.log_scalar
    viewer.gui = SimpleNamespace(ui=SimpleNamespace(dpi_scale=1.0))
    viewer._implot = Mock(begin_plot=Mock(return_value=True))
    imgui = Mock(collapsing_header=Mock(return_value=True))
    imgui.get_content_region_avail.return_value.x = 200
    plots._render_array_heatmap = Mock()
    visualizer = _make_newton_visualizer(viewer, cfg=NewtonGLVisualizerCfg(live_plots_update_interval=1))
    samples = iter((1.25, 2.5))
    visualizer.add_live_plots({}, scalars={"metrics": {"loss": lambda: next(samples)}})

    for value in (1.25, 2.5):
        visualizer._render_live_plots()
        visualizer._live_plots_panel_imgui(imgui)
        viewer._implot.plot_line.assert_called()
        label, array = viewer._implot.plot_line.call_args.args
        assert label == "loss" and array[-1] == value
    np.testing.assert_allclose(array[-2:], [1.25, 2.5])

    heatmap = np.arange(4, dtype=np.float32).reshape(2, 2)
    plots.log_array("heatmap", heatmap)
    visualizer._live_plots_panel_imgui(imgui)
    plots._render_array_heatmap.assert_called_once_with(imgui, "heatmap", heatmap, 180.0, dpi_scale=1.0)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("origins", ["tracked", "filtered", "physx"])
def test_newton_visualizer_contact_sensor_fallback_obeys_show_contacts(monkeypatch, device, origins):
    from isaaclab_newton.physics import NewtonManager

    state = SimpleNamespace(body_q=wp.empty(1, dtype=wp.transform, device="cpu"))
    viewer = _Viewer()
    viewer.device = device
    positions = torch.tensor(
        [[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], device=device
    )
    sensor = _ContactSensor(
        net_normal_forces_w=torch.tensor([[[0.0, 0.0, 2.0], [0.0, 0.0, 0.5]]], device=device).repeat(2, 1, 1),
        pos_w=positions,
        force_threshold=1.0,
    )
    if origins == "filtered":
        sensor.data.contact_pos_w = _Proxy(positions.unsqueeze(2))
        sensor.data.normal_force_matrix_w = _Proxy(sensor.data.net_normal_forces_w.torch.unsqueeze(2))
        sensor.data.pos_w = None
    elif origins == "physx":
        poses = torch.zeros((2, 2, 7), device=device)
        poses[..., :3] = positions.transpose(0, 1)  # PhysX orders body poses before environments.
        sensor.body_physx_view = SimpleNamespace(get_transforms=lambda: wp.from_torch(poses.reshape(-1, 7)))
        sensor.data.pos_w = None
        monkeypatch.setattr(BaseVisualizer, "physics_backend", property(lambda self: "physx"))
    scene_data_provider = _SceneDataProvider({"contact_forces": sensor})
    monkeypatch.setattr(NewtonManager, "get_contacts", lambda: None)

    visualizer = _make_newton_visualizer(viewer, scene_data_provider, state=state)
    visualizer.backend.model.num_envs = 2
    visualizer._env_ids = [1]
    visualizer.step(0.1)
    assert viewer.logged_arrows == ("/contacts", None, None, None)

    viewer.show_contacts = True
    with monkeypatch.context() as display:
        display.setattr(torch.Tensor, "numpy", Mock(side_effect=AssertionError("Arrow display downloaded arrays")))
        visualizer.step(0.1)

    name, starts, ends, colors = viewer.logged_arrows
    assert name == "/contacts"
    assert len(starts) == len(ends) == 1
    assert starts.device == ends.device == wp.get_device(device)
    assert colors == (0.0, 1.0, 0.0)
    torch.testing.assert_close(wp.to_torch(starts).cpu(), torch.tensor([[1.0, 2.0, 3.0]]))
    torch.testing.assert_close(wp.to_torch(ends).cpu(), torch.tensor([[1.0, 2.0, 3.1]]))


# ── USD marker inference and None-normal guard ────────────────────────


class UsdFileCfg:
    """Minimal stand-in that duck-types ``isaaclab.sim.spawners.UsdFileCfg``."""

    def __init__(self, usd_path, scale=None):
        self.usd_path = usd_path
        self.scale = scale


def test_infer_newton_marker_cfg_generic_usd_loads_mesh():
    import os

    import newton
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    usd_path = os.path.join(os.path.dirname(newton.__file__), "tests", "assets", "cube_cylinder.usda")
    spec = _infer_newton_marker_cfg(UsdFileCfg(usd_path))

    assert spec.renderer == "mesh"
    assert spec.mesh_type == "usd"
    assert spec.preloaded_mesh is not None
    assert spec.preloaded_mesh.vertices.shape[0] > 0


def test_infer_newton_marker_cfg_missing_usd_falls_back_to_renderer_none():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    spec = _infer_newton_marker_cfg(UsdFileCfg("/nonexistent/missing.usd"))

    assert spec.renderer == "none"


def test_infer_newton_marker_cfg_arrow_x_usd_still_maps_to_builtin_arrow():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    spec = _infer_newton_marker_cfg(UsdFileCfg("/assets/arrow_x.usd"))

    assert spec.renderer == "mesh"
    assert spec.mesh_type == "arrow"
    assert spec.preloaded_mesh is None


def test_infer_newton_marker_cfg_frame_prim_usd_still_maps_to_frame_renderer():
    from isaaclab_visualizers.newton.newton_visualization_markers import _infer_newton_marker_cfg

    spec = _infer_newton_marker_cfg(UsdFileCfg("/assets/frame_prim.usd"))

    assert spec.renderer == "frame"


def test_ensure_mesh_registered_handles_none_normals_and_uvs(monkeypatch):
    import isaaclab_visualizers.newton.newton_visualization_markers as _mod
    import numpy as np
    from isaaclab_visualizers.newton.newton_visualization_markers import (
        NewtonVisualizationMarkers,
        _NewtonMarkerSpec,
    )

    fake_mesh = SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        indices=np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        normals=None,
        uvs=None,
    )
    monkeypatch.setattr(_mod, "_create_mesh", lambda cfg: fake_mesh)

    log_calls = []

    class _LoggingViewer:
        device = "cpu"

        def log_mesh(self, name, vertices, indices, normals=None, uvs=None, texture=None, hidden=True):
            log_calls.append({"normals": normals, "uvs": uvs})

    fake_self = SimpleNamespace(_registered_meshes=set())
    spec = _NewtonMarkerSpec(renderer="mesh", mesh_type="usd", preloaded_mesh=fake_mesh)

    NewtonVisualizationMarkers._ensure_mesh_registered(fake_self, _LoggingViewer(), "/test/mesh", spec)

    assert len(log_calls) == 1
    assert log_calls[0]["normals"] is None
    assert log_calls[0]["uvs"] is None


# ---------------------------------------------------------------------------
# RTX backend tests
# ---------------------------------------------------------------------------


def test_newton_visualizer_cfg():
    assert NewtonGLVisualizerCfg().visualizer_type == "newton_gl"
    assert NewtonRTXVisualizerCfg().visualizer_type == "newton_rtx"
    assert not issubclass(NewtonRTXVisualizer, NewtonGLVisualizer)
    for option in ("rtx_environment", "world_spacing", "enable_picking"):
        with pytest.raises(TypeError, match=option):
            NewtonRTXVisualizerCfg(**{option: None})
    # Public viewer options are accepted as cfg fields.
    NewtonGLVisualizerCfg(enable_picking=False, show_particles=True, particle_color=(0.1, 0.2, 0.3))
    cfg = NewtonRTXVisualizerCfg(cameras=[SceneCameraCfg(prim_path="/Camera")])
    validate(cfg)


@pytest.mark.parametrize("color", [(0.1, 0.2, 0.3), None])
def test_newton_gl_background_color(color: tuple[float, float, float] | None) -> None:
    cfg = NewtonGLVisualizerCfg(background_color=color)
    visualizer = NewtonGLVisualizer(cfg)
    visualizer._viewer = SimpleNamespace(
        renderer=SimpleNamespace(),
    )

    visualizer._configure_viewer()

    assert visualizer._viewer.renderer.draw_sky == (color is None)
    expected_upper = cfg.sky_upper_color if color is None else color
    expected_lower = cfg.sky_lower_color if color is None else color
    assert visualizer._viewer.renderer.sky_upper == expected_upper
    assert visualizer._viewer.renderer.sky_lower == expected_lower


@pytest.mark.rendering
@pytest.mark.kitless
@pytest.mark.parametrize("lighting", [True, False])
@pytest.mark.parametrize("background_color", [None, (0.0, 0.0, 1.0)])
def test_newton_rtx_scene_sky_and_background_override(tmp_path, monkeypatch, lighting, background_color):
    """The native viewer preserves authored lighting and borrows a stage without owning sensors."""
    if lighting:
        pytest.skip(
            "Two OVRTX renderers can hang; re-enable after adopting https://github.com/newton-physics/newton/pull/4627."
        )
    import gymnasium as gym
    from isaaclab_newton.renderers import NewtonWarpRendererCfg
    from isaaclab_ov.renderers import OVRTXRendererCfg

    import isaaclab.sim as sim_utils
    from isaaclab.app import launch_simulation
    from isaaclab.assets import VisualMaterialCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.visualizers import PerspectiveCameraCfg

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import resolve_task_config

    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")

    texture = tmp_path / "red.hdr"
    texture.write_bytes(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 2 +X 4\n" + bytes((128, 0, 0, 129)) * 8)
    cfg, _ = resolve_task_config("Isaac-Cartpole", "", overrides=("physics=newton_mjwarp",))
    cfg.scene.num_envs = 2
    cfg.seed = 0
    cfg.scene.distant_light = AssetBaseCfg(
        prim_path="/World/Sky", spawn=sim_utils.DomeLightCfg(texture_file=str(texture), intensity=750.0)
    )
    if not lighting:
        cfg.scene.distant_light = None
    cfg.scene.target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(size=(1.0, 1.0, 1.0), visual_material=sim_utils.PreviewSurfaceCfg()),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -2.0, 2.0)),
    )
    cfg.scene.material = VisualMaterialCfg(prim_path="{ENV_REGEX_NS}/Target/geometry/material", spawn=None)
    cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        width=128,
        height=128,
        data_types=["rgba"],
        renderer_cfg=OVRTXRendererCfg() if lighting else NewtonWarpRendererCfg(),
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=15.2908),
        offset=CameraCfg.OffsetCfg(pos=(0.0, -6.0, 2.0), rot=(0.0, 0.0, 2**-0.5, 2**-0.5), convention="world"),
    )
    cfg.sim.visualizer_cfgs = [
        NewtonRTXVisualizerCfg(
            headless=True,
            window=WindowCfg(size=(128, 128)),
            streaming_envs=[0],
            background_color=background_color,
            cameras=[
                PerspectiveCameraCfg(eye=(0.0, -6.0, 2.0), lookat=(0.0, -2.0, 2.0), focal_length=24.0),
                SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera"),
            ],
        )
    ]
    with launch_simulation(cfg, {"visualizer": ["newton_rtx"], "device": "cuda:0"}):
        env = gym.make("Isaac-Cartpole", cfg=cfg)
        try:
            env.reset()
            visualizer = env.unwrapped.sim.visualizers[0]
            camera = env.unwrapped.scene["camera"]
            viewer = visualizer._viewer
            native_stage = viewer._borrowed_stage
            # Exercise the native presentation path through an EGL window in headless CI.
            viewer._headless = False
            assert visualizer._camera_sensor is None
            assert len(env.unwrapped.scene.sensors) == 1
            origin = env.unwrapped.scene.env_origins[0].cpu().numpy()
            visualizer.set_camera_view(origin + (0.0, -6.0, 2.0), origin + (0.0, -2.0, 2.0))
            for _ in range(40):
                pixels = visualizer.render_rgb_array()
            background = pixels[8:24, 8:24].mean(axis=(0, 1))
            color = pixels[56:72, 56:72].mean(axis=(0, 1))
            channel = 0 if background_color is None else 2
            if lighting or background_color is not None:
                assert background[channel] > 180 and np.delete(background, channel).max() < 10, background
            else:
                assert background.max() < 10, background
            if lighting:
                assert color[0] > 80 and color[0] > 2 * color[1:].max(), color
            else:
                assert color.max() < 10, color

            output = camera.data.output["rgba"].warp
            sensor_frame = camera.frame.torch.clone()
            sensor_ptr = output.ptr
            captured_output = wp.empty_like(output)
            with wp.ScopedCapture(device=output.device) as capture:
                wp.copy(captured_output, output)
            viewer._window.set_size(160, 96)
            viewer._window.dispatch_events()
            pixels = visualizer.render_rgb_array()
            assert pixels.shape == (128, 128, 3)
            assert camera.image_shape == (128, 128)
            assert camera.data.output["rgba"].warp.ptr == sensor_ptr
            torch.testing.assert_close(camera.frame.torch, sensor_frame)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(captured_output.numpy(), output.numpy())

            visualizer._select_camera(1)
            image = visualizer.render_tiled_rgba_array()
            np.testing.assert_array_equal(image.numpy(), camera.data.output["rgba"].warp.numpy()[0])
            assert image.device.is_cuda
            with monkeypatch.context() as gpu_display:
                gpu_display.setattr(wp.array, "numpy", Mock(side_effect=AssertionError("Display downloaded pixels")))
                visualizer._render_frame()
            if lighting and background_color is None:
                env.unwrapped.sim.reset()
                visualizer._select_camera(0)
                assert visualizer.render_rgb_array().shape == (128, 128, 3)
            visualizer.close()
            query = native_stage.get_attribute_write_floor()
            try:
                assert native_stage.fetch_ordinal(query) > 0
            finally:
                native_stage.release_ordinal_query(query).wait()
            camera.update(cfg.sim.dt)
            assert np.ptp(camera.data.output["rgba"].warp.numpy()) > 0
        finally:
            env.close()


def test_newton_rtx_visualizer_render_rgb_array_returns_none_when_viewer_unavailable():
    visualizer = NewtonRTXVisualizer(NewtonRTXVisualizerCfg())

    assert visualizer.render_rgb_array() is None


@pytest.mark.parametrize("backend", ["physx", "isaacsim_physx"])
def test_newton_rtx_visualizer_rejects_kit_physics_backend(monkeypatch, backend):
    """OVRTX is kitless and must fail fast instead of crashing the render thread on first step().

    "physx" is what FactoryBase._get_backend() reports at runtime (covers both an explicit
    ``physics=isaacsim_physx`` and the ``physics=physx`` auto selector once resolved to Kit);
    "isaacsim_physx" is checked too in case a future/alternate backend-name source reports the
    explicit selector string instead.
    """
    from isaaclab.visualizers.base_visualizer import BaseVisualizer

    monkeypatch.setattr(BaseVisualizer, "physics_backend", property(lambda self: backend))
    visualizer = NewtonRTXVisualizer(NewtonRTXVisualizerCfg())

    with pytest.raises(RuntimeError, match="Newton RTX"):
        visualizer.initialize(Mock(), cameras=[])
