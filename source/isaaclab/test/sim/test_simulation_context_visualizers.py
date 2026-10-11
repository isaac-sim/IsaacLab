# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SimulationContext visualizer orchestration."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonBackendCfg
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import (
    NewtonGLVisualizerCfg,
    NewtonRTXVisualizerCfg,
    NewtonVisualizerCfg,
)
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.visualizers import WindowCfg
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import PerspectiveCameraCfg, SceneCameraCfg, VisualizerCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


def test_web_visualizer_cfgs_do_not_open_browser_by_default():
    assert RerunVisualizerCfg().open_browser is False
    assert ViserVisualizerCfg().open_browser is False


class _FakePhysicsManager:
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeProvider:
    """Fake new-style SceneDataProvider for tests; only provides what visualizers read."""

    def __init__(self, num_envs: int = 0):
        self._num_envs = num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def get_camera_transforms(self):
        return None


class _FakeVisualizer(BaseVisualizer):
    """Minimal visualizer for orchestration tests."""

    def __init__(
        self,
        cfg=None,
        *,
        env_ids=None,
        running=True,
        closed=False,
        rendering_paused=False,
        training_paused_steps=0,
        raises_on_step=False,
        requires_forward=False,
        pumps_app_update=False,
    ):
        super().__init__(cfg or VisualizerCfg())
        self._env_ids = env_ids
        self._running = running
        self._closed = closed
        self._rendering_paused = rendering_paused
        self._training_paused_steps = training_paused_steps
        self._raises_on_step = raises_on_step
        self._requires_forward = requires_forward
        self._pumps_app_update = pumps_app_update
        self.step_calls = []
        self.close_calls = 0

    def initialize(self, sim, *, cameras):
        super().initialize(sim, cameras=cameras)
        self._is_initialized = True

    @property
    def is_closed(self):
        return self._closed

    def is_running(self):
        return self._running

    def is_rendering_paused(self):
        return self._rendering_paused

    def is_training_paused(self):
        if self._training_paused_steps > 0:
            self._training_paused_steps -= 1
            return True
        return False

    def step(self, dt):
        self.step_calls.append(dt)
        if self._raises_on_step:
            raise RuntimeError("step failed")

    def close(self):
        self.close_calls += 1
        self._closed = True

    def get_visualized_env_ids(self):
        return self._env_ids

    def requires_forward_before_step(self):
        return self._requires_forward

    def pumps_app_update(self):
        return self._pumps_app_update

    def supports_markers(self):
        return False

    def supports_live_plots(self):
        return False

    def flush_startup_messages(self):
        pass


def _make_context(visualizers, provider=None):
    ctx = object.__new__(SimulationContext)
    ctx._visualizers = list(visualizers)
    ctx._visualizers_started = bool(visualizers)
    ctx._scene_data_provider = provider
    ctx.physics_manager = _FakePhysicsManager()
    ctx.vis_marker_registry = VisMarkerRegistry()
    return ctx


def test_update_visualizers_runs_forward_when_a_visualizer_requires_it():
    provider = _FakeProvider()
    viz_a = _FakeVisualizer(env_ids=[0, 2], requires_forward=True)
    viz_b = _FakeVisualizer(env_ids=[2, 3])
    ctx = _make_context([viz_a, viz_b], provider=provider)

    ctx.update_visualizers(0.1)

    assert ctx.physics_manager.forward_calls == 1
    assert viz_a.step_calls == [0.1]
    assert viz_b.step_calls == [0.1]


def test_update_visualizers_skips_forward_when_no_visualizer_requires_it():
    provider = _FakeProvider()
    viz = _FakeVisualizer(env_ids=[0])
    ctx = _make_context([viz], provider=provider)

    ctx.update_visualizers(0.1)

    assert ctx.physics_manager.forward_calls == 0


def test_update_visualizers_removes_closed_nonrunning_and_failed(caplog):
    provider = _FakeProvider()
    closed_viz = _FakeVisualizer(closed=True)
    stopped_viz = _FakeVisualizer(running=False)
    failing_viz = _FakeVisualizer(raises_on_step=True)
    paused_viz = _FakeVisualizer(rendering_paused=True)
    healthy_viz = _FakeVisualizer(env_ids=[1])
    ctx = _make_context([closed_viz, stopped_viz, failing_viz, paused_viz, healthy_viz], provider=provider)

    with caplog.at_level("ERROR"):
        ctx.update_visualizers(0.1)

    assert ctx._visualizers == [paused_viz, healthy_viz]
    assert closed_viz.close_calls == 1
    assert stopped_viz.close_calls == 1
    assert failing_viz.close_calls == 1
    assert paused_viz.close_calls == 0
    assert paused_viz.step_calls == [0.0]
    assert healthy_viz.step_calls == [0.1]
    assert any("Error stepping visualizer" in r.message for r in caplog.records)


def test_is_running_until_the_last_visualizer_closes():
    """Headless runs keep going; once the last visualizer closes and is dropped, the run ends."""
    assert _make_context([]).is_running()

    stopped_viz = _FakeVisualizer(running=False)
    ctx = _make_context([stopped_viz], provider=_FakeProvider())
    ctx.update_visualizers(0.1)

    assert ctx._visualizers == []
    assert not ctx.is_running()


def test_update_visualizers_skips_zero_dt_for_paused_app_pumping_visualizer():
    provider = _FakeProvider()
    paused_app_pumping_viz = _FakeVisualizer(rendering_paused=True, pumps_app_update=True)
    ctx = _make_context([paused_app_pumping_viz], provider=provider)

    ctx.update_visualizers(0.3)

    assert paused_app_pumping_viz.step_calls == []


def test_update_visualizers_handles_training_pause_loop():
    provider = _FakeProvider()
    viz = _FakeVisualizer(training_paused_steps=1)
    ctx = _make_context([viz], provider=provider)

    ctx.update_visualizers(0.2)

    assert viz.step_calls == [0.0, 0.2]


class _LivePlotVisualizer(_FakeVisualizer):
    def __init__(self, *, enable_live_plots: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cfg = VisualizerCfg(enable_live_plots=enable_live_plots)

    def supports_live_plots(self):
        return True


def test_update_visualizers_dispatches_callbacks_for_live_plot_only_visualizer():
    """Live-plot panels share the marker registry, so dispatch must not require marker support."""
    dispatched = []
    ctx = _make_context([_LivePlotVisualizer()], provider=_FakeProvider())
    ctx.vis_marker_registry.add_callback("probe", dispatched.append)

    ctx.update_visualizers(0.1)

    assert len(dispatched) == 1


def test_update_visualizers_skips_dispatch_when_live_plots_disabled():
    """Live-plot support with the flag off consumes nothing, so callbacks stay idle."""
    dispatched = []
    ctx = _make_context([_LivePlotVisualizer(enable_live_plots=False)], provider=_FakeProvider())
    ctx.vis_marker_registry.add_callback("probe", dispatched.append)

    ctx.update_visualizers(0.1)

    assert dispatched == []


def test_newton_visualizer_is_initialized_and_rebound_before_capture():
    created = []
    reset_calls = []
    camera_calls = []

    class _Cfg(VisualizerCfg):
        def __init__(self, visualizer_type, enable_picking=False):
            super().__init__()
            self.visualizer_type = visualizer_type
            self.enable_picking = enable_picking
            self.headless = False
            self.class_type = self._construct

        def _construct(self, cfg):
            viz = _FakeVisualizer(cfg)
            viz.initialize = lambda _provider, **_scene: created.append(cfg.visualizer_type)
            viz.reset = lambda soft: reset_calls.append((cfg.visualizer_type, soft))
            viz.set_camera_view = lambda eye, target: camera_calls.append((cfg.visualizer_type, eye, target))
            return viz

    ctx = _make_context_with_settings(
        {}, visualizer_cfgs=[_Cfg("newton_gl", True), _Cfg("newton_rtx", True), _Cfg("rerun")]
    )
    ctx.get_or_create_backend = Mock(side_effect=AssertionError("Core must not construct viewer backends"))
    ctx._create_visualizers()
    eye, target = (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)
    ctx.set_camera_view(eye, target)
    ctx._prepare_newton_visualizer_for_capture()
    assert created == ["newton_gl"]

    ctx.initialize_visualizers()
    ctx._prepare_newton_visualizer_for_capture()

    assert created == ["newton_gl", "newton_rtx", "rerun"]
    assert len(ctx._visualizers) == 3
    assert camera_calls == [(name, eye, target) for name in ("newton_gl", "newton_rtx", "rerun")]
    assert ctx._pending_camera_view is None
    assert reset_calls == [
        ("newton_gl", False),
        ("newton_gl", False),
    ]


def test_reset_initializes_visualizers_before_playing_timeline():
    """Initial visualizers must see the PhysX views created by reset before play() pumps timeline events."""
    events: list[str] = []
    ctx = object.__new__(SimulationContext)
    ctx.cfg = SimpleNamespace(physics=object())
    ctx._visualizers = [
        SimpleNamespace(
            cfg=SimpleNamespace(visualizer_type="newton_rtx"), reset=lambda soft: events.append("viewer_reset")
        )
    ]
    ctx.get_or_create_backend = Mock(side_effect=AssertionError("Core must not rebind viewer backends"))

    class _PhysicsManager:
        @staticmethod
        def get_device():
            return "cpu"

        @staticmethod
        def reset(soft=False):
            events.append(f"reset:{soft}")

        @staticmethod
        def play():
            events.append("play")

    class _RenderContext:
        @staticmethod
        def finalize_consumers(visualizers, *, rebuild):
            events.append(f"finalize_consumers:{len(visualizers)}:{rebuild}")

    def _initialize_visualizers():
        events.append("initialize_visualizers")
        ctx._visualizers = [_FakeVisualizer()]

    ctx.physics_manager = _PhysicsManager()
    ctx._render_context = _RenderContext()
    ctx.initialize_visualizers = _initialize_visualizers

    ctx.reset()

    assert events == ["reset:False", "viewer_reset", "initialize_visualizers", "finalize_consumers:1:True", "play"]
    assert ctx.is_playing()
    assert not ctx.is_stopped()


class _DummyViserSceneDataProvider:
    @property
    def num_envs(self) -> int:
        return 4

    def get_camera_transforms(self):
        return {}

    def create_mapping(self, paths):
        return None

    def get_transforms(self, output, **kwargs):
        output.transforms = wp.zeros(1, dtype=wp.transform, device="cpu")
        return True


@pytest.fixture
def web_backend(monkeypatch):
    model = SimpleNamespace(body_label=["/Object"], body_count=1, num_envs=4)
    backend = SimpleNamespace(model=model, state_0=SimpleNamespace(body_q=None), geometry_offsets={})
    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=object(), device="cpu"),
        device="cpu",
        get_or_create_backend=Mock(return_value=backend),
        vis_marker_registry=VisMarkerRegistry(),
        stage=None,
        get_scene_data_provider=Mock(return_value=_DummyViserSceneDataProvider()),
    )
    monkeypatch.setattr(SimulationContext, "instance", lambda: sim)
    return sim


class _DummyViserViewer:
    def __init__(self):
        self.calls = []
        self._server = object()
        self.set_model = Mock()
        self.set_visible_worlds = Mock()
        self.set_world_offsets = Mock()

    def begin_frame(self, sim_time: float) -> None:
        self.calls.append(("begin_frame", sim_time))

    def log_state(self, state) -> None:
        self.calls.append(("log_state", state))

    def end_frame(self) -> None:
        self.calls.append(("end_frame",))

    def is_running(self) -> bool:
        return True


def test_viser_visualizer_reads_sdp_and_rebinds_native_resource(monkeypatch, web_backend):
    provider = _DummyViserSceneDataProvider()
    web_backend.get_scene_data_provider.return_value = provider
    viewer = _DummyViserViewer()

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        assert record_to_viser is None
        assert metadata == {"num_envs": provider.num_envs}
        self._viewer = viewer

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_setup_isaaclab_sidebar", lambda self, server: None)

    cfg = NewtonBackendCfg(physics_cfg=web_backend.cfg.physics, device=web_backend.device)
    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(web_backend, cameras=[])
    visualizer.step(0.25)

    assert visualizer.is_initialized
    backend = web_backend.get_or_create_backend.return_value
    assert visualizer.backend is backend
    assert backend.state_0.body_q.shape == (1,)
    assert visualizer._sim_time == pytest.approx(0.25)
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[0][1] == pytest.approx(0.25)
    assert viewer.calls[1] == ("log_state", backend.state_0)
    assert viewer.calls[2] == ("end_frame",)
    replacement = SimpleNamespace(model=backend.model, state_0=SimpleNamespace(body_q=None), geometry_offsets={})
    web_backend.get_or_create_backend.return_value = replacement
    visualizer.reset(soft=True)
    assert visualizer.backend is backend
    visualizer.reset()
    visualizer.reset()
    viewer.set_model.assert_called_once_with(replacement.model)
    web_backend.get_or_create_backend.assert_called_with(cfg)
    visualizer.step(0.25)
    assert viewer.calls[-2] == ("log_state", replacement.state_0)


@pytest.mark.parametrize(
    ("cfg_max_visible_envs", "expected_visible"),
    [
        (None, None),
        (0, []),
        (3, [0, 1, 2]),
    ],
)
def test_viser_visualizer_create_viewer_applies_visible_worlds(
    monkeypatch: pytest.MonkeyPatch,
    cfg_max_visible_envs: int | None,
    expected_visible: list[int] | None,
):
    captured = {}

    class _FakeNewtonViewerViser:
        def __init__(
            self,
            *,
            port: int,
            bind_address: str,
            label: str | None,
            verbose: bool,
            share: bool,
            record_to_viser: str | None,
            metadata: dict | None = None,
        ):
            captured["init"] = {
                "port": port,
                "bind_address": bind_address,
                "label": label,
                "verbose": verbose,
                "share": share,
                "record_to_viser": record_to_viser,
                "metadata": metadata,
            }

        def set_model(self, model: Any) -> None:
            captured["set_model"] = model

        def set_visible_worlds(self, worlds) -> None:
            captured["visible_worlds"] = worlds

        def set_world_offsets(self, spacing) -> None:
            captured["set_world_offsets"] = tuple(spacing)

        @property
        def share_url(self) -> str | None:
            return None

    monkeypatch.setattr(viser_visualizer, "NewtonViewerViser", _FakeNewtonViewerViser)
    apply_pose = Mock()
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_set_viser_camera_view", apply_pose)

    cfg = ViserVisualizerCfg(
        max_visible_envs=cfg_max_visible_envs,
        open_browser=False,
        randomly_sample_visible_envs=False,
    )
    visualizer = viser_visualizer.ViserVisualizer(cfg)
    visualizer.backend = SimpleNamespace(model="dummy-model")
    sim = Mock(stage=None, get_scene_data_provider=Mock(return_value=SimpleNamespace(num_envs=8)))
    BaseVisualizer.initialize(visualizer, sim, cameras=[])
    visualizer._create_viewer(record_to_viser="record.viser", metadata={"num_envs": 8})

    assert captured["set_model"] == "dummy-model"
    assert captured["init"]["bind_address"] == cfg.bind_address
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)
    apply_pose.assert_called_once_with((cfg.eye, cfg.lookat))


@pytest.mark.parametrize(
    ("cfg_max_visible_envs", "expected_visible"),
    [
        (None, None),
        (0, []),
        (3, [0, 1, 2]),
    ],
)
def test_rerun_visualizer_initialize_applies_visible_worlds_and_world_offsets(
    monkeypatch: pytest.MonkeyPatch,
    web_backend,
    cfg_max_visible_envs: int | None,
    expected_visible: list[int] | None,
):
    captured = {}

    class _FakeNewtonViewerRerun:
        def __init__(self, **kwargs):
            captured["streaming_view"] = kwargs["streaming_view"]

        def set_model(self, model: Any) -> None:
            captured["set_model"] = model

        def set_visible_worlds(self, worlds) -> None:
            captured["visible_worlds"] = worlds

        def set_world_offsets(self, spacing) -> None:
            captured["set_world_offsets"] = tuple(spacing)

        def close(self) -> None:
            captured["closed"] = True

    monkeypatch.setattr(rerun_visualizer, "NewtonViewerRerun", _FakeNewtonViewerRerun)
    monkeypatch.setattr(
        rerun_visualizer, "_ensure_rerun_server", lambda **kwargs: ("rerun+http://127.0.0.1:9876/proxy", False)
    )
    monkeypatch.setattr(rerun_visualizer, "_open_rerun_web_viewer", lambda *args, **kwargs: None)
    apply_pose = Mock()
    monkeypatch.setattr(rerun_visualizer.RerunVisualizer, "_apply_camera_pose", apply_pose)

    cfg = RerunVisualizerCfg(
        open_browser=False,
        max_visible_envs=cfg_max_visible_envs,
        randomly_sample_visible_envs=False,
    )
    visualizer = rerun_visualizer.RerunVisualizer(cfg)
    visualizer.initialize(web_backend, cameras=[])

    assert captured["streaming_view"] is False
    assert captured["set_model"] is web_backend.get_or_create_backend.return_value.model
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)
    apply_pose.assert_called_once_with((cfg.eye, cfg.lookat))
    replacement = SimpleNamespace(model=SimpleNamespace(body_label=["/Replacement"]))
    web_backend.get_or_create_backend.return_value = replacement
    visualizer.reset()
    assert visualizer.backend is replacement
    web_backend.get_or_create_backend.assert_called_with(visualizer.newton_cfg)
    assert captured["set_model"] is replacement.model
    assert captured["visible_worlds"] == expected_visible


def test_kit_visualizer_default_camera_source_does_not_require_camera_prim(monkeypatch: pytest.MonkeyPatch):
    """Default ``--viz kit`` should work for envs without a camera prim."""

    class _FakeViewportApi:
        def __init__(self):
            self.set_active_camera_calls = []

        def get_active_camera(self):
            return "/OmniverseKit_Persp"

        def set_active_camera(self, camera_path):
            self.set_active_camera_calls.append(camera_path)

    class _FakeViewportWindow:
        def __init__(self):
            self.viewport_api = _FakeViewportApi()

    class _FakeStage:
        def GetPrimAtPath(self, path):
            raise AssertionError(f"default Kit visualizer should not look up camera prims: {path}")

    viewport_window = _FakeViewportWindow()
    viewport_utility = type(
        "ViewportUtility",
        (),
        {
            "create_viewport_window": staticmethod(lambda **kwargs: viewport_window),
            "get_active_viewport_window": staticmethod(lambda: viewport_window),
        },
    )
    monkeypatch.setitem(sys.modules, "omni", type(sys)("omni"))
    monkeypatch.setitem(sys.modules, "omni.kit", type(sys)("omni.kit"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", type(sys)("omni.kit.viewport"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", viewport_utility)
    monkeypatch.setitem(sys.modules, "omni.ui", type("OmniUi", (), {"DockPosition": object})())

    applied_camera_poses = []
    monkeypatch.setattr(kit_visualizer.KitVisualizer, "_write_desktop_entry", lambda self: None)
    monkeypatch.setattr(
        kit_visualizer.KitVisualizer,
        "_set_viewport_camera",
        lambda self, eye, target: applied_camera_poses.append((tuple(eye), tuple(target))),
    )

    cfg = KitVisualizerCfg()
    visualizer = kit_visualizer.KitVisualizer(cfg)
    monkeypatch.setattr(SimulationContext, "_instance", SimpleNamespace(stage=_FakeStage()))
    visualizer._runtime_headless = False

    visualizer._setup_viewport()

    assert not cfg.streaming_view
    assert applied_camera_poses == [(cfg.eye, cfg.lookat)]
    assert viewport_window.viewport_api.set_active_camera_calls == []
    assert visualizer._controlled_camera_path == "/OmniverseKit_Persp"


def test_kit_visualizer_default_camera_source_accepts_set_camera_view(monkeypatch: pytest.MonkeyPatch):
    """Default Kit visualizer camera follows SimulationContext set_camera_view updates."""
    applied_camera_poses = []
    monkeypatch.setattr(
        kit_visualizer.KitVisualizer,
        "_set_viewport_camera",
        lambda self, eye, target: applied_camera_poses.append((tuple(eye), tuple(target))),
    )

    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg())
    visualizer._is_initialized = True

    visualizer.set_camera_view((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))

    assert applied_camera_poses == [((1.0, 2.0, 3.0), (0.0, 0.0, 1.0))]


def test_kit_visualizer_set_viewport_camera_does_not_require_authored_coi(monkeypatch: pytest.MonkeyPatch):
    """Regression: ``_set_viewport_camera`` must not feed an unauthored ``omni:kit:centerOfInterest`` into
    ``ViewportCameraState.set_position_world``.

    A freshly-opened stage's default ``/OmniverseKit_Persp`` camera has no ``omni:kit:centerOfInterest`` attribute
    authored. ``ViewportCameraState.set_position_world(..., rotate=True)`` reads that attribute as ``None`` and
    crashes inside ``Matrix4d.Transform`` (the boost binding rejects ``NoneType``). ``_set_viewport_camera`` must
    therefore use ``rotate=False`` for the eye set; the follow-up ``set_target_world(..., rotate=True)`` performs
    the look-at rotation and authors the COI as a side effect.

    The fake ``ViewportCameraState`` here mirrors that boost-binding behavior: ``set_position_world(..., rotate=True)``
    raises ``TypeError``, so the old call path would surface inside ``_set_viewport_camera`` exactly as it did in
    production.
    """

    class _FakeViewportApi:
        def get_active_camera(self):
            return "/OmniverseKit_Persp"

    state_holder: dict[str, Any] = {}

    class _FakeCameraState:
        def __init__(self, camera_path: str, viewport_api):
            self.position_calls: list[tuple[Any, bool]] = []
            self.target_calls: list[tuple[Any, bool]] = []
            state_holder["state"] = self

        def set_position_world(self, world_position, rotate):
            if rotate:
                raise TypeError(
                    "Python argument types in Matrix4d.Transform(Matrix4d, NoneType) did not match C++ signature"
                )
            self.position_calls.append((world_position, rotate))

        def set_target_world(self, world_target, rotate):
            self.target_calls.append((world_target, rotate))

    camera_state_module = type(sys)("omni.kit.viewport.utility.camera_state")
    camera_state_module.ViewportCameraState = _FakeCameraState

    monkeypatch.setitem(sys.modules, "omni", type(sys)("omni"))
    monkeypatch.setitem(sys.modules, "omni.kit", type(sys)("omni.kit"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport", type(sys)("omni.kit.viewport"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", type(sys)("omni.kit.viewport.utility"))
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility.camera_state", camera_state_module)

    cfg = KitVisualizerCfg()
    visualizer = kit_visualizer.KitVisualizer(cfg)
    visualizer._viewport_api = _FakeViewportApi()

    eye = (1.0, 2.0, 3.0)
    target = (4.0, 5.0, 6.0)

    visualizer._set_viewport_camera(eye, target)

    state = state_holder["state"]
    assert len(state.position_calls) == 1
    pos_arg, pos_rotate = state.position_calls[0]
    assert pos_rotate is False
    assert (float(pos_arg[0]), float(pos_arg[1]), float(pos_arg[2])) == eye

    assert len(state.target_calls) == 1
    tgt_arg, tgt_rotate = state.target_calls[0]
    assert tgt_rotate is True
    assert (float(tgt_arg[0]), float(tgt_arg[1]), float(tgt_arg[2])) == target


# ---------------------------------------------------------------------------
# Shared helpers for config-resolution and initialize_visualizers tests
# ---------------------------------------------------------------------------


class _FakeVisualizerCfg(VisualizerCfg):
    """Minimal visualizer config for testing initialize_visualizers."""

    def __init__(self, visualizer_type: str, *, fail_construct: bool = False, fail_init: bool = False):
        super().__init__()
        self.visualizer_type = visualizer_type
        self.class_type = (
            self._raise_construction_error
            if fail_construct
            else (_FailingInitVisualizer if fail_init else _FakeVisualizer)
        )

    @staticmethod
    def _raise_construction_error(_cfg):
        raise RuntimeError("construction failed")


class _FailingInitVisualizer(_FakeVisualizer):
    def initialize(self, sim, *, cameras):
        raise RuntimeError("init failed")


@pytest.mark.parametrize("fail_construct", [False, True])
def test_visualizer_construction_precedes_initialization_and_happens_once(monkeypatch, fail_construct):
    import isaaclab.sim.simulation_context as context_module

    seen = []
    cfg = _FakeVisualizerCfg("kit")
    cfg.cloning_contexts = (object,)
    cfg.class_type = lambda actual: seen.append(actual) or _FakeVisualizer(actual)
    settings = {}
    physics_cfg = SimpleNamespace(class_type=Mock(), dt=0.01)
    monkeypatch.setattr(context_module, "_resolve_physics_cfg", lambda cfg, use_isaac_sim: physics_cfg)
    monkeypatch.setattr(context_module, "has_kit", lambda: False)
    monkeypatch.setattr(context_module, "SceneDataProvider", lambda backend: _FakeProvider())
    monkeypatch.setattr(
        context_module, "get_settings_manager", lambda: SimpleNamespace(get=settings.get, set=settings.__setitem__)
    )
    monkeypatch.setattr(SimulationContext, "_init_usd_physics_scene", lambda self: None)
    monkeypatch.setattr(SimulationContext, "_instance", None)
    cfgs = [cfg, _FakeVisualizerCfg("kit", fail_construct=True)] if fail_construct else [cfg]
    sim_cfg = context_module.SimulationCfg(device="cpu", visualizer_cfgs=cfgs)
    cfg = sim_cfg.visualizer_cfgs[0]
    if fail_construct:
        with pytest.raises(RuntimeError, match="construction failed"):
            SimulationContext(sim_cfg)
        ctx = SimulationContext.instance()
    else:
        ctx = SimulationContext(sim_cfg)

    assert seen == [cfg]
    assert ctx._pending_visualizers[0].cfg is cfg
    assert ctx._render_context.clone_contexts == {object}
    assert ctx.get_clone_plan() is None
    assert not ctx._visualizers
    assert ctx.requires_usd_stage

    visualizer = ctx._pending_visualizers[0]
    if not fail_construct:
        ctx._clone_plan = SimpleNamespace(env_template="/Scenes/world_{}")
        camera = SimpleNamespace(cfg=SimpleNamespace(prim_path="/Scenes/world_[^/]+/Camera", data_types=["rgb"]))
        ctx._scene_data_provider.get_camera_sensors = Mock(return_value={"camera": camera})
        source = SceneCameraCfg(prim_path="{ENV_REGEX_NS}/Camera")
        perspective = PerspectiveCameraCfg(eye=(1.0, 2.0, 3.0))
        cfg.cameras, cfg.streaming_view = [source, perspective], True
        ctx.initialize_visualizers()
        ctx.initialize_visualizers()
        assert visualizer._cameras == [camera, perspective]
        assert cfg.cameras == [source, perspective]
        assert source.prim_path == "{ENV_REGEX_NS}/Camera"
        assert not {"_clone_plan", "_scene_stage", "_get_backend"}.intersection(vars(visualizer))
        ctx._scene_data_provider.get_camera_sensors.assert_called_once_with()
        assert visualizer._sim is ctx
        assert seen == [cfg]
        assert ctx._visualizers == [visualizer]
    assert visualizer.close_calls == 0
    SimulationContext.clear_instance()
    assert visualizer.close_calls == 1


def _make_context_with_settings(
    settings: dict,
    visualizer_cfgs=None,
    default_visualizer_cfg=None,
    *,
    has_gui: bool = False,
    has_offscreen_render: bool = False,
):
    """Build a minimal SimulationContext for visualizer construction, initialization, and rendering checks."""
    cfg = type(
        "Cfg",
        (),
        {
            "visualizer_cfgs": [] if visualizer_cfgs is None else visualizer_cfgs,
            "default_visualizer_cfg": default_visualizer_cfg,
            "physics": type("PhysicsCfg", (), {"dt": 0.01})(),
            "dt": 0.01,
            "render_interval": 1,
        },
    )()
    ctx = object.__new__(SimulationContext)
    ctx.cfg = cfg
    ctx._has_gui = has_gui
    ctx._has_offscreen_render = has_offscreen_render
    ctx._xr_enabled = False
    ctx._pending_camera_view = None
    ctx._render_generation = 0
    ctx._visualizers = []
    ctx._pending_visualizers = []
    ctx._render_context = SimpleNamespace(clone_contexts=set())
    ctx.physics_manager = SimpleNamespace(register_callback=Mock())
    ctx._scene_data_provider = _FakeProvider()
    ctx.requires_usd_stage = False
    ctx.requires_newton_model = False
    ctx._clone_plan = None
    ctx.stage = None
    ctx._viz_dt = 0.01
    ctx.get_setting = lambda name: settings.get(name)
    return ctx


def test_default_visualizer_cfg_applies_to_cli_created_configs():
    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    default_cfg = VisualizerCfg(
        background_color=(0.1, 0.2, 0.3),
        streaming_sensor_prim_path="/World/envs/*/Camera",
        window=WindowCfg(size=(640, 480), fps=60),
    )
    visualizer_cfgs = resolve_visualizer_cfgs([], ["newton_gl", "newton_rtx"])
    ctx = _make_context_with_settings({}, visualizer_cfgs=visualizer_cfgs, default_visualizer_cfg=default_cfg)

    ctx._create_visualizers()
    cfgs = [visualizer.cfg for visualizer in ctx._pending_visualizers]

    assert len(cfgs) == 2
    assert isinstance(cfgs[0], NewtonVisualizerCfg)
    assert cfgs[0].background_color == (0.1, 0.2, 0.3)
    assert cfgs[0].streaming_sensor_prim_path == "/World/envs/*/Camera"
    assert cfgs[0].window == cfgs[1].window == default_cfg.window
    cfgs[0].window.fps = 20
    assert cfgs[1].window.fps == default_cfg.window.fps == 60


def test_cli_type_newton_rtx_resolves_to_newton_rtx_visualizer_cfg():
    """Requesting 'newton_rtx' via CLI resolves to a NewtonRTXVisualizerCfg."""
    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    cfgs = resolve_visualizer_cfgs([], ["newton_rtx"])

    assert len(cfgs) == 1
    assert isinstance(cfgs[0], NewtonRTXVisualizerCfg)


def test_default_visualizer_cfg_applies_to_explicit_visualizer_cfgs():
    """default_visualizer_cfg fills in env-level hints (eye, lookat) on explicit cfgs.

    When visualizer_cfgs is set directly (e.g. for video recording), fields that are
    still at the backend class's own factory default are overridden by default_visualizer_cfg
    so the env's intended camera position is respected.
    """
    settings = {}
    default_cfg = KitVisualizerCfg(
        eye=(8.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.5),
        streaming_sensor_prim_path="/World/envs/*/Camera",
        window=WindowCfg(size=(640, 480), fps=60),
    )
    # Explicit Newton cfg with only the window size customized; eye/lookat at class defaults.
    explicit_cfg = NewtonGLVisualizerCfg(window=WindowCfg(size=(320, 240)))
    rtx_cfg = NewtonRTXVisualizerCfg(window=WindowCfg(fps=20))
    ctx = _make_context_with_settings(
        settings, visualizer_cfgs=[explicit_cfg, rtx_cfg], default_visualizer_cfg=default_cfg
    )

    ctx._create_visualizers()
    cfgs = [visualizer.cfg for visualizer in ctx._pending_visualizers]

    assert len(cfgs) == 2
    # env-level hints applied (were at class defaults on explicit_cfg)
    assert cfgs[0].eye == (8.0, 0.0, 5.0)
    assert cfgs[0].lookat == (0.0, 0.0, 0.5)
    assert cfgs[0].streaming_sensor_prim_path == "/World/envs/*/Camera"
    # user-customized fields preserved
    assert cfgs[0].window.size == (320, 240)
    assert cfgs[0].window.fps == 60
    assert cfgs[1].window == WindowCfg(size=(640, 480), fps=20)
    assert cfgs[0].class_type.__name__ == "NewtonGLVisualizer"
    assert cfgs[0].visualizer_type == "newton_gl"
    assert cfgs[0].cloning_contexts == NewtonGLVisualizerCfg().cloning_contexts


def test_default_visualizer_cfg_does_not_override_explicitly_customized_fields():
    """Explicitly-set fields on a visualizer cfg beat default_visualizer_cfg."""
    settings = {}
    default_cfg = VisualizerCfg(eye=(8.0, 0.0, 5.0))
    # eye explicitly set — should NOT be overridden by default_cfg
    explicit_cfg = NewtonGLVisualizerCfg(eye=(1.0, 2.0, 3.0))
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[explicit_cfg], default_visualizer_cfg=default_cfg)

    ctx._create_visualizers()
    cfgs = [visualizer.cfg for visualizer in ctx._pending_visualizers]

    assert cfgs[0].eye == (1.0, 2.0, 3.0)


def test_is_rendering_true_when_only_cfg_visualizer_is_set():
    cfg_visualizer = VisualizerCfg(visualizer_type="newton_gl")
    settings = {
        "/isaaclab/render/rtx_sensors": False,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[cfg_visualizer])
    assert ctx.is_rendering is True


def test_is_rendering_false_when_only_cfg_visualizer_is_headless():
    """A capture-only headless visualizer must not trigger continuous rendering."""
    cfg_visualizer = VisualizerCfg(visualizer_type="kit", headless=True)
    settings = {
        "/isaaclab/render/rtx_sensors": False,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[cfg_visualizer])
    assert ctx.is_rendering is False


def test_explicit_missing_package_raises(monkeypatch: pytest.MonkeyPatch):
    """Requesting a valid type whose package is not installed raises RuntimeError."""
    # Force import to fail for the rerun visualizer module
    import importlib

    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    real_import = importlib.import_module

    def _failing_import(name, *args, **kwargs):
        if "isaaclab_visualizers.rerun" in name:
            raise ModuleNotFoundError("No module named 'isaaclab_visualizers.rerun'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _failing_import)

    with pytest.raises(RuntimeError, match="rerun"):
        resolve_visualizer_cfgs([], ["rerun"])


def test_visualizer_init_keeps_requirements_published_before_reset():
    """Scene and renderer requirements survive visualizer initialization.

    The scene and the renderers publish their requirements while the scene is built, which happens
    before the first reset initializes visualizers, so a stage-only visualizer must not clear the
    Newton model requirement someone else already asked for.
    """

    settings = {}
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[_FakeVisualizerCfg("kit")])
    ctx.requires_newton_model = True

    ctx._create_visualizers()
    ctx.initialize_visualizers()

    assert ctx.requires_newton_model
    assert ctx.requires_usd_stage


@pytest.mark.parametrize("fail_construct", [False, True])
def test_visualizer_failures_propagate_and_retain_constructed_instances(fail_construct):
    """Cfg-requested failures propagate naturally; completed instances stay owned until explicit teardown."""
    good_cfg = _FakeVisualizerCfg("kit")
    failing_cfg = _FakeVisualizerCfg("newton_gl", fail_construct=fail_construct, fail_init=not fail_construct)
    ctx = _make_context_with_settings({}, visualizer_cfgs=[good_cfg, failing_cfg])

    with pytest.raises(RuntimeError, match="construction failed" if fail_construct else "init failed"):
        ctx._create_visualizers()
        ctx.initialize_visualizers()
    assert len(ctx._pending_visualizers) == 1
    assert len(ctx._visualizers) == (0 if fail_construct else 1)
    assert all(viz.close_calls == 0 for viz in ctx._visualizers + ctx._pending_visualizers)
    assert ctx._scene_data_provider is not None


def test_explicit_type_matches_existing_cfg():
    """Requesting 'newton_gl' via CLI when cfg.visualizer_cfgs already has a customized 'newton_gl'
    config selects and returns that exact instance, rather than discarding it and building a
    fresh default -- exercising the branch that filters pre-existing cfgs by type."""
    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    existing_cfg = NewtonGLVisualizerCfg(background_color=(0.4, 0.5, 0.6))

    cfgs = resolve_visualizer_cfgs([existing_cfg, KitVisualizerCfg()], ["newton_gl"])

    assert len(cfgs) == 1
    assert cfgs[0] is existing_cfg
    assert cfgs[0].background_color == (0.4, 0.5, 0.6)


def test_explicit_existing_cfg_plus_failing_requested_type_raises_for_the_failure(monkeypatch: pytest.MonkeyPatch):
    """A pre-existing cfg satisfies one requested type; a second requested type that cannot be
    resolved still raises, exercising the branch that extends pre-existing cfgs with freshly-created
    defaults for the remaining requested types."""
    import importlib

    from isaaclab.visualizers.visualizer_cfg import resolve_visualizer_cfgs

    real_import = importlib.import_module
    requested = []

    def _failing_import(name, *args, **kwargs):
        requested.append(name)
        if name == "isaaclab_visualizers.rerun":
            raise ModuleNotFoundError("No module named 'isaaclab_visualizers.rerun'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _failing_import)

    with pytest.raises(RuntimeError) as exc_info:
        resolve_visualizer_cfgs([_FakeVisualizerCfg("kit")], ["kit", "rerun"])
    # 'kit' was satisfied by the pre-existing cfg, so only the unresolved type is constructed and reported.
    assert "'rerun'" in str(exc_info.value)
    assert requested == ["isaaclab_visualizers.rerun"]


# ---------------------------------------------------------------------------
# RerunVisualizer streaming-view tests
# ---------------------------------------------------------------------------


def test_rerun_streaming_blueprint_includes_spatial2d():
    """The streaming panel uses a 2D blueprint rather than replacing it with a 3D camera."""
    import rerun.blueprint as rrb

    blueprint_viewer = object.__new__(rerun_visualizer.NewtonViewerRerun)
    blueprint_viewer._streaming_view_active = True
    blueprint_viewer._live_plot_manager_names = []
    blueprint_viewer._camera_pose = None

    bp = blueprint_viewer._get_blueprint()

    # The root container wraps a Spatial2DView for the streaming panel.
    contents = bp.root_container.contents
    flat = []
    stack = list(contents)
    while stack:
        item = stack.pop()
        flat.append(item)
        sub = getattr(item, "contents", None)
        if sub:
            stack.extend(sub)
    assert any(isinstance(item, rrb.Spatial2DView) for item in flat), (
        "_get_blueprint with streaming_view_active=True must include a Spatial2DView panel"
    )
