# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SimulationContext visualizer orchestration."""

from __future__ import annotations

import contextlib
import importlib
import sys
import types
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import pytest
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

import isaaclab.sim.simulation_context as context_module
from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim.simulation_context import SimulationContext
from isaaclab.visualizers.base_visualizer import BaseVisualizer
from isaaclab.visualizers.visualizer_cfg import VisualizerCfg

pytestmark = [pytest.mark.integration, pytest.mark.rendering]


# ---------------------------------------------------------------------------
# Fakes and context builders
# ---------------------------------------------------------------------------


class _FakePhysicsManager:
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeProvider:
    """Fake SceneDataProvider for tests; only provides what visualizers read."""

    def __init__(self, num_envs: int = 0):
        self._num_envs = num_envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def usd_stage(self):
        return None

    def get_camera_transforms(self):
        return None


class _FakeVisualizer(BaseVisualizer):
    """Minimal visualizer for orchestration tests."""

    def __init__(
        self,
        cfg=None,
        *,
        running=True,
        closed=False,
        rendering_paused=False,
        training_paused_steps=0,
        raises_on_step=False,
        requires_forward=False,
        pumps_app_update=False,
    ):
        super().__init__(cfg or VisualizerCfg())
        self._running = running
        self._closed = closed
        self._rendering_paused = rendering_paused
        self._training_paused_steps = training_paused_steps
        self._raises_on_step = raises_on_step
        self._requires_forward = requires_forward
        self._pumps_app_update = pumps_app_update
        self.step_calls = []
        self.close_calls = 0

    def initialize(self, provider):
        self._set_scene_data_provider(provider)
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


class _LivePlotVisualizer(_FakeVisualizer):
    def __init__(self, *, enable_live_plots: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.cfg = VisualizerCfg(enable_live_plots=enable_live_plots)

    def supports_live_plots(self):
        return True


class _FailingInitVisualizer(_FakeVisualizer):
    def initialize(self, provider):
        raise RuntimeError("init failed")


class _FakeVisualizerCfg:
    """Minimal visualizer config whose ``class_type`` constructs a fake visualizer."""

    cloning_contexts = ()

    def __init__(self, visualizer_type: str, *, fail_construct: bool = False, fail_init: bool = False):
        self.visualizer_type = visualizer_type
        self.class_type = (
            self._raise_construction_error
            if fail_construct
            else (_FailingInitVisualizer if fail_init else _FakeVisualizer)
        )

    @staticmethod
    def _raise_construction_error(_cfg):
        raise RuntimeError("construction failed")


def _cli_settings(types: str = "", *, explicit: bool = False, disable_all: bool = False) -> dict:
    """Settings the app launcher writes for ``--visualizer`` (rtx sensors off, no env-count override)."""
    return {
        "/isaaclab/visualizer/types": types,
        "/isaaclab/visualizer/explicit": explicit,
        "/isaaclab/visualizer/disable_all": disable_all,
        "/isaaclab/visualizer/max_visible_envs": None,
        "/isaaclab/render/rtx_sensors": False,
    }


def _make_context(
    visualizers=(),
    *,
    settings: dict | None = None,
    visualizer_cfgs=None,
    default_visualizer_cfg=None,
) -> SimulationContext:
    """Build a SimulationContext without running its constructor, wired for visualizer orchestration.

    Centralises the ``object.__new__`` construction so new internal attributes only need to be added
    in one place when the production code changes.
    """
    settings = _cli_settings() if settings is None else settings
    ctx = object.__new__(SimulationContext)
    ctx.cfg = SimpleNamespace(
        visualizer_cfgs=visualizer_cfgs,
        default_visualizer_cfg=default_visualizer_cfg,
        physics=SimpleNamespace(dt=0.01),
        dt=0.01,
        render_interval=1,
    )
    ctx._has_gui = False
    ctx._has_offscreen_render = False
    ctx._xr_enabled = False
    ctx._pending_camera_view = None
    ctx._render_generation = 0
    ctx._visualizers = list(visualizers)
    ctx._pending_visualizers = []
    ctx._render_context = SimpleNamespace(clone_contexts=set())
    ctx._scene_data_provider = _FakeProvider()
    ctx.physics_manager = _FakePhysicsManager()
    ctx.vis_marker_registry = VisMarkerRegistry()
    ctx.requires_usd_stage = False
    ctx.requires_newton_model = False
    ctx._clone_plan = None
    ctx._viz_dt = 0.01
    ctx.get_setting = settings.get
    return ctx


# ---------------------------------------------------------------------------
# update_visualizers / reset
# ---------------------------------------------------------------------------


def test_web_visualizer_cfgs_do_not_open_browser_by_default():
    assert RerunVisualizerCfg().open_browser is False
    assert ViserVisualizerCfg().open_browser is False


def test_update_visualizers_forwards_only_when_a_visualizer_requires_it():
    forwarding = _FakeVisualizer(requires_forward=True)
    plain = _FakeVisualizer()

    ctx = _make_context([forwarding, plain])
    ctx.update_visualizers(0.1)
    assert ctx.physics_manager.forward_calls == 1
    assert forwarding.step_calls == plain.step_calls == [0.1]

    ctx = _make_context([plain])
    ctx.update_visualizers(0.1)
    assert ctx.physics_manager.forward_calls == 0


def test_update_visualizers_steps_pauses_and_removes(caplog):
    """Closed, stopped, and failing visualizers are removed; paused ones idle; the rest step with ``dt``."""
    closed_viz = _FakeVisualizer(closed=True)
    stopped_viz = _FakeVisualizer(running=False)
    failing_viz = _FakeVisualizer(raises_on_step=True)
    paused_viz = _FakeVisualizer(rendering_paused=True)
    paused_pumping_viz = _FakeVisualizer(rendering_paused=True, pumps_app_update=True)
    training_paused_viz = _FakeVisualizer(training_paused_steps=1)
    healthy_viz = _FakeVisualizer()
    ctx = _make_context(
        [closed_viz, stopped_viz, failing_viz, paused_viz, paused_pumping_viz, training_paused_viz, healthy_viz]
    )

    with caplog.at_level("ERROR"):
        ctx.update_visualizers(0.1)

    assert ctx._visualizers == [paused_viz, paused_pumping_viz, training_paused_viz, healthy_viz]
    assert [viz.close_calls for viz in (closed_viz, stopped_viz, failing_viz)] == [1, 1, 1]
    assert all(viz.close_calls == 0 for viz in ctx._visualizers)
    # a paused visualizer keeps its event loop alive with step(0.0) unless it would pump the Kit app
    assert paused_viz.step_calls == [0.0]
    assert paused_pumping_viz.step_calls == []
    # training pause spins step(0.0) until released, then the real step follows
    assert training_paused_viz.step_calls == [0.0, 0.1]
    assert healthy_viz.step_calls == [0.1]
    assert any("Error stepping visualizer" in r.message for r in caplog.records)


@pytest.mark.parametrize("enable_live_plots", [True, False])
def test_update_visualizers_dispatches_marker_callbacks_for_live_plots(enable_live_plots):
    """Live-plot panels share the marker registry, so dispatch follows the live-plot flag, not marker support."""
    dispatched = []
    ctx = _make_context([_LivePlotVisualizer(enable_live_plots=enable_live_plots)])
    ctx.vis_marker_registry.add_callback("probe", dispatched.append)

    ctx.update_visualizers(0.1)

    assert len(dispatched) == (1 if enable_live_plots else 0)


def test_newton_visualizer_is_initialized_and_rebound_before_capture():
    created = []
    reset_calls = []
    camera_calls = []

    class _Cfg:
        cloning_contexts = ()

        def __init__(self, visualizer_type, enable_picking=False):
            self.visualizer_type = visualizer_type
            self.enable_picking = enable_picking
            self.headless = False
            self.class_type = self._construct

        def _construct(self, cfg):
            viz = _FakeVisualizer(cfg)
            viz.initialize = lambda _provider: created.append(cfg.visualizer_type)
            viz.reset = lambda soft: reset_calls.append((cfg.visualizer_type, soft))
            viz.set_camera_view = lambda eye, target: camera_calls.append((cfg.visualizer_type, eye, target))
            return viz

    ctx = _make_context(visualizer_cfgs=[_Cfg("newton_gl", True), _Cfg("newton_rtx", True), _Cfg("rerun")])
    ctx._create_visualizers()
    eye, target = (1.0, 2.0, 3.0), (0.0, 0.0, 0.0)
    ctx.set_camera_view(eye, target)
    ctx._prepare_newton_visualizer_for_capture()
    assert created == ["newton_gl", "newton_rtx"]

    ctx.initialize_visualizers()
    ctx._prepare_newton_visualizer_for_capture()

    assert created == ["newton_gl", "newton_rtx", "rerun"]
    assert len(ctx._visualizers) == 3
    assert camera_calls == [(name, eye, target) for name in ("newton_gl", "newton_rtx", "rerun")]
    assert ctx._pending_camera_view is None
    assert reset_calls == [
        ("newton_gl", False),
        ("newton_rtx", False),
        ("newton_gl", False),
        ("newton_rtx", False),
    ]


def test_reset_initializes_visualizers_before_playing_timeline():
    """Initial visualizers must see the PhysX views created by reset before play() pumps timeline events."""
    events: list[str] = []
    ctx = object.__new__(SimulationContext)
    ctx._visualizers = []

    class _PhysicsManager:
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

    assert events == ["reset:False", "initialize_visualizers", "finalize_consumers:1:True", "play"]
    assert ctx.is_playing()
    assert not ctx.is_stopped()


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        ({"types": "newton,kit"}, []),
        ("", []),
        ("kit, newton_gl rerun", ["kit", "newton_gl", "rerun"]),
    ],
    ids=["non_string", "empty", "mixed_separators"],
)
def test_get_cli_visualizer_types(setting, expected):
    ctx = _make_context(settings={"/isaaclab/visualizer/types": setting})
    assert ctx._get_cli_visualizer_types() == expected


@pytest.mark.parametrize("fail_construct", [False, True])
def test_visualizer_construction_precedes_initialization_and_happens_once(monkeypatch, fail_construct):
    seen = []
    cfg = _FakeVisualizerCfg("kit")
    cfg.cloning_contexts = (object,)
    cfg.class_type = lambda actual: seen.append(actual) or _FakeVisualizer(actual)
    settings = _cli_settings()
    physics_cfg = SimpleNamespace(class_type=Mock(), dt=0.01)
    monkeypatch.setattr(context_module, "_resolve_physics_cfg", lambda cfg, use_isaac_sim: physics_cfg)
    monkeypatch.setattr(context_module, "has_kit", lambda: False)
    monkeypatch.setattr(context_module, "SceneDataProvider", lambda backend: _FakeProvider())
    monkeypatch.setattr(
        context_module.SettingsManager,
        "instance",
        lambda: SimpleNamespace(get=settings.get, set_bool=settings.__setitem__),
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
        ctx.initialize_visualizers()
        ctx.initialize_visualizers()
        assert seen == [cfg]
        assert ctx._visualizers == [visualizer]
    assert visualizer.close_calls == 0
    SimulationContext.clear_instance()
    assert visualizer.close_calls == 1


@pytest.mark.parametrize(
    ("types", "expected_cls", "deprecated"),
    [("newton_rtx", NewtonRTXVisualizerCfg, False), ("newton", NewtonGLVisualizerCfg, True)],
)
def test_cli_type_resolves_to_default_cfg(types, expected_cls, deprecated):
    """CLI types resolve to their backend cfg; the deprecated ``newton`` alias warns and maps to ``newton_gl``."""
    ctx = _make_context(settings=_cli_settings(types, explicit=True))

    warns = (
        pytest.warns(DeprecationWarning, match="newton.*deprecated.*newton_gl")
        if deprecated
        else contextlib.nullcontext()
    )
    with warns:
        cfgs = ctx._create_default_visualizer_configs([types])

    assert len(cfgs) == 1
    assert isinstance(cfgs[0], expected_cls)


def test_default_visualizer_cfg_applies_to_cli_created_configs():
    default_cfg = VisualizerCfg(
        background_color=(0.1, 0.2, 0.3),
        streaming_cam_target_prim_path="/World/envs/*/Object",
        streaming_cam_eye=(1.0, -1.0, 0.5),
    )
    ctx = _make_context(settings=_cli_settings("newton_gl", explicit=True), default_visualizer_cfg=default_cfg)

    cfgs = ctx._resolve_visualizer_cfgs()

    assert len(cfgs) == 1
    assert isinstance(cfgs[0], NewtonGLVisualizerCfg)
    assert cfgs[0].background_color == (0.1, 0.2, 0.3)
    assert cfgs[0].streaming_cam_target_prim_path == "/World/envs/*/Object"
    assert cfgs[0].streaming_cam_eye == (1.0, -1.0, 0.5)


def test_default_visualizer_cfg_fills_only_untouched_fields_of_explicit_cfgs():
    """default_visualizer_cfg fills env-level hints on explicit cfgs but never beats a caller's explicit value."""
    default_cfg = KitVisualizerCfg(
        eye=(8.0, 0.0, 5.0),
        lookat=(0.0, 0.0, 0.5),
        streaming_cam_target_prim_path="/World/envs/*/Object",
    )
    # eye and window size customised; lookat and streaming target still at the class defaults
    explicit_cfg = NewtonGLVisualizerCfg(window_width=320, window_height=240, eye=(1.0, 2.0, 3.0))
    ctx = _make_context(visualizer_cfgs=[explicit_cfg], default_visualizer_cfg=default_cfg)

    cfgs = ctx._resolve_visualizer_cfgs()

    assert len(cfgs) == 1
    assert cfgs[0].lookat == (0.0, 0.0, 0.5)
    assert cfgs[0].streaming_cam_target_prim_path == "/World/envs/*/Object"
    assert cfgs[0].eye == (1.0, 2.0, 3.0)
    assert (cfgs[0].window_width, cfgs[0].window_height) == (320, 240)
    assert cfgs[0].class_type.__name__ == "NewtonGLVisualizer"
    assert cfgs[0].visualizer_type == "newton_gl"


@pytest.mark.parametrize(
    ("cfg_attrs", "settings", "expected"),
    [
        ({"visualizer_type": "newton_rtx"}, _cli_settings(), True),
        ({"visualizer_type": "newton_gl"}, _cli_settings(), True),
        ({"visualizer_type": "kit", "headless": True}, _cli_settings(), False),
        ({"visualizer_type": "newton_gl"}, _cli_settings(explicit=True, disable_all=True), False),
    ],
    ids=["newton_rtx", "newton_gl", "headless_capture_only", "cli_disable_all"],
)
def test_is_rendering_follows_cfg_visualizers(cfg_attrs, settings, expected):
    ctx = _make_context(settings=settings, visualizer_cfgs=[SimpleNamespace(**cfg_attrs)])
    assert ctx.is_rendering is expected


@pytest.mark.parametrize("types", ["bogus_viz", "newton,bogus_viz"], ids=["unknown", "partially_unknown"])
def test_explicit_unknown_visualizer_type_raises(types):
    """Requesting an unknown visualizer type via CLI raises, even beside valid types."""
    ctx = _make_context(settings=_cli_settings(types, explicit=True))

    with pytest.raises(RuntimeError, match="bogus_viz"):
        ctx._create_visualizers()


def test_explicit_missing_package_raises(monkeypatch: pytest.MonkeyPatch):
    """Requesting a valid type whose package is not installed raises RuntimeError."""
    ctx = _make_context(settings=_cli_settings("rerun", explicit=True))
    real_import = importlib.import_module

    def _failing_import(name, *args, **kwargs):
        if "isaaclab_visualizers.rerun" in name:
            raise ImportError("No module named 'isaaclab_visualizers.rerun'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", _failing_import)

    with pytest.raises(RuntimeError, match="rerun"):
        ctx._create_visualizers()


def test_non_explicit_unknown_type_silently_skipped():
    """Without --visualizer flag, unknown types are silently skipped (no error)."""
    ctx = _make_context(settings=_cli_settings("bogus_viz"))

    ctx._create_visualizers()
    ctx.initialize_visualizers()

    assert ctx._visualizers == []


def test_visualizer_init_keeps_requirements_published_before_reset():
    """Scene and renderer requirements survive visualizer initialization.

    The scene and the renderers publish their requirements while the scene is built, which happens
    before the first reset initializes visualizers, so a stage-only visualizer must not clear the
    Newton model requirement someone else already asked for.
    """
    ctx = _make_context(settings=_cli_settings("kit", explicit=True), visualizer_cfgs=[_FakeVisualizerCfg("kit")])
    ctx.requires_newton_model = True

    ctx._create_visualizers()
    ctx.initialize_visualizers()

    assert ctx.requires_newton_model
    assert ctx.requires_usd_stage


@pytest.mark.parametrize("cli_explicit", [False, True])
@pytest.mark.parametrize("fail_construct", [False, True])
def test_visualizer_failures_propagate_and_retain_constructed_instances(cli_explicit, fail_construct):
    """Cfg-requested failures propagate naturally; completed instances stay owned until explicit teardown."""
    good_cfg = _FakeVisualizerCfg("kit")
    failing_cfg = _FakeVisualizerCfg("newton_gl", fail_construct=fail_construct, fail_init=not fail_construct)
    ctx = _make_context(
        settings=_cli_settings("kit newton_gl", explicit=cli_explicit), visualizer_cfgs=[good_cfg, failing_cfg]
    )

    with pytest.raises(RuntimeError, match="construction failed" if fail_construct else "init failed"):
        ctx._create_visualizers()
        ctx.initialize_visualizers()
    assert len(ctx._pending_visualizers) == 1
    assert len(ctx._visualizers) == (0 if fail_construct else 1)
    assert all(viz.close_calls == 0 for viz in ctx._visualizers + ctx._pending_visualizers)


# ---------------------------------------------------------------------------
# Backend visualizers driven by the context
# ---------------------------------------------------------------------------


class _DummySceneDataProvider:
    @property
    def num_envs(self) -> int:
        return 4

    @property
    def usd_stage(self):
        return None

    def get_camera_transforms(self):
        return {}


class _FakeNewtonManager:
    state_calls: list[object] = []

    @staticmethod
    def get_model():
        return "dummy-model"

    @classmethod
    def get_state(cls, scene_data_provider=None):
        cls.state_calls.append(scene_data_provider)
        return {"state_call": len(cls.state_calls)}

    @staticmethod
    def get_num_envs() -> int:
        return 1


@pytest.fixture
def fake_newton_manager(monkeypatch):
    import isaaclab_newton.physics as newton_physics

    _FakeNewtonManager.state_calls = []
    monkeypatch.setattr(newton_physics, "NewtonManager", _FakeNewtonManager)
    return _FakeNewtonManager


def test_viser_visualizer_initialize_and_step_uses_newton_manager_state(monkeypatch, fake_newton_manager):
    provider = _DummySceneDataProvider()

    class _DummyViewer:
        def __init__(self):
            self.calls = []

        def begin_frame(self, sim_time: float) -> None:
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state) -> None:
            self.calls.append(("log_state", state))

        def end_frame(self) -> None:
            self.calls.append(("end_frame",))

        def is_running(self) -> bool:
            return True

    viewer = _DummyViewer()

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        assert record_to_viser is None
        assert metadata == {"num_envs": provider.num_envs}
        self._viewer = viewer

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(cast(Any, provider))
    visualizer.step(0.25)

    assert visualizer.is_initialized
    assert fake_newton_manager.state_calls == [provider, provider]
    assert visualizer._sim_time == pytest.approx(0.25)
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[0][1] == pytest.approx(0.25)
    # log_state passes NewtonManager.get_state(provider) through as-is; no env_ids merged in.
    assert viewer.calls[1] == ("log_state", {"state_call": 2})
    assert viewer.calls[2] == ("end_frame",)


_VISIBLE_ENV_CASES = [(None, None), (0, []), (3, [0, 1, 2])]


@pytest.mark.parametrize(("cfg_max_visible_envs", "expected_visible"), _VISIBLE_ENV_CASES)
def test_viser_visualizer_create_viewer_applies_visible_worlds(monkeypatch, cfg_max_visible_envs, expected_visible):
    captured = {}

    class _FakeNewtonViewerViser:
        def __init__(self, *, port, bind_address, label, verbose, share, record_to_viser, metadata=None):
            captured["bind_address"] = bind_address

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
    monkeypatch.setattr(
        viser_visualizer.ViserVisualizer,
        "_resolve_initial_camera_pose",
        lambda self: ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_set_viser_camera_view", lambda self, pose: None)

    cfg = ViserVisualizerCfg(
        max_visible_envs=cfg_max_visible_envs,
        open_browser=False,
        randomly_sample_visible_envs=False,
    )
    visualizer = viser_visualizer.ViserVisualizer(cfg)
    visualizer._model = "dummy-model"
    visualizer._env_ids = None  # normally set by initialize() -> _compute_visualized_env_ids()
    visualizer._create_viewer(record_to_viser="record.viser", metadata={"num_envs": 8})

    assert captured["set_model"] == "dummy-model"
    assert captured["bind_address"] == cfg.bind_address
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


@pytest.mark.parametrize(("cfg_max_visible_envs", "expected_visible"), _VISIBLE_ENV_CASES)
def test_rerun_visualizer_initialize_applies_visible_worlds_and_world_offsets(
    monkeypatch, fake_newton_manager, cfg_max_visible_envs, expected_visible
):
    captured = {}

    class _FakeNewtonViewerRerun:
        def __init__(self, **kwargs):
            pass

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
    monkeypatch.setattr(
        rerun_visualizer.RerunVisualizer,
        "_resolve_initial_camera_pose",
        lambda self: ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0)),
    )
    monkeypatch.setattr(rerun_visualizer.RerunVisualizer, "_apply_camera_pose", lambda self, pose: None)

    cfg = RerunVisualizerCfg(
        open_browser=False,
        max_visible_envs=cfg_max_visible_envs,
        randomly_sample_visible_envs=False,
    )
    visualizer = rerun_visualizer.RerunVisualizer(cfg)
    visualizer.initialize(cast(Any, _DummySceneDataProvider()))

    assert captured["set_model"] == "dummy-model"
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


def _install_fake_omni_modules(monkeypatch, **leaf_modules) -> None:
    """Install ``omni``, ``omni.kit``, ``omni.kit.viewport`` stubs plus the given ``omni.kit.viewport.*`` leaves."""
    for name in ("omni", "omni.kit", "omni.kit.viewport", "omni.kit.viewport.utility"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    for name, module in leaf_modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_kit_visualizer_default_camera_source_does_not_require_camera_prim(monkeypatch: pytest.MonkeyPatch):
    """Default ``--viz kit`` should work for envs without a camera prim."""

    class _FakeViewportApi:
        def __init__(self):
            self.set_active_camera_calls = []

        def get_active_camera(self):
            return "/OmniverseKit_Persp"

        def set_active_camera(self, camera_path):
            self.set_active_camera_calls.append(camera_path)

    class _FakeStage:
        def GetPrimAtPath(self, path):
            raise AssertionError(f"default Kit visualizer should not look up camera prims: {path}")

    class _FakeProvider:
        def get_usd_stage(self):
            return _FakeStage()

    viewport_window = SimpleNamespace(viewport_api=_FakeViewportApi())
    viewport_utility = SimpleNamespace(
        create_viewport_window=lambda **kwargs: viewport_window,
        get_active_viewport_window=lambda: viewport_window,
    )
    _install_fake_omni_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "omni.kit.viewport.utility", viewport_utility)
    monkeypatch.setitem(sys.modules, "omni.ui", SimpleNamespace(DockPosition=object))

    applied_camera_poses = []
    monkeypatch.setattr(
        kit_visualizer.KitVisualizer,
        "_set_viewport_camera",
        lambda self, eye, target: applied_camera_poses.append((tuple(eye), tuple(target))),
    )

    cfg = KitVisualizerCfg()
    visualizer = kit_visualizer.KitVisualizer(cfg)
    visualizer._scene_data_provider = _FakeProvider()
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
    states: list[_FakeCameraState] = []

    class _FakeCameraState:
        def __init__(self, camera_path: str, viewport_api):
            self.position_calls: list[tuple[Any, bool]] = []
            self.target_calls: list[tuple[Any, bool]] = []
            states.append(self)

        def set_position_world(self, world_position, rotate):
            if rotate:
                raise TypeError(
                    "Python argument types in Matrix4d.Transform(Matrix4d, NoneType) did not match C++ signature"
                )
            self.position_calls.append((world_position, rotate))

        def set_target_world(self, world_target, rotate):
            self.target_calls.append((world_target, rotate))

    camera_state_module = types.ModuleType("omni.kit.viewport.utility.camera_state")
    camera_state_module.ViewportCameraState = _FakeCameraState
    _install_fake_omni_modules(monkeypatch, **{"omni.kit.viewport.utility.camera_state": camera_state_module})

    visualizer = kit_visualizer.KitVisualizer(KitVisualizerCfg())
    visualizer._viewport_api = SimpleNamespace(get_active_camera=lambda: "/OmniverseKit_Persp")

    eye = (1.0, 2.0, 3.0)
    target = (4.0, 5.0, 6.0)
    visualizer._set_viewport_camera(eye, target)

    (state,) = states
    ((pos_arg, pos_rotate),) = state.position_calls
    ((tgt_arg, tgt_rotate),) = state.target_calls
    assert pos_rotate is False
    assert tuple(float(v) for v in pos_arg) == eye
    assert tgt_rotate is True
    assert tuple(float(v) for v in tgt_arg) == target


def test_rerun_visualizer_setup_streaming_view_sets_flag_and_blueprint_includes_spatial2d(
    monkeypatch: pytest.MonkeyPatch,
):
    """_setup_streaming_view sets _streaming_view_active and _get_blueprint returns a Spatial2DView.

    Uses streaming_sensor_prim_path to avoid the create_visualizer_camera code path,
    so no actual Isaac Sim session is required.
    """
    camera_sensor = object()
    env_ids = [0, 1]

    camera_colorizer_mod = types.ModuleType("isaaclab.envs.utils.camera_colorizer")
    camera_colorizer_mod.SUPPORTED_GT_TYPES = {"rgb"}
    camera_colorizer_mod.sensor_keys_for_gt_types = lambda gt_types: list(gt_types)

    camera_view_mod = types.ModuleType("isaaclab.envs.utils.camera_view")
    camera_view_mod.VISUALIZER_TILED_CAMERA_MAX_TILES = 16
    camera_view_mod.create_visualizer_camera = None  # should not be called in this path
    camera_view_mod.find_camera_by_prim_path = lambda cameras, path, env_ids: camera_sensor
    camera_view_mod.resolve_streaming_envs = lambda num_envs, streaming_envs, max_tiles, sample_from: env_ids

    monkeypatch.setitem(sys.modules, "isaaclab.envs.utils.camera_colorizer", camera_colorizer_mod)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.utils.camera_view", camera_view_mod)

    class _FakeStreamingProvider:
        @property
        def num_envs(self) -> int:
            return 2

        def get_camera_sensors(self):
            return []

    # Build a minimal RerunVisualizer without triggering __init__.
    cfg = RerunVisualizerCfg(
        open_browser=False,
        streaming_view=True,
        streaming_sensor_prim_path="/World/envs/env_0/Camera",
    )
    visualizer = object.__new__(rerun_visualizer.RerunVisualizer)
    visualizer.cfg = cfg
    visualizer._scene_data_provider = _FakeStreamingProvider()
    visualizer._resolved_visible_env_ids = None
    visualizer._camera_env_indices = []
    visualizer._camera_sensor = None
    visualizer._camera_sensor_indices = []
    visualizer._camera_is_owned = False
    visualizer._streaming_view_active = False
    visualizer._streaming_camera_key = None
    visualizer._generated_camera_prim_paths = []
    fake_viewer = SimpleNamespace(_streaming_view_active=False, _live_plot_manager_names=[], _camera_pose=None)
    visualizer._viewer = fake_viewer

    visualizer._setup_streaming_view(num_envs=2)

    # Both the RerunVisualizer flag and the viewer flag must be set.
    assert visualizer._streaming_view_active is True
    assert fake_viewer._streaming_view_active is True
    assert visualizer._camera_sensor is camera_sensor
    assert visualizer._camera_sensor_indices == env_ids

    # The blueprint's root container wraps a Spatial2DView for the streaming panel.
    import rerun.blueprint as rrb

    blueprint_viewer = object.__new__(rerun_visualizer.NewtonViewerRerun)
    blueprint_viewer._streaming_view_active = True
    blueprint_viewer._live_plot_manager_names = []
    blueprint_viewer._camera_pose = None

    stack = list(blueprint_viewer._get_blueprint().root_container.contents)
    flat = []
    while stack:
        item = stack.pop()
        flat.append(item)
        stack.extend(getattr(item, "contents", None) or ())
    assert any(isinstance(item, rrb.Spatial2DView) for item in flat), (
        "_get_blueprint with streaming_view_active=True must include a Spatial2DView panel"
    )
