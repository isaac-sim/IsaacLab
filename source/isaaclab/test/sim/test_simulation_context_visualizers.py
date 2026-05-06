# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for SimulationContext visualizer orchestration."""

from __future__ import annotations

import sys
from typing import Any, cast

import isaaclab_visualizers.kit.kit_visualizer as kit_visualizer
import isaaclab_visualizers.newton.newton_visualization_markers as newton_markers
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import numpy as np
import pytest
import torch
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

from isaaclab.markers.vis_marker_registry import VisMarkerRegistry
from isaaclab.sim.simulation_context import SimulationContext


class _FakePhysicsManager:
    def __init__(self):
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class _FakeProvider:
    def __init__(self):
        self.update_calls = []

    def update(self):
        self.update_calls.append(True)


class _FakeVisualizer:
    """Minimal visualizer for orchestration tests."""

    def __init__(
        self,
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


def _make_context(visualizers, provider=None):
    ctx = object.__new__(SimulationContext)
    ctx._visualizers = list(visualizers)
    ctx._scene_data_provider = provider
    ctx.physics_manager = _FakePhysicsManager()
    ctx._visualizer_step_counter = 0
    return ctx


def test_update_scene_data_provider_forwards_and_updates_provider():
    provider = _FakeProvider()
    viz_a = _FakeVisualizer(env_ids=[0, 2], requires_forward=True)
    viz_b = _FakeVisualizer(env_ids=[2, 3])
    viz_c = _FakeVisualizer(env_ids=None)
    ctx = _make_context([viz_a, viz_b, viz_c], provider=provider)

    ctx.update_scene_data_provider()

    assert ctx.physics_manager.forward_calls == 1
    assert provider.update_calls == [True]
    assert ctx._visualizer_step_counter == 1


def test_update_scene_data_provider_force_forward_with_no_visualizers():
    provider = _FakeProvider()
    ctx = _make_context([], provider=provider)
    ctx.update_scene_data_provider(force_require_forward=True)
    assert ctx.physics_manager.forward_calls == 1
    assert provider.update_calls == [True]


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


def test_vis_marker_registry_dispatch_allows_callback_mutation():
    registry = VisMarkerRegistry()
    calls = []

    def _remove_other_callback(event):
        calls.append(("remove_other", event))
        registry.remove_callback("other")

    def _other_callback(event):
        calls.append(("other", event))

    registry.add_callback("remove_other", _remove_other_callback)
    registry.add_callback("other", _other_callback)

    registry.dispatch_callbacks("tick")

    assert calls == [("remove_other", "tick"), ("other", "tick")]
    assert "other" not in registry._callbacks


class _DummyViserSceneDataProvider:
    def __init__(self):
        self._metadata = {"num_envs": 4}
        self.state_calls: list[list[int] | None] = []

    def get_metadata(self) -> dict:
        return self._metadata

    def get_newton_model(self):
        return "dummy-model"

    def get_newton_state(self):
        self.state_calls.append(None)
        return {"state_call": len(self.state_calls)}

    def get_camera_transforms(self):
        return {}


class _DummyViserViewer:
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


def test_viser_visualizer_initialize_and_step_uses_provider_state(monkeypatch: pytest.MonkeyPatch):
    provider = _DummyViserSceneDataProvider()
    viewer = _DummyViserViewer()

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        assert record_to_viser is None
        assert metadata == provider.get_metadata()
        self._viewer = viewer

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(cast(Any, provider))
    visualizer.step(0.25)

    assert visualizer.is_initialized
    assert provider.state_calls == [None, None]
    assert visualizer._sim_time == pytest.approx(0.25)
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[0][1] == pytest.approx(0.25)
    # log_state passes through get_newton_state() as-is; no env_ids (or other) keys are merged in.
    assert viewer.calls[1] == ("log_state", {"state_call": 2})
    assert viewer.calls[2] == ("end_frame",)


def test_viser_visualizer_marker_render_failure_does_not_interrupt_state_updates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    provider = _DummyViserSceneDataProvider()
    viewer = _DummyViserViewer()
    marker_calls = []

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        self._viewer = viewer

    def _raise_marker_render(*args, **kwargs):
        marker_calls.append((args, kwargs))
        raise RuntimeError("marker overlay failed")

    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)
    monkeypatch.setattr(viser_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(cast(Any, provider))

    with caplog.at_level("WARNING"):
        visualizer.step(0.25)

    assert marker_calls
    assert viewer.calls[0][0] == "begin_frame"
    assert viewer.calls[1] == ("log_state", {"state_call": 2})
    assert viewer.calls[2] == ("end_frame",)
    assert "Marker rendering failed; continuing body updates" in caplog.text


def test_newton_marker_mesh_registration_is_per_viewer(monkeypatch: pytest.MonkeyPatch):
    marker = object.__new__(newton_markers.NewtonVisualizationMarkers)
    marker._registered_meshes = set()

    class _FakeMesh:
        vertices = np.zeros((1, 3), dtype=np.float32)
        indices = np.zeros((3,), dtype=np.int32)
        normals = np.zeros((0, 3), dtype=np.float32)
        uvs = np.zeros((0, 2), dtype=np.float32)

    class _FakeViewer:
        def __init__(self):
            self.meshes = []

        def log_mesh(self, name, vertices, indices, **kwargs):
            self.meshes.append((name, vertices, indices, kwargs))

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None: value)

    spec = newton_markers._NewtonMarkerSpec(renderer="mesh", mesh_type="box", mesh_params={"size": (1.0, 1.0, 1.0)})
    viewer_a = _FakeViewer()
    viewer_b = _FakeViewer()

    marker._ensure_mesh_registered(viewer_a, "/Visuals/marker/meshes/arrow", spec)
    marker._ensure_mesh_registered(viewer_a, "/Visuals/marker/meshes/arrow", spec)
    marker._ensure_mesh_registered(viewer_b, "/Visuals/marker/meshes/arrow", spec)

    assert len(viewer_a.meshes) == 1
    assert len(viewer_b.meshes) == 1


class _FakeNewtonMarkerMesh:
    vertices = np.zeros((1, 3), dtype=np.float32)
    indices = np.zeros((3,), dtype=np.int32)
    normals = np.zeros((0, 3), dtype=np.float32)
    uvs = np.zeros((0, 2), dtype=np.float32)


class _FakeNewtonMarkerViewer:
    def __init__(self):
        self.meshes = []
        self.instances = []
        self.lines = []

    def log_mesh(self, name, vertices, indices, **kwargs):
        self.meshes.append((name, vertices, indices, kwargs))

    def log_instances(self, batch_name, mesh_name, xforms, scales, colors, materials, hidden=False):
        self.instances.append(
            {
                "batch_name": batch_name,
                "mesh_name": mesh_name,
                "xforms": xforms,
                "scales": scales,
                "colors": colors,
                "materials": materials,
                "hidden": hidden,
            }
        )

    def log_lines(self, batch_name, starts, ends, colors, width=None, hidden=False):
        self.lines.append(
            {
                "batch_name": batch_name,
                "starts": starts,
                "ends": ends,
                "colors": colors,
                "width": width,
                "hidden": hidden,
            }
        )


def _make_newton_marker_for_render(
    *,
    marker_names: list[str],
    translations: torch.Tensor,
    marker_indices: torch.Tensor | None = None,
    visible: bool = True,
):
    marker = object.__new__(newton_markers.NewtonVisualizationMarkers)
    marker_cfg_type = type("MarkerCfg", (), {"visual_material": None})
    marker.cfg = type("Cfg", (), {"markers": {name: marker_cfg_type() for name in marker_names}})()
    marker.group_id = "/Visuals/marker::test"
    marker.visible = visible
    marker.translations = translations
    marker.orientations = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32).repeat(translations.shape[0], 1)
    marker.scales = torch.ones((translations.shape[0], 3), dtype=torch.float32)
    marker.marker_indices = marker_indices
    marker.count = translations.shape[0]
    marker._registered_meshes = set()
    marker._warned_unsupported = set()
    return marker


def _patch_newton_marker_render_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    specs = {
        "arrow": newton_markers._NewtonMarkerSpec(
            renderer="mesh",
            mesh_type="box",
            mesh_params={"size": (1.0, 1.0, 1.0)},
            color=(1.0, 1.0, 1.0),
            texture=np.zeros((2, 2, 3), dtype=np.uint8),
        ),
        "sphere": newton_markers._NewtonMarkerSpec(renderer="mesh", mesh_type="sphere", mesh_params={"radius": 1.0}),
        "frame": newton_markers._NewtonMarkerSpec(renderer="frame"),
    }

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeNewtonMarkerMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None: value)
    monkeypatch.setattr(newton_markers, "_resolve_newton_marker_cfg", lambda name, marker_cfg, cfg: specs[name])


def test_newton_marker_render_filters_visible_envs(monkeypatch: pytest.MonkeyPatch):
    _patch_newton_marker_render_deps(monkeypatch)
    translations = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer()

    marker.render(viewer, visible_env_ids=[1, 3], num_envs=4)

    assert len(viewer.instances) == 1
    assert viewer.instances[0]["hidden"] is False
    assert viewer.instances[0]["xforms"][:, 0].tolist() == [1.0, 3.0, 5.0, 7.0]


def test_newton_marker_render_routes_instances_by_prototype(monkeypatch: pytest.MonkeyPatch):
    _patch_newton_marker_render_deps(monkeypatch)
    translations = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=translations,
        marker_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer()

    marker.render(viewer, visible_env_ids=None, num_envs=4)

    visible_instances = [call for call in viewer.instances if not call["hidden"]]
    assert [call["batch_name"] for call in visible_instances] == [
        "/Visuals/marker::test/arrow",
        "/Visuals/marker::test/sphere",
    ]
    assert [call["xforms"].shape[0] for call in visible_instances] == [2, 2]
    assert visible_instances[0]["materials"][:, 3].tolist() == [1.0, 1.0]
    assert visible_instances[1]["materials"][:, 3].tolist() == [0.0, 0.0]


def test_newton_marker_render_hides_unselected_prototypes(monkeypatch: pytest.MonkeyPatch):
    _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere", "frame"],
        translations=torch.zeros((3, 3), dtype=torch.float32),
        marker_indices=torch.zeros(3, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer()

    marker.render(viewer, visible_env_ids=None, num_envs=3)

    hidden_instances = [call for call in viewer.instances if call["hidden"]]
    assert [call["batch_name"] for call in hidden_instances] == ["/Visuals/marker::test/sphere"]
    assert viewer.lines == [
        {
            "batch_name": "/Visuals/marker::test/frame",
            "starts": None,
            "ends": None,
            "colors": None,
            "width": None,
            "hidden": True,
        }
    ]


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
            label: str | None,
            verbose: bool,
            share: bool,
            record_to_viser: str | None,
            metadata: dict | None = None,
        ):
            captured["init"] = {
                "port": port,
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
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


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
    cfg_max_visible_envs: int | None,
    expected_visible: list[int] | None,
):
    captured = {}

    class _FakeNewtonViewerRerun:
        def __init__(
            self,
            *,
            app_id: str,
            address: str | None,
            serve_web_viewer: bool,
            web_port: int,
            grpc_port: int,
            keep_historical_data: bool,
            keep_scalar_history: bool,
            record_to_rrd: str | None,
        ):
            captured["init"] = {
                "app_id": app_id,
                "address": address,
                "serve_web_viewer": serve_web_viewer,
                "web_port": web_port,
                "grpc_port": grpc_port,
                "keep_historical_data": keep_historical_data,
                "keep_scalar_history": keep_scalar_history,
                "record_to_rrd": record_to_rrd,
            }

        def set_model(self, model: Any) -> None:
            captured["set_model"] = model

        def set_visible_worlds(self, worlds) -> None:
            captured["visible_worlds"] = worlds

        def set_world_offsets(self, spacing) -> None:
            captured["set_world_offsets"] = tuple(spacing)

        def close(self) -> None:
            captured["closed"] = True

    class _DummyRerunSceneDataProvider:
        def get_metadata(self) -> dict:
            return {"num_envs": 4}

        def get_newton_model(self):
            return "dummy-model"

        def get_newton_state(self):
            return {"ok": True}

        def get_camera_transforms(self):
            return {}

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
    visualizer.initialize(cast(Any, _DummyRerunSceneDataProvider()))

    assert captured["set_model"] == "dummy-model"
    assert captured["visible_worlds"] == expected_visible
    assert captured["set_world_offsets"] == (0.0, 0.0, 0.0)


def test_rerun_visualizer_marker_failure_still_ends_frame(monkeypatch: pytest.MonkeyPatch):
    class _FakeRerunViewer:
        def __init__(self):
            self.calls = []

        def is_paused(self):
            return False

        def begin_frame(self, sim_time):
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state):
            self.calls.append(("log_state", state))

        def end_frame(self):
            self.calls.append(("end_frame",))

    class _DummyRerunSceneDataProvider:
        def get_metadata(self) -> dict:
            return {"num_envs": 4}

        def get_newton_state(self):
            return {"ok": True}

        def get_camera_transforms(self):
            return {}

    def _raise_marker_render(*args, **kwargs):
        raise RuntimeError("marker render failed")

    monkeypatch.setattr(rerun_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = rerun_visualizer.RerunVisualizer(RerunVisualizerCfg())
    viewer = _FakeRerunViewer()
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._viewer = viewer
    visualizer._scene_data_provider = _DummyRerunSceneDataProvider()
    visualizer._resolved_visible_env_ids = None

    with pytest.raises(RuntimeError, match="marker render failed"):
        visualizer.step(0.25)

    assert [call[0] for call in viewer.calls] == ["begin_frame", "log_state", "end_frame"]


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

    class _FakeProvider:
        def get_usd_stage(self):
            return _FakeStage()

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

    assert cfg.cam_source == "cfg"
    assert applied_camera_poses == [(cfg.eye, cfg.lookat)]
    assert viewport_window.viewport_api.set_active_camera_calls == []
    assert visualizer._controlled_camera_path == "/OmniverseKit_Persp"


def test_get_cli_visualizer_types_handles_non_string_setting_without_crashing():
    ctx = object.__new__(SimulationContext)
    ctx.get_setting = lambda name: {"types": "newton,kit"} if name == "/isaaclab/visualizer/types" else None

    assert ctx._get_cli_visualizer_types() == []


# ---------------------------------------------------------------------------
# Shared helpers for config-resolution and initialize_visualizers tests
# ---------------------------------------------------------------------------


class _FakeVisualizerCfg:
    """Minimal visualizer config for testing initialize_visualizers."""

    def __init__(self, visualizer_type: str, *, fail_create: bool = False, fail_init: bool = False):
        self.visualizer_type = visualizer_type
        self._fail_create = fail_create
        self._fail_init = fail_init

    def create_visualizer(self):
        if self._fail_create:
            raise RuntimeError("create failed")
        return _FakeVisualizer() if not self._fail_init else _FailingInitVisualizer()


class _FailingInitVisualizer(_FakeVisualizer):
    def initialize(self, provider):
        raise RuntimeError("init failed")


def _make_context_with_settings(
    settings: dict,
    visualizer_cfgs=None,
    *,
    has_gui: bool = False,
    has_offscreen_render: bool = False,
):
    """Build a minimal SimulationContext suitable for testing is_rendering, _resolve_visualizer_cfgs,
    and initialize_visualizers.

    Centralises the ``object.__new__`` construction so new internal attributes only need to be added
    in one place when the production code changes.
    """
    cfg = type(
        "Cfg",
        (),
        {
            "visualizer_cfgs": visualizer_cfgs,
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
    ctx._scene_data_provider = _FakeProvider()
    ctx._scene_data_requirements = None
    ctx._clone_plans = {}
    ctx._visualizer_step_counter = 0
    ctx._viz_dt = 0.01
    ctx.get_setting = lambda name: settings.get(name)
    return ctx


def test_is_rendering_true_when_only_cfg_visualizer_is_set():
    cfg_visualizer = type("CfgVisualizer", (), {"visualizer_type": "newton"})()
    settings = {
        "/isaaclab/render/rtx_sensors": False,
        "/isaaclab/visualizer/types": "",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[cfg_visualizer])
    assert ctx.is_rendering is True


def test_is_rendering_false_when_cli_disable_all_even_with_cfg_visualizer():
    cfg_visualizer = type("CfgVisualizer", (), {"visualizer_type": "newton"})()
    settings = {
        "/isaaclab/render/rtx_sensors": False,
        "/isaaclab/visualizer/types": "",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": True,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[cfg_visualizer])
    assert ctx.is_rendering is False


def test_explicit_unknown_visualizer_type_raises():
    """Requesting an unknown visualizer type via CLI raises RuntimeError."""
    settings = {
        "/isaaclab/visualizer/types": "bogus_viz",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings)

    with pytest.raises(RuntimeError, match="bogus_viz"):
        ctx.initialize_visualizers()


def test_explicit_missing_package_raises(monkeypatch: pytest.MonkeyPatch):
    """Requesting a valid type whose package is not installed raises RuntimeError."""
    settings = {
        "/isaaclab/visualizer/types": "rerun",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings)

    # Force import to fail for the rerun visualizer module
    import builtins

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if "isaaclab_visualizers.rerun" in name:
            raise ImportError("No module named 'isaaclab_visualizers.rerun'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)

    with pytest.raises(RuntimeError, match="rerun"):
        ctx.initialize_visualizers()


def test_explicit_visualizer_create_failure_raises(monkeypatch: pytest.MonkeyPatch):
    """When cli_explicit, a failure in create_visualizer raises RuntimeError."""
    failing_cfg = _FakeVisualizerCfg("newton", fail_create=True)
    settings = {
        "/isaaclab/visualizer/types": "newton",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[failing_cfg])

    import isaaclab.sim.simulation_context as sc_mod

    monkeypatch.setattr(sc_mod, "resolve_scene_data_requirements", lambda **kwargs: type("R", (), {})())

    with pytest.raises(RuntimeError, match="failed to create or initialize"):
        ctx.initialize_visualizers()


def test_explicit_visualizer_init_failure_raises(monkeypatch: pytest.MonkeyPatch):
    """When cli_explicit, a failure in visualizer.initialize raises RuntimeError."""
    failing_cfg = _FakeVisualizerCfg("newton", fail_init=True)
    settings = {
        "/isaaclab/visualizer/types": "newton",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[failing_cfg])

    import isaaclab.sim.simulation_context as sc_mod

    monkeypatch.setattr(sc_mod, "resolve_scene_data_requirements", lambda **kwargs: type("R", (), {})())

    with pytest.raises(RuntimeError, match="failed to create or initialize"):
        ctx.initialize_visualizers()


def test_explicit_partial_valid_types_raises_for_invalid():
    """Requesting 'newton,bogus_viz' via CLI raises for the unknown type even though newton is valid."""
    settings = {
        "/isaaclab/visualizer/types": "newton,bogus_viz",
        "/isaaclab/visualizer/explicit": True,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings)

    with pytest.raises(RuntimeError, match="bogus_viz"):
        ctx.initialize_visualizers()


def test_non_explicit_unknown_type_silently_skipped(caplog):
    """Without --visualizer flag, unknown types are silently skipped (no error)."""
    settings = {
        "/isaaclab/visualizer/types": "bogus_viz",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings)

    # Non-explicit: should not raise
    ctx.initialize_visualizers()
    assert ctx._visualizers == []


def test_non_explicit_create_failure_silently_logged(monkeypatch: pytest.MonkeyPatch, caplog):
    """Without --visualizer flag, create_visualizer failures are logged, not raised."""
    failing_cfg = _FakeVisualizerCfg("newton", fail_create=True)
    settings = {
        "/isaaclab/visualizer/types": "",
        "/isaaclab/visualizer/explicit": False,
        "/isaaclab/visualizer/disable_all": False,
        "/isaaclab/visualizer/max_visible_envs": None,
    }
    ctx = _make_context_with_settings(settings, visualizer_cfgs=[failing_cfg])

    import isaaclab.sim.simulation_context as sc_mod

    monkeypatch.setattr(sc_mod, "resolve_scene_data_requirements", lambda **kwargs: type("R", (), {})())

    with caplog.at_level("ERROR"):
        ctx.initialize_visualizers()
    assert ctx._visualizers == []
    assert any("Failed to initialize visualizer" in r.message for r in caplog.records)
