# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaaclab_visualizers.newton.newton_visualization_markers as newton_markers
import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import numpy as np
import pytest
import torch
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonVisualizerCfg
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import random_orientation


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""
    # Simulation time-step
    dt = 0.01
    # Open a new stage
    sim_utils.create_new_stage()
    # Load kit helper
    sim_context = SimulationContext(SimulationCfg(dt=dt))
    yield sim_context
    # Cleanup
    sim_context._disable_app_control_on_stop_handle = True  # prevent timeout
    sim_context.stop()
    sim_context.clear_instance()
    sim_utils.close_stage()


class _FakeMarkerVisualizer:
    def __init__(self, *, enable_markers: bool = True, pumps_app_update: bool = False):
        self.cfg = type("Cfg", (), {"enable_markers": enable_markers})()
        self._pumps_app_update = pumps_app_update

    def supports_markers(self):
        return True

    def pumps_app_update(self):
        return self._pumps_app_update

    def stop(self):
        pass

    def close(self):
        pass


def test_instantiation(sim):
    """Test that the class can be initialized properly."""
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/test",
        markers={
            "test": sim_utils.SphereCfg(radius=1.0),
        },
    )
    test_marker = VisualizationMarkers(config)
    print(test_marker)
    # check number of markers
    assert test_marker.num_prototypes == 1


@pytest.mark.parametrize(
    ("is_rendering", "visualizers", "expected_backends"),
    [
        (True, [], ["kit"]),
        (False, [], []),
        (False, [KitVisualizer(KitVisualizerCfg())], ["kit"]),
        (False, [newton_visualizer.NewtonVisualizer(NewtonVisualizerCfg())], ["newton"]),
        (False, [rerun_visualizer.RerunVisualizer(RerunVisualizerCfg())], ["newton"]),
        (False, [viser_visualizer.ViserVisualizer(ViserVisualizerCfg())], ["newton"]),
    ],
)
def test_marker_backend_selection(monkeypatch, is_rendering: bool, visualizers: list, expected_backends: list[str]):
    """Marker backend selection follows rendering state and active visualizer type."""
    marker = object.__new__(VisualizationMarkers)
    marker._backends = []
    fake_sim = type("FakeSim", (), {"is_rendering": is_rendering, "visualizers": visualizers})()

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", staticmethod(lambda: fake_sim))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_kit_backend", lambda self: self._backends.append("kit"))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_newton_backend", lambda self: self._backends.append("newton"))

    marker._ensure_backends_initialized()

    assert marker._backends == expected_backends


def test_rendering_context_authors_visible_usd_point_instancer(sim):
    """Rendering-active contexts should create visible USD marker prims."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/rendered_marker",
        markers={
            "failure": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15)),
                visible=True,
            ),
            "success": sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15)),
                visible=True,
            ),
        },
    )
    test_marker = VisualizationMarkers(config)
    test_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device),
        marker_indices=torch.tensor([0, 1], device=sim.device),
    )

    stage = sim_utils.get_current_stage()
    instancer_prim = stage.GetPrimAtPath(test_marker.prim_path)
    instancer = UsdGeom.PointInstancer(instancer_prim)

    assert instancer_prim.IsValid()
    assert instancer
    assert UsdGeom.Imageable(instancer_prim).GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible
    assert len(instancer.GetPositionsAttr().Get()) == 2
    assert list(instancer.GetProtoIndicesAttr().Get()) == [0, 1]


def test_usd_marker(sim):
    """Test with marker from a USD."""
    # create a marker
    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
    # create a buffer
    num_frames = 0
    # run with randomization of poses
    for count in range(1000):
        # sample random poses
        if count % 50 == 0:
            num_frames = torch.randint(10, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            frame_rotations = random_orientation(num_frames, device=sim.device)
            # set the marker
            test_marker.visualize(translations=frame_translations, orientations=frame_rotations)
        # update the kit
        sim.step()
        # asset that count is correct
        assert test_marker.count == num_frames


def test_multiple_prototypes_marker(sim):
    """Test with multiple prototypes of spheres."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
    # run with randomization of poses
    for count in range(1000):
        # sample random poses
        if count % 50 == 0:
            num_frames = torch.randint(100, 1000, (1,)).item()
            frame_translations = torch.randn(num_frames, 3, device=sim.device)
            # randomly choose a prototype
            marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)
            # set the marker
            test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
        # update the kit
        sim.step()


def test_visualization_skips_updates_when_invisible(sim):
    """When invisible, visualize should not update marker state."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()

    # check that visibility is true
    assert test_marker.is_visible()
    frame_translations = torch.randn(4, 3, device=sim.device)
    marker_indices = torch.zeros(4, dtype=torch.int32, device=sim.device)
    test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
    assert test_marker.count == 4

    # update the kit
    sim.step()
    # make invisible
    test_marker.set_visibility(False)

    # check that visibility is false
    assert not test_marker.is_visible()
    test_marker.visualize(
        translations=torch.randn(8, 3, device=sim.device),
        marker_indices=torch.zeros(8, dtype=torch.int32, device=sim.device),
    )

    assert test_marker.count == 4


def test_newton_marker_backend_registers_and_updates_state_without_frame_capture(sim):
    """Newton marker backend state should be registered and ready for Newton-family viewers."""
    sim._visualizers.append(_FakeMarkerVisualizer(pumps_app_update=False))
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/newton_marker_state"
    test_marker = VisualizationMarkers(config)
    translations = torch.arange(6, dtype=torch.float32, device=sim.device).reshape(2, 3)
    marker_indices = torch.tensor([0, 0], device=sim.device)

    test_marker.visualize(translations=translations, marker_indices=marker_indices)

    newton_backend = test_marker._backends[0]
    assert isinstance(newton_backend, newton_markers.NewtonVisualizationMarkers)
    assert sim.vis_marker_registry.get_groups()[newton_backend.group_id] is newton_backend
    assert torch.equal(newton_backend.translations, translations)
    assert torch.equal(newton_backend.marker_indices, marker_indices.to(dtype=torch.int32))
    assert newton_backend.count == 2


def test_newton_visualizer_step_renders_markers(monkeypatch: pytest.MonkeyPatch):
    """NewtonVisualizer.step should ask active Newton marker groups to render."""
    marker_calls = []

    class _FakeViewer:
        _update_frequency = 1

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

    class _FakeNewtonManager:
        @staticmethod
        def get_state(scene_data_provider=None):
            assert scene_data_provider == "provider"
            return {"state": "ok"}

        @staticmethod
        def get_num_envs() -> int:
            return 4

    def _fake_render_markers(viewer, visible_env_ids, num_envs):
        marker_calls.append((viewer, visible_env_ids, num_envs))

    import isaaclab_newton.physics as newton_physics

    monkeypatch.setattr(newton_physics, "NewtonManager", _FakeNewtonManager)
    monkeypatch.setattr(newton_visualizer, "render_newton_visualization_markers", _fake_render_markers)

    viewer = _FakeViewer()
    visualizer = newton_visualizer.NewtonVisualizer(NewtonVisualizerCfg(enable_markers=True))
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._viewer = viewer
    visualizer._scene_data_provider = "provider"
    visualizer._resolved_visible_env_ids = [1, 3]

    visualizer.step(0.25)

    assert viewer.calls == [("begin_frame", pytest.approx(0.25)), ("log_state", {"state": "ok"}), ("end_frame",)]
    assert marker_calls == [(viewer, [1, 3], 4)]


def test_viser_visualizer_marker_render_failure_does_not_interrupt_state_updates(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Viser marker failures should be logged without dropping body state frames."""
    marker_calls = []

    class _FakeViewer:
        def __init__(self):
            self.calls = []

        def begin_frame(self, sim_time: float) -> None:
            self.calls.append(("begin_frame", sim_time))

        def log_state(self, state) -> None:
            self.calls.append(("log_state", state))

        def end_frame(self) -> None:
            self.calls.append(("end_frame",))

    class _FakeProvider:
        num_envs = 4
        usd_stage = None

        def get_camera_transforms(self):
            return {}

    class _FakeNewtonManager:
        @staticmethod
        def get_model():
            return "dummy-model"

        @staticmethod
        def get_state(scene_data_provider=None):
            assert scene_data_provider is provider
            return {"state": "ok"}

        @staticmethod
        def get_num_envs() -> int:
            return provider.num_envs

    def _fake_create_viewer(self, record_to_viser: str | None, metadata: dict | None = None):
        self._viewer = viewer

    def _raise_marker_render(*args, **kwargs):
        marker_calls.append((args, kwargs))
        raise RuntimeError("marker overlay failed")

    import isaaclab_newton.physics as newton_physics

    provider = _FakeProvider()
    viewer = _FakeViewer()
    monkeypatch.setattr(newton_physics, "NewtonManager", _FakeNewtonManager)
    monkeypatch.setattr(viser_visualizer.ViserVisualizer, "_create_viewer", _fake_create_viewer)
    monkeypatch.setattr(viser_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = viser_visualizer.ViserVisualizer(ViserVisualizerCfg())
    visualizer.initialize(provider)

    with caplog.at_level("WARNING"):
        visualizer.step(0.25)

    assert marker_calls
    assert viewer.calls == [("begin_frame", pytest.approx(0.25)), ("log_state", {"state": "ok"}), ("end_frame",)]
    assert "Marker rendering failed; continuing body updates" in caplog.text


def test_rerun_visualizer_marker_failure_still_ends_frame(monkeypatch: pytest.MonkeyPatch):
    """Rerun should close the frame even if marker rendering raises."""
    captured = {}

    class _FakeViewer:
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

    class _FakeProvider:
        def get_metadata(self) -> dict:
            return {"num_envs": 4}

        def get_newton_state(self):
            return {"ok": True}

        def get_camera_transforms(self):
            return {}

    class _FakeNewtonManager:
        @staticmethod
        def get_model():
            return "dummy-model"

        @staticmethod
        def get_state(scene_data_provider=None):
            captured["state_provider"] = scene_data_provider
            return {"ok": True}

        @staticmethod
        def get_num_envs() -> int:
            return 4

    def _raise_marker_render(*args, **kwargs):
        raise RuntimeError("marker render failed")

    import isaaclab_newton.physics as newton_physics

    monkeypatch.setattr(newton_physics, "NewtonManager", _FakeNewtonManager)
    monkeypatch.setattr(rerun_visualizer, "render_newton_visualization_markers", _raise_marker_render)

    visualizer = rerun_visualizer.RerunVisualizer(RerunVisualizerCfg())
    viewer = _FakeViewer()
    visualizer._is_initialized = True
    visualizer._is_closed = False
    visualizer._viewer = viewer
    visualizer._scene_data_provider = _FakeProvider()
    visualizer._resolved_visible_env_ids = None

    with pytest.raises(RuntimeError, match="marker render failed"):
        visualizer.step(0.25)

    assert captured["state_provider"] is visualizer._scene_data_provider
    assert [call[0] for call in viewer.calls] == ["begin_frame", "log_state", "end_frame"]


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
