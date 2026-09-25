# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from types import SimpleNamespace

import isaaclab_visualizers.newton.newton_visualization_markers as newton_markers
import isaaclab_visualizers.newton.newton_visualizer as newton_visualizer
import isaaclab_visualizers.rerun.rerun_visualizer as rerun_visualizer
import isaaclab_visualizers.viser.viser_visualizer as viser_visualizer
import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_visualizers.kit.kit_visualizer import KitVisualizer
from isaaclab_visualizers.kit.kit_visualizer_cfg import KitVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer_cfg import NewtonGLVisualizerCfg
from isaaclab_visualizers.rerun.rerun_visualizer_cfg import RerunVisualizerCfg
from isaaclab_visualizers.viser.viser_visualizer_cfg import ViserVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import random_orientation

pytestmark = pytest.mark.integration


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


@pytest.mark.parametrize(
    ("has_gui", "rtx_sensors", "xr_enabled", "has_offscreen_render", "visualizers", "expected_backends"),
    [
        (True, False, False, False, [], ["kit"]),
        (False, True, False, False, [], ["kit"]),
        (False, False, True, False, [], ["kit"]),
        (False, False, False, True, [], ["kit"]),
        (False, False, False, False, [], []),
        (False, False, False, False, [KitVisualizer(KitVisualizerCfg())], ["kit"]),
        (False, False, False, False, [newton_visualizer.NewtonVisualizer(NewtonGLVisualizerCfg())], ["newton"]),
        (False, False, False, False, [rerun_visualizer.RerunVisualizer(RerunVisualizerCfg())], ["newton"]),
        (False, False, False, False, [viser_visualizer.ViserVisualizer(ViserVisualizerCfg())], ["newton"]),
    ],
)
def test_marker_backend_selection(
    monkeypatch,
    has_gui: bool,
    rtx_sensors: bool,
    xr_enabled: bool,
    has_offscreen_render: bool,
    visualizers: list,
    expected_backends: list[str],
):
    """Marker backend selection follows rendering state and active visualizer type.

    Regression coverage for a bug where a non-Kit-pumping visualizer (e.g. ``newton_gl``) alone
    would still spin up the Kit/USD marker backend (because it also makes ``sim.is_rendering``
    true), leaving raw USD marker writes undigested by Fabric. That desynced the point-instancer
    prototype table and crashed the next PhysX GPU step. The ``newton_gl``-only case below
    (``rtx_sensors``/``xr_enabled``/``has_gui``/``has_offscreen_render`` all False) must select
    only the ``newton`` backend, never ``kit``.
    """
    marker = object.__new__(VisualizationMarkers)
    marker._backends = []
    settings = {"/isaaclab/render/rtx_sensors": rtx_sensors, "/isaaclab/xr/enabled": xr_enabled}
    fake_sim = type(
        "FakeSim",
        (),
        {
            "has_gui": has_gui,
            "has_offscreen_render": has_offscreen_render,
            "visualizers": visualizers,
            "get_setting": lambda self, key: settings.get(key, False),
        },
    )()

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
    assert test_marker.num_prototypes == 2
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


def test_environment_ids_author_scene_partitions_and_rebuild_only_on_change(sim, monkeypatch):
    """Per-instance environment IDs author vertex-interpolated scene-partition tokens, rebuilt only on change.

    Marker ownership is static in most tasks, but ``visualize`` runs every frame. Rebuilding the
    token array anyway costs a device synchronization and one string per marker per frame.
    """
    from pxr import Sdf, UsdGeom, Vt

    sim._has_offscreen_render = True
    stage = sim_utils.get_current_stage()
    for env_id in range(2):
        env_prim = stage.DefinePrim(f"/World/envs/env_{env_id}", "Xform")
        env_prim.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set(f"env_{env_id}")

    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/cached_partition_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = VisualizationMarkers(config)
    translations = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device)
    environment_ids = torch.tensor([1, 0], device=sim.device)
    test_marker.visualize(translations=translations, environment_ids=environment_ids)

    primvar = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(test_marker.prim_path)).GetPrimvar("omni:scenePartition")
    assert primvar
    assert primvar.GetTypeName() == Sdf.ValueTypeNames.TokenArray
    assert primvar.GetInterpolation() == UsdGeom.Tokens.vertex
    assert list(primvar.Get()) == ["env_1", "env_0"]

    rebuilt_token_arrays = []
    original_token_array = Vt.TokenArray

    def _counting_token_array(*args, **kwargs):
        rebuilt_token_arrays.append(args)
        return original_token_array(*args, **kwargs)

    monkeypatch.setattr(Vt, "TokenArray", _counting_token_array)
    # Markers move every frame while their environment ownership stays fixed.
    test_marker.visualize(translations=translations + 0.1, environment_ids=environment_ids)

    assert rebuilt_token_arrays == []
    assert list(primvar.Get()) == ["env_1", "env_0"]

    # New environment IDs still re-author the partition tokens.
    test_marker.visualize(translations=translations, environment_ids=torch.tensor([0, 1], device=sim.device))
    assert list(primvar.Get()) == ["env_0", "env_1"]


def test_environment_ids_require_active_scene_partitions(sim):
    """Environment IDs should not partition markers when renderer stage preparation is inactive."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/unpartitioned_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = VisualizationMarkers(config)
    test_marker.visualize(
        translations=torch.tensor([[0.0, 0.0, 0.0]], device=sim.device),
        environment_ids=torch.tensor([0], device=sim.device),
    )

    instancer_prim = sim_utils.get_current_stage().GetPrimAtPath(test_marker.prim_path)
    primvar = UsdGeom.PrimvarsAPI(instancer_prim).GetPrimvar("omni:scenePartition")
    assert not primvar or not primvar.GetAttr().HasAuthoredValueOpinion()


def test_environment_ids_must_match_marker_count(sim):
    """Each marker instance should require one environment ID."""
    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/mismatched_partition_marker",
        markers={"test": sim_utils.SphereCfg(radius=0.1)},
    )
    test_marker = VisualizationMarkers(config)

    with pytest.raises(ValueError, match="one index per marker"):
        test_marker.visualize(
            translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device),
            environment_ids=torch.tensor([0], device=sim.device),
        )


def test_first_visualize_defaults_to_first_prototype_when_count_matches_prototypes(sim):
    """Omitted marker indices should not preserve initialization prototype placeholders."""
    from pxr import UsdGeom

    sim._has_offscreen_render = True
    config = VisualizationMarkersCfg(
        prim_path="/World/Visuals/default_marker_indices",
        markers={
            "frame": sim_utils.SphereCfg(radius=0.1),
            "line": sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1)),
        },
    )
    test_marker = VisualizationMarkers(config)

    test_marker.visualize(translations=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], device=sim.device))

    instancer = UsdGeom.PointInstancer(sim_utils.get_current_stage().GetPrimAtPath(test_marker.prim_path))
    assert list(instancer.GetProtoIndicesAttr().Get()) == [0, 0]


def test_usd_marker(sim):
    """Test with marker from a USD."""
    # create a marker
    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
    # grow and then shrink the number of frames
    for num_frames in (500, 20):
        frame_translations = torch.randn(num_frames, 3, device=sim.device)
        frame_rotations = random_orientation(num_frames, device=sim.device)
        test_marker.visualize(translations=frame_translations, orientations=frame_rotations)
        assert test_marker.count == num_frames
    # update the kit
    sim.step()
    assert test_marker.count == num_frames


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


@pytest.mark.parametrize(
    "module, cfg_type, marker_error",
    [
        (newton_visualizer, NewtonGLVisualizerCfg, False),
        (viser_visualizer, ViserVisualizerCfg, True),
        (rerun_visualizer, RerunVisualizerCfg, True),
    ],
    ids=["newton", "viser", "rerun"],
)
def test_visualizer_step_renders_markers_and_closes_frame(monkeypatch, caplog, module, cfg_type, marker_error):
    """Markers use the current native state; overlay failures still close the frame."""
    calls, marker_calls = [], []
    state = SimpleNamespace(body_q=None)
    backend = SimpleNamespace(model=SimpleNamespace(num_envs=4, body_count=0), state_0=state, geometry_offsets={})

    class Viewer:
        _update_frequency = 1
        show_contacts = False

        def is_paused(self):
            return False

        def is_running(self):
            return True

        def begin_frame(self, sim_time):
            calls.append(("begin_frame", sim_time))

        def log_state(self, value):
            calls.append(("log_state", value))

        def log_arrows(self, name, starts, ends, colors):
            pass

        def end_frame(self):
            calls.append(("end_frame",))

    def render_markers(viewer, visible_env_ids, num_envs):
        marker_calls.append((viewer, visible_env_ids, num_envs))
        if marker_error:
            raise RuntimeError("marker overlay failed")

    provider = SimpleNamespace(
        get_transforms=lambda output, **kwargs: False,
        get_camera_transforms=lambda: {},
        get_contact_sensors=lambda: {},
    )
    monkeypatch.setattr(module, "render_newton_visualization_markers", render_markers)
    monkeypatch.setattr(newton_visualizer.NewtonManager, "get_contacts", lambda: None)
    cfg = cfg_type()
    visualizer = cfg.class_type(cfg)
    visualizer.backend = backend
    visualizer._is_initialized = True
    visualizer._viewer = viewer = Viewer()
    visualizer._scene_data_provider = provider
    visualizer._transform_mapping = None
    visualizer._resolved_visible_env_ids = [1, 3]

    with caplog.at_level("WARNING"):
        if module is rerun_visualizer:
            with pytest.raises(RuntimeError, match="marker overlay failed"):
                visualizer.step(0.25)
        else:
            visualizer.step(0.25)

    assert calls == [("begin_frame", pytest.approx(0.25)), ("log_state", state), ("end_frame",)]
    assert marker_calls == [(viewer, [1, 3], 4)]
    if module is viser_visualizer:
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
        device = "cpu"

        def __init__(self):
            self.meshes = []

        def log_mesh(self, name, vertices, indices, **kwargs):
            self.meshes.append((name, vertices, indices, kwargs))

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None, device=None: value)

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


_NEWTON_MARKER_SPECS = {
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


class _FakeNewtonMarkerViewer:
    def __init__(self, world_offsets):
        self.world_offsets = world_offsets
        self.device = world_offsets.device
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
    marker._marker_specs = {name: _NEWTON_MARKER_SPECS[name] for name in marker_names}
    return marker


def _patch_newton_marker_render_deps(
    monkeypatch: pytest.MonkeyPatch, world_offsets: np.ndarray | None = None, world_offsets_device: str = "cpu"
):
    if world_offsets is None:
        world_offsets = np.zeros((4, 3), dtype=np.float32)
    warp_world_offsets = wp.array(world_offsets, dtype=wp.vec3, device=world_offsets_device)

    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeNewtonMarkerMesh())
    monkeypatch.setattr(newton_markers.wp, "array", lambda value, dtype=None, device=None: value)
    return warp_world_offsets


def test_newton_marker_partial_update_preserves_prototype_indices():
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=torch.zeros((4, 3), dtype=torch.float32),
        marker_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    expected_indices = marker.marker_indices

    marker.visualize(
        translations=torch.ones((4, 3), dtype=torch.float32),
        orientations=None,
        scales=None,
        marker_indices=None,
    )

    assert marker.marker_indices is expected_indices


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA marker state")
def test_newton_marker_render_uses_viewer_device(monkeypatch: pytest.MonkeyPatch):
    world_offsets = wp.zeros(4, dtype=wp.vec3, device="cpu")
    monkeypatch.setattr(newton_markers, "_create_mesh", lambda cfg: _FakeNewtonMarkerMesh())
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=torch.zeros((4, 3), device="cuda:0"),
        marker_indices=torch.zeros(4, dtype=torch.int32, device="cuda:0"),
    )
    marker.orientations = marker.orientations.to("cuda:0")
    marker.scales = marker.scales.to("cuda:0")
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=None, num_envs=4)

    call = viewer.instances[0]
    assert all(call[key].device == viewer.device for key in ("xforms", "scales", "colors", "materials"))
    assert all(viewer.meshes[0][index].device == viewer.device for index in (1, 2))


@pytest.mark.parametrize(
    ("visible_env_ids", "expected"),
    [
        ([1, 3], [12.0, 13.0, 36.0, 37.0]),
        (None, [0.0, 1.0, 12.0, 13.0, 24.0, 25.0, 36.0, 37.0]),
    ],
)
def test_newton_marker_render_applies_world_offsets(
    monkeypatch: pytest.MonkeyPatch, visible_env_ids: list[int] | None, expected: list[float]
):
    world_offsets_device = "cuda:0" if wp.is_cuda_available() else "cpu"
    world_offsets = _patch_newton_marker_render_deps(
        monkeypatch,
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        world_offsets_device=world_offsets_device,
    )
    translations = torch.arange(8, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=visible_env_ids, num_envs=4)

    assert len(viewer.instances) == 1
    assert viewer.instances[0]["hidden"] is False
    assert viewer.instances[0]["xforms"][:, 0].tolist() == expected


def test_newton_marker_render_preserves_reordered_env_state(monkeypatch: pytest.MonkeyPatch):
    offsets = np.array(
        [
            [0.0, 0.0, 0.0],
            [100.0, 200.0, 300.0],
            [200.0, 400.0, 600.0],
            [300.0, 600.0, 900.0],
        ],
        dtype=np.float32,
    )
    world_offsets = _patch_newton_marker_render_deps(monkeypatch, offsets)
    translations = torch.arange(24, dtype=torch.float32).reshape(3, 8).T
    orientations = torch.arange(32, dtype=torch.float32).reshape(4, 8).T
    scales = torch.arange(1, 25, dtype=torch.float32).reshape(3, 8).T
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(8, dtype=torch.int32),
    )
    marker.orientations = orientations
    marker.scales = scales
    source_state = (
        marker.translations,
        marker.orientations,
        marker.scales,
        marker.marker_indices,
    )
    source_values = tuple(value.clone() for value in source_state)
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    selections = (
        ([3, 1], torch.tensor([6, 7, 2, 3])),
        ([1, 3], torch.tensor([2, 3, 6, 7])),
    )

    monkeypatch.setattr(
        newton_markers.torch,
        "arange",
        lambda *args, **kwargs: pytest.fail("render should not construct environment index tensors"),
    )

    for visible_env_ids, expected_indices in selections:
        selected_offsets = torch.from_numpy(offsets[np.repeat(visible_env_ids, 2)])
        expected_positions = translations[expected_indices] + selected_offsets
        expected_xforms = torch.cat((expected_positions, orientations[expected_indices]), dim=1)
        expected_scales = scales[expected_indices]
        marker.render(viewer, visible_env_ids=visible_env_ids, num_envs=4)
        call = viewer.instances[-1]
        assert call["hidden"] is False
        np.testing.assert_allclose(call["xforms"], expected_xforms.numpy(), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(call["scales"], expected_scales.numpy(), rtol=0.0, atol=0.0)

    current_state = (
        marker.translations,
        marker.orientations,
        marker.scales,
        marker.marker_indices,
    )
    for current, source, expected in zip(current_state, source_state, source_values):
        assert current is source
        assert torch.equal(current, expected)
    assert len(viewer.instances) == 2


def test_newton_marker_render_hides_empty_env_selection(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=torch.zeros((8, 3), dtype=torch.float32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=[], num_envs=4)

    assert len(viewer.instances) == 1
    assert viewer.instances[0]["hidden"] is True


def test_newton_marker_render_defaults_to_first_prototype(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=torch.zeros((4, 3), dtype=torch.float32),
    )
    marker.orientations = None
    marker.scales = None
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=None, num_envs=4)

    visible_instances = [call for call in viewer.instances if not call["hidden"]]
    assert len(visible_instances) == 1
    assert visible_instances[0]["batch_name"] == "/Visuals/marker::test/arrow"
    assert visible_instances[0]["xforms"][:, 3:].tolist() == [[0.0, 0.0, 0.0, 1.0]] * 4
    assert visible_instances[0]["scales"].tolist() == [[1.0, 1.0, 1.0]] * 4
    hidden_batches = [call["batch_name"] for call in viewer.instances if call["hidden"]]
    assert hidden_batches == ["/Visuals/marker::test/sphere"]


def test_newton_marker_render_keeps_global_batch_unmodified(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(
        monkeypatch,
        np.array(
            [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )
    translations = torch.arange(3, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow"],
        translations=translations,
        marker_indices=torch.zeros(3, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

    marker.render(viewer, visible_env_ids=[1, 3], num_envs=4)

    assert viewer.instances[0]["xforms"][:, 0].tolist() == [0.0, 1.0, 2.0]


def test_newton_marker_render_routes_instances_by_prototype(monkeypatch: pytest.MonkeyPatch):
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    translations = torch.arange(4, dtype=torch.float32).unsqueeze(1).repeat(1, 3)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere"],
        translations=translations,
        marker_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

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
    world_offsets = _patch_newton_marker_render_deps(monkeypatch)
    marker = _make_newton_marker_for_render(
        marker_names=["arrow", "sphere", "frame"],
        translations=torch.zeros((3, 3), dtype=torch.float32),
        marker_indices=torch.zeros(3, dtype=torch.int32),
    )
    viewer = _FakeNewtonMarkerViewer(world_offsets)

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
