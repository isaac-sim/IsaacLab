# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.math import random_orientation
from isaaclab.utils.timer import Timer


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


def test_rendering_without_visualizers_initializes_kit_backend(monkeypatch):
    """Rendering observations should still author USD markers when no visualizer is launched."""

    class _FakeSim:
        is_rendering = True
        visualizers = []

    marker = object.__new__(VisualizationMarkers)
    marker._backends = []

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", staticmethod(lambda: _FakeSim()))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_kit_backend", lambda self: self._backends.append("kit"))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_newton_backend", lambda self: self._backends.append("newton"))

    marker._ensure_backends_initialized()

    assert marker._backends == ["kit"]


def test_non_rendering_without_visualizers_defers_backend_initialization(monkeypatch):
    """Avoid creating render backends when neither rendering nor visualizers are active."""

    class _FakeSim:
        is_rendering = False
        visualizers = []

    marker = object.__new__(VisualizationMarkers)
    marker._backends = []

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", staticmethod(lambda: _FakeSim()))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_kit_backend", lambda self: self._backends.append("kit"))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_newton_backend", lambda self: self._backends.append("newton"))

    marker._ensure_backends_initialized()

    assert marker._backends == []


def test_kit_visualizer_initializes_kit_backend_when_rendering_flag_is_false(monkeypatch):
    """Keep the explicit Kit visualizer path independent of SimulationContext.is_rendering."""

    class _FakeKitVisualizer:
        cfg = type("Cfg", (), {"enable_markers": True})()

        def supports_markers(self):
            return True

        def pumps_app_update(self):
            return True

    class _FakeSim:
        is_rendering = False
        visualizers = [_FakeKitVisualizer()]

    marker = object.__new__(VisualizationMarkers)
    marker._backends = []

    monkeypatch.setattr(sim_utils.SimulationContext, "instance", staticmethod(lambda: _FakeSim()))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_kit_backend", lambda self: self._backends.append("kit"))
    monkeypatch.setattr(VisualizationMarkers, "_ensure_newton_backend", lambda self: self._backends.append("newton"))

    marker._ensure_backends_initialized()

    assert marker._backends == ["kit"]


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


def test_usd_marker_color(sim):
    """Test with marker from a USD with its color modified."""
    # create a marker
    config = FRAME_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_frames"
    config.markers["frame"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
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


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_visualization_time_based_on_prototypes(sim):
    """Test with time taken when number of prototypes is increased."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
    # number of frames
    num_frames = 4096

    # check that visibility is true
    assert test_marker.is_visible()
    # run with randomization of poses and indices
    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)
    # set the marker
    with Timer("Marker visualization with explicit indices") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
        # save the time
        time_with_marker_indices = timer.time_elapsed

    with Timer("Marker visualization with no indices") as timer:
        test_marker.visualize(translations=frame_translations)
        # save the time
        time_with_no_marker_indices = timer.time_elapsed

    # update the kit
    sim.step()
    # check that the time is less
    assert time_with_no_marker_indices < time_with_marker_indices


def test_visualization_time_based_on_visibility(sim):
    """Test with visibility of markers. When invisible, the visualize call should return."""
    # create a marker
    config = POSITION_GOAL_MARKER_CFG.copy()
    config.prim_path = "/World/Visuals/test_protos"
    test_marker = VisualizationMarkers(config)

    # play the simulation
    sim.reset()
    # number of frames
    num_frames = 4096

    # check that visibility is true
    assert test_marker.is_visible()
    # run with randomization of poses and indices
    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)
    # set the marker
    with Timer("Marker visualization") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
        # save the time
        time_with_visualization = timer.time_elapsed

    # update the kit
    sim.step()
    # make invisible
    test_marker.set_visibility(False)

    # check that visibility is false
    assert not test_marker.is_visible()
    # run with randomization of poses and indices
    frame_translations = torch.randn(num_frames, 3, device=sim.device)
    marker_indices = torch.randint(0, test_marker.num_prototypes, (num_frames,), device=sim.device)
    # set the marker
    with Timer("Marker no visualization") as timer:
        test_marker.visualize(translations=frame_translations, marker_indices=marker_indices)
        # save the time
        time_with_no_visualization = timer.time_elapsed

    # check that the time is less
    assert time_with_no_visualization < time_with_visualization
