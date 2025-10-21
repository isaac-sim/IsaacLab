# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, POSITION_GOAL_MARKER_CFG
from isaaclab.utils.math import random_orientation
from isaaclab.utils.timer import Timer


@pytest.fixture
def sim():
    """Create a blank new stage for each test."""
    # Simulation time-step
    dt = 0.01
    # Open a new stage
    stage_utils.create_new_stage()
    # Load kit helper
    sim_context = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="torch", device="cuda:0")
    yield sim_context
    # Cleanup
    sim_context.stop()
    sim_context.clear_instance()
    stage_utils.close_stage()


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
