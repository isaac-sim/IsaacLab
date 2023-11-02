# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import logging

from omni.isaac.kit import SimulationApp

# launch the simulator
config = {"headless": True}
simulation_app = SimulationApp(config)

# disable matplotlib debug messages
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

"""Rest everything follows."""


import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import unittest

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects.cuboid import DynamicCuboid
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.compat.utils.kit as kit_utils
from omni.isaac.orbit.compat.sensors.height_scanner import HeightScanner, HeightScannerCfg
from omni.isaac.orbit.compat.sensors.height_scanner.utils import create_points_from_grid, plot_height_grid
from omni.isaac.orbit.utils.math import convert_quat
from omni.isaac.orbit.utils.timer import Timer


class TestHeightScannerSensor(unittest.TestCase):
    """Test fixture for checking height scanner interface."""

    @classmethod
    def tearDownClass(cls):
        """Closes simulator after running all test fixtures."""
        simulation_app.close()

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Simulation time-step
        self.dt = 0.1
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
        # Set camera view
        set_camera_view(eye=[5.0, 5.0, 5.0], target=[0.0, 0.0, 0.0])
        # Spawn things into stage
        self._populate_scene()

        # Add height scanner
        # -- create query points from a gri
        self.grid_size = (1.0, 0.6)
        self.grid_resolution = 0.1
        scan_points = create_points_from_grid(self.grid_size, self.grid_resolution)
        # -- create sensor instance
        scanner_config = HeightScannerCfg(
            sensor_tick=0.0,
            offset=(0.0, 0.0, 0.0),
            points=scan_points,
            max_distance=0.45,
        )
        self.height_scanner = HeightScanner(scanner_config)
        # -- spawn sensor
        self.height_scanner.spawn("/World/heightScanner")
        self.height_scanner.set_visibility(True)

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        self.sim.stop()
        self.sim.clear()

    def test_height_scanner_visibility(self):
        """Checks that height map visibility method works."""
        # Play simulator
        self.sim.reset()
        # Setup the stage
        self.height_scanner.initialize()
        # flag for visualizing sensor
        toggle = True

        # Simulate physics
        for i in range(100):
            # set visibility
            if i % 100 == 0:
                toggle = not toggle
                print(f"Setting visibility: {toggle}")
                self.height_scanner.set_visibility(toggle)
            # perform rendering
            self.sim.step()
            # compute yaw -> quaternion
            yaw = 10 * i
            quat = tf.Rotation.from_euler("z", yaw, degrees=True).as_quat()
            quat = convert_quat(quat, "wxyz")
            # update sensor
            self.height_scanner.update(self.dt, [0.0, 0.0, 0.5], quat)

    def test_static_height_scanner(self):
        """Checks that static height map scanner is set correctly and provides right measurements."""
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(test_dir, "output", "height_scan", "static")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        # Play simulator
        self.sim.reset()
        # Setup the stage
        self.height_scanner.initialize()

        # Simulate physics
        for i in range(5):
            # perform rendering
            self.sim.step()
            # update camera
            self.height_scanner.update(self.dt, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
            # print state
            print(self.height_scanner)
            # create figure
            fig = plt.figure()
            ax = plt.gca()
            # plot the scanned distance
            caxes = plot_height_grid(self.height_scanner.data.hit_distance, self.grid_size, self.grid_resolution, ax=ax)
            fig.colorbar(caxes, ax=ax)
            # add grid
            ax.grid(color="w", linestyle="--", linewidth=1)
            # disreset figure
            plt.savefig(os.path.join(plot_dir, f"{i:03d}.png"), bbox_inches="tight", pad_inches=0.1)
            plt.close()

    def test_dynamic_height_scanner(self):
        """Checks that height map scanner works when base frame is rotating."""
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(test_dir, "output", "height_scan", "dynamic")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        # Play simulator
        self.sim.reset()
        # Setup the stage
        self.height_scanner.initialize()

        # Simulate physics
        for i in range(5):
            # perform rendering
            self.sim.step()
            # compute yaw -> quaternion
            yaw = 10 * i
            quat = tf.Rotation.from_euler("z", yaw, degrees=True).as_quat()
            quat = convert_quat(quat, "wxyz")
            # update sensor
            self.height_scanner.update(self.dt, [0.0, 0.0, 0.5], quat)
            # create figure
            fig = plt.figure()
            ax = plt.gca()
            # plot the scanned distance
            caxes = plot_height_grid(self.height_scanner.data.hit_distance, self.grid_size, self.grid_resolution, ax=ax)
            fig.colorbar(caxes, ax=ax)
            # add grid
            ax.grid(color="w", linestyle="--", linewidth=1)
            # disreset figure
            plt.savefig(os.path.join(plot_dir, f"{i:03d}.png"), bbox_inches="tight", pad_inches=0.1)
            plt.close()

    def test_height_scanner_filtering(self):
        """Checks that static height map scanner filters out the ground prim.

        The first time, all the cube prims are ignored. After that, they are ignored one-by-one cyclically.
        """
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        plot_dir = os.path.join(test_dir, "output", "height_scan", "filter")
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir, exist_ok=True)

        # Configure filter prims
        self.height_scanner.set_filter_prims([f"/World/Cube/cube{i:02d}" for i in range(4)])

        # Play simulator
        self.sim.reset()
        # Setup the stage
        self.height_scanner.initialize()

        # Simulate physics
        for i in range(6):
            # set different filter prims
            if i > 0:
                cube_id = i - 1
                self.height_scanner.set_filter_prims([f"/World/Cube/cube{cube_id:02}"])
            if i > 4:
                self.height_scanner.set_filter_prims(None)
            # perform rendering
            self.sim.step()
            # update sensor
            self.height_scanner.update(self.dt, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])
            # create figure
            fig = plt.figure()
            ax = plt.gca()
            # plot the scanned distance
            caxes = plot_height_grid(self.height_scanner.data.hit_distance, self.grid_size, self.grid_resolution, ax=ax)
            fig.colorbar(caxes, ax=ax)
            # add grid
            ax.grid(color="w", linestyle="--", linewidth=1)
            # disreset figure
            plt.savefig(os.path.join(plot_dir, f"{i:03d}.png"), bbox_inches="tight", pad_inches=0.1)
            plt.close()

    def test_scanner_throughput(self):
        """Measures the scanner throughput while using scan points used for ANYmal robot."""
        # Add height scanner
        # -- create sensor instance
        scanner_config = HeightScannerCfg(
            sensor_tick=0.0,
            offset=(0.0, 0.0, 0.0),
            points=self._compute_anymal_height_scanner_points(),
            max_distance=1.0,
        )
        self.anymal_height_scanner = HeightScanner(scanner_config)
        # -- spawn sensor
        self.anymal_height_scanner.spawn("/World/heightScannerAnymal")
        self.anymal_height_scanner.set_visibility(True)

        # Play simulator
        self.sim.reset()
        # Setup the stage
        self.anymal_height_scanner.initialize()

        # Turn rendering on
        self.anymal_height_scanner.set_visibility(True)
        # Simulate physics
        for i in range(2):
            # perform rendering
            self.sim.step()
            # update sensor
            with Timer(f"[No Vis  , Step {i:02d}]: Scanning time for {scanner_config.points.shape[0]} points"):
                self.anymal_height_scanner.update(self.dt, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])

        # Turn rendering off
        self.anymal_height_scanner.set_visibility(False)
        # Simulate physics
        for i in range(2):
            # perform rendering
            self.sim.step()
            # update sensor
            with Timer(f"[With Vis, Step {i:02d}] Scanning time for {scanner_config.points.shape[0]} points"):
                self.anymal_height_scanner.update(self.dt, [0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.0])

    """
    Helper functions.
    """

    @staticmethod
    def _populate_scene():
        """Add prims to the scene."""
        # Ground-plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane")
        # Lights-1
        prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
        # Lights-2
        prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))
        # Cubes
        num_cubes = 4
        for i in range(num_cubes):
            # resolve position to put them on vertex of a regular polygon
            theta = 2 * np.pi / num_cubes
            c, s = np.cos(theta * i), np.sin(theta * i)
            rotm = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            position = np.matmul(rotm, np.asarray([0.25, 0.25, 0.1 + 0.05 * i]))
            color = np.array([random.random(), random.random(), random.random()])
            # create prim
            _ = DynamicCuboid(
                prim_path=f"/World/Cube/cube{i:02d}", position=position, size=position[2] * 2, color=color
            )

    @staticmethod
    def _compute_anymal_height_scanner_points() -> np.ndarray:
        """Computes the query height-scan points relative to base frame of robot.

        Returns:
            A numpy array of shape (N, 3) comprising of quey scan points.
        """
        # offset from the base frame - over each leg
        offsets = [[0.45, 0.3, 0.0], [-0.46, 0.3, 0.0], [0.45, -0.3, 0.0], [-0.46, -0.3, 0.0]]
        offsets = np.asarray(offsets)
        # local grid over each offset point
        measurement_points = [
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [-0.1, 0.0, 0.0],
            [0.0, -0.1, 0.0],
            [0.1, 0.1, 0.0],
            [-0.1, 0.1, 0.0],
            [0.1, -0.1, 0.0],
            [-0.1, -0.1, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.2, 0.0],
            [-0.2, 0.0, 0.0],
            [0.0, -0.2, 0.0],
            [0.2, 0.2, 0.0],
            [-0.2, 0.2, 0.0],
            [0.2, -0.2, 0.0],
            [-0.2, -0.2, 0.0],
            [0.3, 0.0, 0.0],
            [0.3, 0.1, 0.0],
            [0.3, 0.2, 0.0],
            [0.3, -0.1, 0.0],
            [0.3, -0.2, 0.0],
            [-0.3, 0.0, 0.0],
            [-0.3, 0.1, 0.0],
            [-0.3, 0.2, 0.0],
            [-0.3, -0.1, 0.0],
            [-0.3, -0.2, 0.0],
            [0.0, 0.0, 0.0],
        ]
        measurement_points = np.asarray(measurement_points)
        # create a joined list over offsets and local measurement points
        # we use broadcasted addition to make this operation quick
        scan_points = (offsets[:, None, :] + measurement_points).reshape(-1, 3)

        return scan_points


if __name__ == "__main__":
    unittest.main()
