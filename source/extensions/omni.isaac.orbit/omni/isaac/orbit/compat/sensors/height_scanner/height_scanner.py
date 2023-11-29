# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
from dataclasses import dataclass
from typing import Sequence

import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import XFormPrim

from omni.isaac.orbit.utils.math import convert_quat

from ..sensor_base import SensorBase
from .height_scanner_cfg import HeightScannerCfg
from .height_scanner_marker import HeightScannerMarker


@dataclass
class HeightScannerData:
    """Data container for the height-scanner sensor."""

    position: np.ndarray = None
    """Position of the sensor origin in world frame."""
    orientation: np.ndarray = None
    """Orientation of the sensor origin in quaternion (w, x, y, z) in world frame."""
    hit_points: np.ndarray = None
    """The end point locations of ray-casted rays. Shape is (N, 3), where N is
    the number of scan points."""
    hit_distance: np.ndarray = None
    """The ray-cast travel distance from query point. Shape is (N,), where N is
    the number of scan points."""
    hit_status: np.ndarray = None
    """Whether the ray hit an object or not. Shape is (N,), where N is
    the number of scan points.

    It is set to ``1`` if the ray hit an object, and ``0`` otherwise.
    """


class HeightScanner(SensorBase):
    """A two-dimensional height-map scan sensor.

    A local map is often required to plan a robot's motion over a limited time horizon. For mobile systems,
    often we care about the terrain for locomotion. The height-map, also called elevation map, simplifies the
    terrain as a two-dimensional surface. Each grid-cell represents the height of the terrain.

    Unlike algorithms which fuse depth measurements to create an elevation map :cite:p:`frankhauser2018probabilistic`,
    in this method we directly use the PhysX API for ray-casting and query the height of the terrain from a set
    of query scan points. These points represent the location of the grid cells.

    The height-scanner uses PhysX for ray-casting to collision bodies. To prevent the casting to certain prims
    in the scene (such as the robot on which height-scanner is present), one needs to provide the names of the
    prims to not check collision with as a part of the dictionary config.

    The scanner offset :math:`(x_o, y_o, z_o)` is the offset of the sensor from the frame it is attached to.
    During the :meth:`update` or :meth:`buffer`, the pose of the mounted frame needs to be provided.

    If visualization is enabled, rays that have a hit are displayed in red, while a miss is displayed in blue.
    During a miss, the point's distance is set to the maximum ray-casting distance.

    """

    def __init__(self, cfg: HeightScannerCfg):
        """Initializes the scanner object.

        Args:
            cfg: The configuration parameters.
        """
        # TODO: Use generic range sensor from Isaac Sim?
        # Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_range_sensor.html#isaac-sim-generic-range-sensor-example
        # store inputs
        self.cfg = cfg
        # initialize base class
        super().__init__(self.cfg.sensor_tick)

        # Points to query ray-casting from
        self._scan_points = np.asarray(self.cfg.points)
        # If points are 2D, add dimension along z (z=0 relative to the sensor frame)
        if self._scan_points.shape[1] == 2:
            self._scan_points = np.pad(self._scan_points, [(0, 0), (0, 1)], constant_values=0)

        # Flag to check that sensor is spawned.
        self._is_spawned = False
        # Whether to visualize the scanner points. Defaults to False.
        self._visualize = False
        # Xform prim for the sensor rig
        self._sensor_xform: XFormPrim = None
        # Create empty variables for storing output data
        self._data = HeightScannerData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        # message for class
        return (
            f"Height Scanner @ '{self.prim_path}': \n"
            f"\ttick rate (s) : {self.sensor_tick}\n"
            f"\ttimestamp (s) : {self.timestamp}\n"
            f"\tframe         : {self.frame}\n"
            f"\tposition      : {self.data.position}\n"
            f"\torientation   : {self.data.orientation}\n"
            f"\t# of hits     : {np.sum(self.data.hit_status)} / {self._scan_points[0]}\n"
        )

    """
    Properties
    """

    @property
    def prim_path(self) -> str:
        """The path to the height-map sensor."""
        return self._sensor_xform.prim_path

    @property
    def data(self) -> HeightScannerData:
        """Data related to height scanner."""
        return self._data

    """
    Helpers
    """

    def set_visibility(self, visible: bool):
        """Enables drawing of the scan points in the viewport.

        Args:
            visible: Whether to draw scan points or not.
        """
        # copy argument
        self._visualize = visible
        # set visibility
        self._height_scanner_vis.set_visibility(visible)

    def set_filter_prims(self, names: list[str]):
        """Set the names of prims to ignore ray-casting collisions with.

        If None is passed into argument, then no filtering is performed.

        Args:
            names: A list of prim names to ignore ray-cast collisions with.
        """
        # default
        if names is None:
            self.cfg.filter_prims = list()
        else:
            # set into the class
            self.cfg.filter_prims = names

    """
    Operations
    """

    def spawn(self, parent_prim_path: str):  # noqa: D102
        # Check if sensor is already spawned
        if self._is_spawned:
            raise RuntimeError(f"The height scanner sensor instance has already been spawned at: {self.prim_path}.")
        # Create sensor prim path
        prim_path = stage_utils.get_next_free_path(f"{parent_prim_path}/HeightScan_Xform")
        # Create the sensor prim
        prim_utils.create_prim(prim_path, "XForm")
        self._sensor_xform = XFormPrim(prim_path, translation=self.cfg.offset)
        # Create visualization marker
        # TODO: Move this inside the height-scan prim to make it cleaner?
        vis_prim_path = stage_utils.get_next_free_path("/World/Visuals/HeightScan")
        self._height_scanner_vis = HeightScannerMarker(vis_prim_path, count=self._scan_points.shape[0], radius=0.02)
        # Set spawning to true
        self._is_spawned = True

    def initialize(self):  # noqa: D102
        # Check that sensor has been spawned
        if not self._is_spawned:
            raise RuntimeError("Height scanner sensor must be spawned first. Please call `spawn(...)`.")
        # Initialize Xform class
        self._sensor_xform.initialize()
        # Acquire physx ray-casting interface
        self._physx_query_interface = omni.physx.get_physx_scene_query_interface()
        # Since height scanner is fictitious sensor, we have no schema config to set in this case.
        # Initialize buffers
        self.reset()

    def reset(self):  # noqa: D102
        # reset the timestamp
        super().reset()
        # reset the buffer
        self._data.hit_points = np.empty(shape=(self._scan_points.shape))
        self._data.hit_distance = np.empty(shape=(self._scan_points.shape[0]))
        self._data.hit_status = np.zeros(shape=(self._scan_points.shape[0]))
        self._data.position = None
        self._data.orientation = None

    def update(self, dt: float, pos: Sequence[float], quat: Sequence[float]):
        """Updates the buffers at sensor frequency.

        Args:
            dt: The simulation time-step.
            pos: Position of the frame to which the sensor is attached.
            quat: Quaternion (w, x, y, z) of the frame to which the sensor is attached.
        """
        super().update(dt, pos, quat)

    def buffer(self, pos: Sequence[float], quat: Sequence[float]):
        """Fills the buffers of the sensor data.

        This function uses the input position and orientation to compute the ray-casting queries
        and fill the buffers. If a collision is detected, then the hit distance is stored in the buffer.
        Otherwise, the hit distance is set to the maximum value specified in the configuration.

        Args:
            pos: Position of the frame to which the sensor is attached.
            quat: Quaternion (w, x, y, z) of the frame to which the sensor is attached.
        """
        # convert to numpy for sanity
        pos = np.asarray(pos)
        quat = np.asarray(quat)
        # account for the offset of the sensor
        self._data.position, self._data.orientation = (pos + self.cfg.offset, quat)

        # construct 3D rotation matrix for grid points
        # TODO: Check if this is the most generic case. It ignores the base pitch and roll.
        tf_rot = tf.Rotation.from_quat(convert_quat(self.data.orientation, "xyzw"))
        rpy = tf_rot.as_euler("xyz", degrees=False)
        rpy[:2] = 0
        rotation = tf.Rotation.from_euler("xyz", rpy).as_matrix()
        # transform the scan points to world frame
        world_scan_points = np.matmul(rotation, self._scan_points.T).T + self.data.position

        # iterate over all the points and query ray-caster
        for i in range(world_scan_points.shape[0]):
            # set current query info to empty
            self._query_info = None
            # perform ray-cast to get distance of a point along (0, 0, -1)
            self._physx_query_interface.raycast_all(
                tuple(world_scan_points[i]),
                self.cfg.direction,
                self.cfg.max_distance,
                self._hit_report_callback,
            )
            # check if hit happened based on query info and add to data
            if self._query_info is not None:
                self._data.hit_status[i] = 1
                self._data.hit_distance[i] = self._query_info.distance
            else:
                self._data.hit_status[i] = 0
                self._data.hit_distance[i] = self.cfg.max_distance
            # add ray-end point (in world frame) to data
            self._data.hit_points[i] = world_scan_points[i] + np.array(self.cfg.direction) * self._data.hit_distance[i]

        # visualize the prim
        if self._visualize:
            self._height_scanner_vis.set_status(status=self._data.hit_status)
            self._height_scanner_vis.set_world_poses(positions=self._data.hit_points)

    """
    Private helpers
    """

    def _hit_report_callback(self, hit) -> bool:
        """A PhysX callback to filter out hit-reports that are on the collision bodies.

        Returns:
            If True, continue casting the ray. Otherwise, stop and report the result.
        """
        # unset the query info
        self._query_info = None
        # get ray's current contact rigid body
        current_hit_body = hit.rigid_body
        # check if the collision body is in the filtering list
        if current_hit_body in self.cfg.filter_prims:
            # continue casting the ray
            return True
        else:
            # set the hit information
            self._query_info = hit
            return False
