# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from typing import Sequence

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.prims import RigidPrimView
from pxr import PhysxSchema

from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import CONTACT_SENSOR_MARKER_CFG

from ..sensor_base import SensorBase
from .contact_sensor_cfg import ContactSensorCfg
from .contact_sensor_data import ContactSensorData


class ContactSensor(SensorBase):
    """A contact reporting sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg (ContactSensorCfg): The configuration parameters.
        """
        # store config
        self.cfg = cfg
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = ContactSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self._view._regex_prim_paths}': \n"
            f"\tview type : {self._view.__class__}\n"
            f"\tupdate period (s) : {self._update_period}\n"
            f"\tnumber of sensors: {self.count}"
            f"\tnumber of bodies : {len(self.num_bodies)}"
        )

    """
    Properties
    """

    @property
    def data(self) -> ContactSensorData:
        """Data related to contact sensor."""
        return self._data

    @property
    def count(self) -> int:
        """Number of prims encapsulated."""
        return self._count

    @property
    def num_bodies(self) -> int:
        """Number of prims encapsulated."""
        return self._num_bodies

    """
    Operations
    """

    def spawn(self, prim_paths_expr: str):
        """Spawns the sensor in the scene.

        In this case, it creates a contact sensor API on the rigid body prims that are
        captured by the given prim path expression.

        Example: If input is ``/World/Robot/Link*``, then the contact sensor API is
        created on all rigid body prims that match the expression (e.g. ``/World/Robot/Link1``,
        ``/World/Robot/Link2``, etc.) and their children.

        Raises:
            RuntimeError: No rigid bodies found in the scene at the given prim path.
        """
        prim_paths = prim_utils.find_matching_prim_paths(prim_paths_expr)
        num_bodies = 0
        # -- create contact sensor api
        for prim_path in prim_paths:
            # create contact sensor
            prim = prim_utils.get_prim_at_path(prim_path)
            # iterate over the prim and apply contact sensor API to all rigid bodies
            for link_prim in prim.GetChildren() + [prim]:
                if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                    # add contact report API with threshold of zero
                    cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
                    cr_api.CreateThresholdAttr().Set(0)
                    # increment number of contact sensors made
                    num_bodies += 1
        # -- store the number of bodies found to shape the data buffer
        self._num_bodies = num_bodies
        # check that we spawned at least one contact sensor
        if num_bodies == 0:
            raise RuntimeError(f"No rigid bodies found for the given prim path expression: {prim_paths_expr}")

    def initialize(self, prim_paths_expr: str):
        # create a rigid prim view for the sensor
        self._view = RigidPrimView(
            prim_paths_expr=prim_paths_expr,
            reset_xform_properties=False,
            track_contact_forces=True,
            contact_filter_prim_paths_expr=self.cfg.filter_prim_paths_expr,
            prepare_contact_sensors=False,
            disable_stablization=True,
        )
        self._view.initialize()
        # resolve the true count of bodies
        self._count = self._view.count // self._num_bodies
        # initialize the base class
        super().initialize(prim_paths_expr)

        # fill the data buffer
        self._data.pos_w = torch.zeros(self.count, self._num_bodies, 3, device=self._device)
        self._data.quat_w = torch.zeros(self.count, self._num_bodies, 4, device=self._device)
        self._data.last_air_time = torch.zeros(self.count, self._num_bodies, device=self._device)
        self._data.current_air_time = torch.zeros(self.count, self._num_bodies, device=self._device)
        self._data.net_forces_w = torch.zeros(self.count, self._num_bodies, 3, device=self._device)
        # force matrix: (num_sensors, num_bodies, num_shapes, num_filter_shapes, 3)
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_shapes = self._view._contact_view.num_shapes // self._num_bodies
            num_filters = self._view._contact_view.num_filters
            self._data.force_matrix_w = torch.zeros(
                self.count, self._num_bodies, num_shapes, num_filters, 3, device=self._device
            )

        # visualization of the contact sensor
        prim_path = stage_utils.get_next_free_path("/Visuals/ContactSensor")
        self.contact_visualizer = VisualizationMarkers(prim_path, cfg=CONTACT_SENSOR_MARKER_CFG)

    def reset_buffers(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals.

        Args:
            env_ids (Sequence[int], optional): The sensor ids to reset. Defaults to None.
        """
        # reset the timers and counters
        super().reset_buffers(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = ...
        # reset the data buffers
        self._data.net_forces_w[env_ids] = 0.0
        self._data.force_matrix_w[env_ids] = 0.0
        self._data.current_air_time[env_ids] = 0.0
        self._data.last_air_time[env_ids] = 0.0

    def debug_vis(self):
        # visualize the point hits
        if self.cfg.debug_vis:
            # marker indices
            # 0: contact, 1: no contact
            net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
            marker_indices = torch.where(net_contact_force_w > 1.0, 0, 1)
            # check if prim is visualized
            self.contact_visualizer.visualize(self._data.pos_w.view(-1, 3), marker_indices=marker_indices.view(-1))

    """
    Implementation.
    """

    def _buffer(self, env_ids: Sequence[int] | None = None):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_world_poses(clone=False)
        self._data.pos_w[env_ids] = pos_w.view(-1, self._num_bodies, 3)[env_ids]
        self._data.quat_w[env_ids] = quat_w.view(-1, self._num_bodies, 4)[env_ids]
        # obtain the contact forces
        # TODO: We are handling the indicing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self._view.get_net_contact_forces(dt=self._sim_physics_dt, clone=False)
        self._data.net_forces_w[env_ids] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]
        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_shapes = self._view._contact_view.num_shapes // self._num_bodies
            num_filters = self._view._contact_view.num_filters
            force_matrix_w = self._view.get_contact_force_matrix(dt=self._sim_physics_dt, clone=False)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_shapes, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]

        # contact state
        # -- time elapsed since last update
        # since this function is called every frame, we can use the difference to get the elapsed time
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        # -- check contact state of bodies
        is_bodies_contact = torch.norm(self._data.net_forces_w[env_ids], dim=-1) > 1.0
        is_bodies_first_contact = (self._data.current_air_time[env_ids] > 0) * is_bodies_contact
        # -- update ongoing timer for bodies air
        self._data.current_air_time[env_ids] += elapsed_time.unsqueeze(-1)
        # -- update time for the last time bodies were in contact
        self._data.last_air_time[env_ids] = self._data.current_air_time[env_ids] * is_bodies_first_contact
        # -- increment timers for bodies that are not in contact
        self._data.current_air_time[env_ids] *= ~is_bodies_contact
