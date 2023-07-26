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
        # check that config is valid
        if self.cfg.history_length < 0:
            raise ValueError("History length must be greater than 0.")
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # visualization markers
        self.contact_visualizer = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self._view._regex_prim_paths}': \n"
            f"\tview type : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors: {self.count}\n"
            f"\tnumber of bodies : {self.num_bodies}"
        )

    """
    Properties
    """

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

    def initialize(self, env_prim_path: str):
        # create a rigid prim view for the sensor
        prim_paths_expr = f"{env_prim_path}/{self.cfg.prim_path_expr}"
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
        self._count = len(prim_utils.find_matching_prim_paths(f"{env_prim_path}"))
        self._num_bodies = self._view.count // self.count

        # initialize the base class
        super().initialize(env_prim_path)

        # fill the data buffer
        self._data.pos_w = torch.zeros(self.count, self._num_bodies, 3, device=self._device)
        self._data.quat_w = torch.zeros(self.count, self._num_bodies, 4, device=self._device)
        self._data.last_air_time = torch.zeros(self.count, self._num_bodies, device=self._device)
        self._data.current_air_time = torch.zeros(self.count, self._num_bodies, device=self._device)
        self._data.net_forces_w = torch.zeros(self.count, self._num_bodies, 3, device=self._device)
        self._data.net_forces_w_history = torch.zeros(
            self.count, self.cfg.history_length + 1, self._num_bodies, 3, device=self._device
        )
        # force matrix: (num_sensors, num_bodies, num_shapes, num_filter_shapes, 3)
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_shapes = self._view._contact_view.num_shapes // self._num_bodies
            num_filters = self._view._contact_view.num_filters
            self._data.force_matrix_w = torch.zeros(
                self.count, self._num_bodies, num_shapes, num_filters, 3, device=self._device
            )

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
        # reset accumulative data buffers
        self._data.current_air_time[env_ids] = 0.0
        self._data.last_air_time[env_ids] = 0.0
        self._data.net_forces_w[env_ids] = 0.0
        # reset the data history
        self._data.net_forces_w_history[env_ids] = 0.0
        # Set all reset sensors to not outdated since their value won't be updated till next sim step.
        self._is_outdated[env_ids] = False

    def debug_vis(self):
        # visualize the contacts
        if self.cfg.debug_vis:
            if self.contact_visualizer is None:
                prim_path = stage_utils.get_next_free_path("/Visuals/ContactSensor")
                self.contact_visualizer = VisualizationMarkers(prim_path, cfg=CONTACT_SENSOR_MARKER_CFG)
            # marker indices
            # 0: contact, 1: no contact
            net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
            marker_indices = torch.where(net_contact_force_w > 1.0, 0, 1)
            # check if prim is visualized
            self.contact_visualizer.visualize(self._data.pos_w.view(-1, 3), marker_indices=marker_indices.view(-1))

    """
    Implementation.
    """

    def _update_buffers(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self.count:
            env_ids = ...

        # obtain the poses of the sensors TODO decide if we really to track poses
        pose = self._view._physics_view.get_transforms()
        self._data.pos_w[env_ids] = pose.view(-1, self._num_bodies, 7)[env_ids, :, :3]
        self._data.quat_w[env_ids] = pose.view(-1, self._num_bodies, 7)[env_ids, :, 3:]

        # obtain the contact forces
        # TODO: We are handling the indexing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self._view._contact_view._physics_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]

        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            num_shapes = self._view._contact_view.num_shapes // self._num_bodies
            num_filters = self._view._contact_view.num_filters
            force_matrix_w = self._view.get_contact_force_matrix(dt=self._sim_physics_dt, clone=False)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_shapes, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]

        # update contact force history
        previous_net_forces_w = self._data.net_forces_w_history.clone()
        self._data.net_forces_w_history[env_ids, 0, :, :] = self._data.net_forces_w[env_ids, :, :]
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids, 1:, :, :] = previous_net_forces_w[env_ids, :-1, :, :]

        # contact state
        # -- time elapsed since last update
        # since this function is called every frame, we can use the difference to get the elapsed time
        elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
        # -- check contact state of bodies
        is_contact = torch.norm(self._data.net_forces_w[env_ids, 0, :, :], dim=-1) > 1.0
        is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
        # -- update ongoing timer for bodies air
        self._data.current_air_time[env_ids] += elapsed_time.unsqueeze(-1)
        # -- update time for the last time bodies were in contact
        self._data.last_air_time[env_ids] = self._data.current_air_time[env_ids] * is_first_contact
        # -- increment timers for bodies that are not in contact
        self._data.current_air_time[env_ids] *= ~is_contact
