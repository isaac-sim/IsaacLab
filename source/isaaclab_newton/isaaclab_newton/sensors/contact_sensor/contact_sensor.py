# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from newton.sensors import MatchKind
from newton.sensors import SensorContact as NewtonContactSensor

import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor
from isaaclab.sim._impl.newton_manager import NewtonManager
from isaaclab.utils.helpers import deprecated

from .contact_sensor_data import ContactSensorData
from .contact_sensor_kernels import (
    compute_first_transition_kernel,
    copy_from_newton_kernel,
    reset_contact_sensor_kernel,
    update_contact_sensor_kernel,
)

if TYPE_CHECKING:
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

logger = logging.getLogger(__name__)


class ContactSensor(BaseContactSensor):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body or shape in the world frame.

    The sensor can be configured to report the contact forces on a set of sensors (bodies or shapes)
    against specific filter objects using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is
    useful when you want to report the contact forces between the sensors and a specific set of objects
    in the scene. The data can be accessed using the :attr:`ContactSensorData.force_matrix_w`.

    .. _Newton SensorContact: https://newton-physics.github.io/newton/api/_generated/newton.sensors.SensorContact.html
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self.num_sensors}\n"
            f"\tsensor names      : {self.sensor_names}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int | None:
        return self._num_sensors

    @property
    def data(self) -> ContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_sensors(self) -> int:
        """Number of sensors (bodies or shapes with contact sensing attached)."""
        return self._num_sensors

    @property
    def sensor_names(self) -> list[str] | None:
        """Ordered names of sensors (shapes or bodies with contact sensing attached)."""
        return self._sensor_names

    @property
    def filter_object_names(self) -> list[str] | None:
        return self._filter_object_names

    @property
    def num_filter_objects(self) -> int:
        """Number of filter objects (counterparts) for contact filtering."""
        return self._num_filter_objects

    @property
    def contact_view(self) -> NewtonContactSensor:
        """View for the contact forces captured (Newton)."""
        return NewtonManager._newton_contact_sensors[self._sensor_key]

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None):
        # reset the timers and counters
        super().reset(env_ids, env_mask)

        # Resolve env_mask (same logic as base class)
        if env_ids is None and env_mask is None:
            env_mask = wp.full(self._num_envs, True, dtype=wp.bool, device=self._device)
        elif env_mask is None:
            from isaaclab.utils.warp.utils import make_mask_from_torch_ids

            env_mask = make_mask_from_torch_ids(self._num_envs, env_ids, device=self._device)

        # Compute num_filter_objects
        num_filter_objects = self._num_filter_objects

        # Reset contact sensor buffers via kernel
        wp.launch(
            reset_contact_sensor_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                self.cfg.history_length,
                num_filter_objects,
                env_mask,
                self._data._net_forces_w,
                self._data._net_forces_w_history,
                self._data._force_matrix_w,
            ],
            outputs=[
                self._data._current_air_time,
                self._data._last_air_time,
                self._data._current_contact_time,
                self._data._last_contact_time,
            ],
            device=self._device,
        )

    def find_sensors(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find sensors based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the sensor names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple containing the sensor mask (wp.array), names (list[str]), and indices (list[int]).
        """
        indices, names = string_utils.resolve_matching_names(name_keys, self.sensor_names, preserve_order)
        mask = wp.array([name in names for name in self.sensor_names], dtype=wp.bool, device=self._device)
        return mask, names, indices

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> wp.array:
        """Checks if sensors that have established contact within the last :attr:`dt` seconds.

        This function checks if the sensors have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the sensors are considered to be in contact.

        Note:
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean array indicating the sensors that have established contact within the last
            :attr:`dt` seconds. Shape is (N, S), where N is the number of environments and S is the
            number of sensors.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        wp.launch(
            compute_first_transition_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[float(dt + abs_tol), self._data._current_contact_time],
            outputs=[self._data._first_transition],
            device=self._device,
        )
        return self._data._first_transition

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> wp.array:
        """Checks if sensors that have broken contact within the last :attr:`dt` seconds.

        This function checks if the sensors have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the sensors are considered to not be in contact.

        Note:
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean array indicating the sensors that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, S), where N is the number of environments and S is the number of sensors.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )

        wp.launch(
            compute_first_transition_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[float(dt + abs_tol), self._data._current_air_time],
            outputs=[self._data._first_transition],
            device=self._device,
        )
        return self._data._first_transition

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        """Initializes the sensor-related handles and internal buffers."""
        # construct regex expression for the sensor names

        if self.cfg.filter_prim_paths_expr is not None or self.cfg.filter_shape_paths_expr is not None:
            self._generate_force_matrix = True
        else:
            self._generate_force_matrix = False

        sensor_body_regex = self.cfg.prim_path
        if self.cfg.shape_path is not None:
            sensor_shape_regex = "(" + "|".join(self.cfg.shape_path) + ")"
        else:
            sensor_shape_regex = None
        if self.cfg.filter_prim_paths_expr is not None:
            filter_object_body_regex = "(" + "|".join(self.cfg.filter_prim_paths_expr) + ")"
        else:
            filter_object_body_regex = None
        if self.cfg.filter_shape_paths_expr is not None:
            filter_object_shape_regex = "(" + "|".join(self.cfg.filter_shape_paths_expr) + ")"
        else:
            filter_object_shape_regex = None

        # Store the sensor key for later lookup
        self._sensor_key = (
            sensor_body_regex,
            sensor_shape_regex,
            filter_object_body_regex,
            filter_object_shape_regex,
        )

        NewtonManager.add_contact_sensor(
            body_names_expr=sensor_body_regex,
            shape_names_expr=sensor_shape_regex,
            contact_partners_body_expr=filter_object_body_regex,
            contact_partners_shape_expr=filter_object_shape_regex,
            prune_noncolliding=True,
        )

        self._create_buffers()

    def _create_buffers(self):
        # Get Newton sensor shape: (n_sensors * n_envs, n_counterparts)
        newton_shape = self.contact_view.shape

        # resolve the true count of sensors
        self._num_sensors = newton_shape[0] // self._num_envs

        # Check that number of sensors is an integer
        if newton_shape[0] % self._num_envs != 0:
            raise RuntimeError(
                "Number of sensors is not an integer multiple of the number of environments. Received:"
                f" {self._num_sensors} sensors and {self._num_envs} environments."
            )
        logger.info(f"Contact sensor initialized with {self._num_sensors} sensors.")

        # Assume homogeneous envs, i.e. all envs have the same number of sensors
        # Only get the names for the first env. Expected structure: /World/envs/env_.*/...
        def get_name(idx, match_kind):
            if match_kind == MatchKind.BODY:
                return NewtonManager._model.body_key[idx].split("/")[-1]
            if match_kind == MatchKind.SHAPE:
                return NewtonManager._model.shape_key[idx].split("/")[-1]
            return "MATCH_ANY"

        self._sensor_names = [get_name(idx, kind) for idx, kind in self.contact_view.sensing_objs]
        # Assumes the environments are processed in order.
        self._sensor_names = self._sensor_names[: self._num_sensors]
        self._filter_object_names = [get_name(idx, kind) for idx, kind in self.contact_view.counterparts]

        # Number of filter objects (counterparts minus the total column)
        self._num_filter_objects = max(newton_shape[1] - 1, 0)

        # Store reshaped Newton net_force view for copying data
        # Newton net_force shape: (n_sensors * n_envs, n_counterparts)
        # Reshaped to: (n_envs, n_sensors, n_counterparts)
        self._newton_forces_view = self.contact_view.net_force.reshape((self._num_envs, self._num_sensors, -1))

        # prepare data buffers
        logger.info(
            f"Creating buffers for contact sensor data with num_envs: {self._num_envs}, num_sensors:"
            f" {self._num_sensors}, num_filter_objects: {self._num_filter_objects}, history_length:"
            f" {self.cfg.history_length}, generate_force_matrix: {self._generate_force_matrix}, track_air_time:"
            f" {self.cfg.track_air_time}, track_pose: {self.cfg.track_pose}, device: {self._device}"
        )
        self._data.create_buffers(
            self._num_envs,
            self._num_sensors,
            self._num_filter_objects,
            self.cfg.history_length,
            self._generate_force_matrix,
            self.cfg.track_air_time,
            self.cfg.track_pose,
            self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data.

        Args:
            env_mask: Mask of the environments to update. None: update all environments.
        """
        # Copy data from Newton into owned buffers (respecting env_mask)
        # Launch with 3D for coalescing: dim=(num_envs, num_sensors, max(num_filter_objects, 1))
        wp.launch(
            copy_from_newton_kernel,
            dim=(self._num_envs, self._num_sensors, max(self._num_filter_objects, 1)),
            inputs=[
                env_mask,
                self._newton_forces_view,
            ],
            outputs=[
                self._data._net_forces_w,
                self._data._force_matrix_w,
            ],
            device=self._device,
        )

        # Update history and air/contact time tracking
        wp.launch(
            update_contact_sensor_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                self.cfg.history_length,
                self.cfg.force_threshold,
                env_mask,
                self._data._net_forces_w,
                self._timestamp,
                self._timestamp_last_update,
                self._data._net_forces_w_history,
                self._data._current_air_time,
                self._data._current_contact_time,
                self._data._last_air_time,
                self._data._last_contact_time,
            ],
            device=self._device,
        )

        # FIXME: Re-enable this when we have a non-physx rigid body view?
        # (tracked in https://github.com/newton-physics/newton/issues/1489)
        # obtain the pose of the sensor origin
        # if self.cfg.track_pose:
        #    pose = self.body_physx_view.get_transforms().view(-1, self._num_sensors, 7)[env_ids]
        #    pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
        #    self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        return
        # note: this invalidity happens because of isaac sim view callbacks
        # if self.body_physx_view is None:
        #    return
        # marker indices
        # 0: contact, 1: no contact
        # net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        # marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        # if self.cfg.track_pose:
        #    frame_origins: torch.Tensor = self._data.pos_w
        # else:
        #    pose = self.body_physx_view.get_transforms()
        #    frame_origins = pose.view(-1, self._num_sensors, 7)[:, :, :3]
        # visualize
        # self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        # TODO: invalidate NewtonManager if necessary

    """
    Renamed
    """

    @property
    @deprecated("use num_sensors")
    def num_bodies(self) -> int:
        return self.num_sensors

    @property
    @deprecated("use sensor_names")
    def body_names(self) -> list[str] | None:
        return self.sensor_names

    @deprecated("use find_sensors")
    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        return self.find_sensors(name_keys, preserve_order)
