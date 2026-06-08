# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from newton.sensors import SensorContact as NewtonContactSensor

import isaaclab.utils.string as string_utils
from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager

from .contact_sensor_data import ContactSensorData
from .contact_sensor_kernels import (
    compute_first_transition_kernel,
    copy_from_newton_kernel,
    reset_contact_sensor_kernel,
    update_contact_sensor_kernel,
)

if TYPE_CHECKING:
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as BaseContactSensorCfg

    from .contact_sensor_cfg import ContactSensorCfg

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

    def __init__(self, cfg: BaseContactSensorCfg | ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg as BaseContactSensorCfg

        from .contact_sensor_cfg import ContactSensorCfg

        if isinstance(cfg, ContactSensorCfg):
            pass
        elif isinstance(cfg, BaseContactSensorCfg):
            cfg = ContactSensorCfg.from_base_cfg(cfg)
        else:
            raise TypeError(f"Invalid config: {cfg}")

        super().__init__(cfg)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # Defaults used before full initialization completes.
        self._num_sensors: int = 0
        self._sensor_names: list[str] = []
        self._filter_object_names: list[str] = []
        self._num_filter_objects: int = 0
        self._init_error: str | None = None

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
        """Ordered names of filter objects (counterparts) for contact filtering."""
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
        # resolve mask via the shared helper (uses self._reset_mask, persistent across calls).
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        # reset the timers and counters
        super().reset(None, env_mask)

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

    def find_sensors(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find sensors based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the sensor names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple containing the sensor indices and names.
        """
        sensor_names = self.sensor_names
        if not sensor_names:
            if self._init_error is not None:
                raise ValueError(f"ContactSensor initialization failed: {self._init_error}")
            raise ValueError(
                "ContactSensor metadata is unavailable. Expected sensor names to be populated during"
                " PHYSICS_READY initialization."
            )
        return string_utils.resolve_matching_names(name_keys, sensor_names, preserve_order)

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> ProxyArray:
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
            A float array (1.0/0.0) indicating the sensors that have established contact within the
            last :attr:`dt` seconds. Shape is (N, S), where N is the number of environments and S is
            the number of sensors. The returned array is a shared internal buffer; it is invalidated
            by the next call to :meth:`compute_first_contact` or :meth:`compute_first_air`.

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
        return self._data._first_transition_ta

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> ProxyArray:
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
            A float array (1.0/0.0) indicating the sensors that have broken contact within the last
            :attr:`dt` seconds. Shape is (N, S), where N is the number of environments and S is the
            number of sensors. The returned array is a shared internal buffer; it is invalidated by
            the next call to :meth:`compute_first_contact` or :meth:`compute_first_air`.

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
        return self._data._first_transition_ta

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor-related handles and internal buffers."""
        super()._initialize_impl()

        if self.cfg.force_threshold is None:
            self.cfg.force_threshold = 0.0

        self._generate_force_matrix = bool(self.cfg.filter_prim_paths_expr or self.cfg.filter_shape_prim_expr)

        try:
            self._sensor_key = NewtonManager.add_contact_sensor(
                body_names_expr=self.cfg.prim_path if not self.cfg.sensor_shape_prim_expr else None,
                shape_names_expr=self.cfg.sensor_shape_prim_expr or None,
                contact_partners_body_expr=self.cfg.filter_prim_paths_expr or None,
                contact_partners_shape_expr=self.cfg.filter_shape_prim_expr or None,
            )

            self._create_buffers()
            self._init_error = None
        except Exception as err:
            self._init_error = (
                f"failed to initialize contact sensor for prim path '{self.cfg.prim_path}'"
                f" with sensor shape expr '{self.cfg.sensor_shape_prim_expr}': {err}"
            )
            raise RuntimeError(self._init_error) from err

    def _create_buffers(self):
        # Get Newton sensor count from total force: (n_sensors * n_envs)
        total_sensor_count = self.contact_view.total_force.shape[0]

        # resolve the true count of sensors
        self._num_sensors = total_sensor_count // self._num_envs

        # Check that number of sensors is an integer
        if total_sensor_count % self._num_envs != 0:
            raise RuntimeError(
                "Number of sensors is not an integer multiple of the number of environments. Received:"
                f" {total_sensor_count} sensors across {self._num_envs} environments."
            )
        if self._num_sensors == 0:
            raise RuntimeError(
                "Contact sensor matched zero sensing objects. This usually indicates a prim-path pattern mismatch"
                f" for expression '{self.cfg.prim_path}'."
            )
        logger.info(f"Contact sensor initialized with {self._num_sensors} sensors.")

        # Assume homogeneous envs, i.e. all envs have the same number of sensors
        # Only get the names for the first env. Expected structure: /World/envs/env_.*/...
        body_labels = self._get_model_labels("body")
        shape_labels = self._get_model_labels("shape")

        s_kind = self.contact_view.sensing_obj_type
        if s_kind == "body":
            s_labels = body_labels
        elif s_kind == "shape":
            s_labels = shape_labels
        else:
            raise RuntimeError(f"Unexpected Newton sensing_obj_type {s_kind!r}; expected 'body' or 'shape'.")
        self._sensor_names = [s_labels[i].split("/")[-1] for i in self.contact_view.sensing_obj_idx]
        # Assumes the environments are processed in order.
        self._sensor_names = self._sensor_names[: self._num_sensors]

        c_kind = self.contact_view.counterpart_type
        c_idx_per_sensor = self.contact_view.counterpart_indices
        if c_kind is None:
            if self._generate_force_matrix:
                raise RuntimeError("Filter expressions were configured but Newton reports no counterpart type.")
            self._filter_object_names = []
        else:
            if c_kind == "body":
                c_labels = body_labels
            elif c_kind == "shape":
                c_labels = shape_labels
            else:
                raise RuntimeError(f"Unexpected Newton counterpart_type {c_kind!r}; expected 'body' or 'shape'.")
            # Envs are homogeneous: every sensor row sees the same counterpart list. Take row 0.
            row0 = c_idx_per_sensor[0] if c_idx_per_sensor else []
            self._filter_object_names = [c_labels[i].split("/")[-1] for i in row0]
            if self._generate_force_matrix and not self._filter_object_names:
                logger.warning("Filter expressions matched zero counterpart objects; force matrix will be empty.")

        force_matrix = self.contact_view.force_matrix
        force_matrix_shape = force_matrix.shape if force_matrix is not None else (total_sensor_count, 0)
        # Number of filter objects.
        self._num_filter_objects = force_matrix_shape[1] if len(force_matrix_shape) > 1 else 0
        if self._num_filter_objects > 0 and force_matrix is None:
            raise RuntimeError("Filter counterparts present but Newton force_matrix is None.")

        # Store flat Newton force views for copying data. These may be non-contiguous
        # views, so the copy kernel indexes them without reshaping.
        self._newton_total_force_view = self.contact_view.total_force
        self._newton_force_matrix_view = force_matrix if self._num_filter_objects > 0 else None

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

    def _get_model_labels(self, kind: str) -> list[str]:
        """Return Newton model labels in a version-compatible way."""
        model = NewtonManager._model
        primary = f"{kind}_label"
        fallback = f"{kind}_key"
        labels = getattr(model, primary, None)
        if labels is None:
            labels = getattr(model, fallback, None)
        if labels is None:
            raise RuntimeError(f"Newton model does not expose '{primary}' or '{fallback}'.")
        return list(labels)

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
                self._num_sensors,
                self._newton_total_force_view,
                self._newton_force_matrix_view,
                self._timestamp,
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

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        return

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
    def body_names(self) -> list[str] | None:
        warnings.warn(
            "ContactSensor.body_names is deprecated; use ContactSensor.sensor_names instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.sensor_names
