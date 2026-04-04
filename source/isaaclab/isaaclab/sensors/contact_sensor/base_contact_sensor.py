# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

import isaaclab.utils.string as string_utils

from ..sensor_base import SensorBase
from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class BaseContactSensor(SensorBase):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body in the world frame.
    It relies on the physics engine's contact reporting API to be activated on the rigid bodies.

    To enable the contact reporter on a rigid body, please make sure to enable the
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors` on your
    asset spawner configuration. This will enable the contact reporter on all the rigid bodies
    in the asset.

    The sensor can be configured to report the contact forces on a set of bodies with a given
    filter pattern using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is useful
    when you want to report the contact forces between the sensor bodies and a specific set of
    bodies in the scene. The data can be accessed using the :attr:`ContactSensorData.force_matrix_w`.

    The PhysX backend only supports one-to-many filtered contact reporting: a single sensor
    body filtered against multiple partners. For many-to-many, create separate sensors per
    body. The Newton backend supports many-to-many natively.
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "base"
    """The name of the backend for the contact sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

        # check that config is valid
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self.num_sensors}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    @abstractmethod
    def num_instances(self) -> int | None:
        """Number of instances of the sensor."""
        raise NotImplementedError(f"Num instances is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def data(self) -> BaseContactSensorData:
        """Data from the sensor."""
        raise NotImplementedError(f"Data is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def num_sensors(self) -> int:
        """Number of sensors per environment."""
        raise NotImplementedError(f"Num sensors is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def body_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies with contact sensors attached."""
        raise NotImplementedError(f"Body names is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def contact_view(self) -> None:
        """View for the contact forces captured.

        .. note::
            None if there is no view associated with the sensor.
        """
        raise NotImplementedError(f"Contact view is not implemented for {self.__class__.__name__}.")

    """
    Operations
    """

    @abstractmethod
    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array(dtype=wp.bool) | None = None):
        """Resets the sensor.

        Args:
            env_ids: The indices of the environments to reset. Defaults to None: all the environments are reset.
            env_mask: The masks of the environments to reset. Defaults to None: all the environments are reset.
        """
        # reset the timers and counters
        super().reset(env_ids, env_mask)

    def find_sensors(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find sensors in the contact sensor based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the sensor indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

    @abstractmethod
    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> wp.array:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact.

        .. note::
            The function assumes that :attr:`dt` is a factor of the sensor update time-step. In other
            words :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true
            if the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contact was established.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have established contact within the last
            :attr:`dt` seconds. Shape is (N, B), where N is the number of sensors and B is the
            number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        raise NotImplementedError(f"Compute first contact is not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> wp.array:
        """Checks if bodies that have broken contact within the last :attr:`dt` seconds.

        This function checks if the bodies have broken contact within the last :attr:`dt` seconds
        by comparing the current air time with the given time period. If the air time is less
        than the given time period, then the bodies are considered to not be in contact.

        .. note::
            It assumes that :attr:`dt` is a factor of the sensor update time-step. In other words,
            :math:`dt / dt_sensor = n`, where :math:`n` is a natural number. This is always true if
            the sensor is updated by the physics or the environment stepping time-step and the sensor
            is read by the environment stepping time-step.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        raise NotImplementedError(f"Compute first air is not implemented for {self.__class__.__name__}.")

    """
    Implementation.
    """

    @abstractmethod
    def _initialize_impl(self):
        super()._initialize_impl()

    @abstractmethod
    def _create_buffers(self):
        """Creates the buffers for the sensor data."""
        raise NotImplementedError(f"Create buffers is not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array | None):
        """Fills the buffers of the sensor data.

        Args:
            env_mask: Mask of the environments to update. None: update all environments.
        """
        raise NotImplementedError(f"Update buffers is not implemented for {self.__class__.__name__}.")

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        # TODO: invalidate NewtonManager if necessary

    @property
    def num_bodies(self) -> int:
        """Deprecated property. Please use `num_sensors` instead."""
        warnings.warn(
            "The `num_bodies` property will be deprecated in a future release. Please use `num_sensors` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.num_sensors

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Deprecated method. Please use `find_sensors` instead."""
        warnings.warn(
            "The `find_bodies` method will be deprecated in a future release. Please use `find_sensors` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.find_sensors(name_keys, preserve_order)
