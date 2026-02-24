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

    The reporting of the filtered contact forces is only possible as one-to-many. This means that only one
    sensor body in an environment can be filtered against multiple bodies in that environment. If you need to
    filter multiple sensor bodies against multiple bodies, you need to create separate sensors for each sensor
    body.

    As an example, suppose you want to report the contact forces for all the feet of a robot against an object
    exclusively. In that case, setting the :attr:`ContactSensorCfg.prim_path` and
    :attr:`ContactSensorCfg.filter_prim_paths_expr` with ``{ENV_REGEX_NS}/Robot/.*_FOOT`` and ``{ENV_REGEX_NS}/Object``
    respectively will not work. Instead, you need to create a separate sensor for each foot and filter
    it against the object.
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

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> BaseContactSensorData:
        raise NotImplementedError

    @property
    @abstractmethod
    def num_sensors(self) -> int:
        """Number of sensors per environment."""
        raise NotImplementedError

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        raise NotImplementedError

    @property
    @abstractmethod
    def contact_view(self):
        """View for the contact reporting.

        .. note::
            None if there is no view associated with the sensor.
        """
        raise NotImplementedError

    """
    Operations
    """

    def find_sensors(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

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

        .. caution::
            This method and :meth:`compute_first_air` share an internal output buffer. Calling one
            after the other will overwrite the previous result. To avoid data loss, consume or clone
            the returned array before calling the other method.

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
        raise NotImplementedError

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

        .. caution::
            This method and :meth:`compute_first_contact` share an internal output buffer. Calling one
            after the other will overwrite the previous result. To avoid data loss, consume or clone
            the returned array before calling the other method.

        Args:
            dt: The time period since the contract is broken.
            abs_tol: The absolute tolerance for the comparison.

        Returns:
            A boolean tensor indicating the bodies that have broken contact within the last :attr:`dt` seconds.
            Shape is (N, B), where N is the number of sensors and B is the number of bodies in each sensor.

        Raises:
            RuntimeError: If the sensor is not configured to track contact time.
        """
        raise NotImplementedError

    """
    Implementation - Abstract methods to be implemented by backend-specific subclasses.
    """

    @abstractmethod
    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        Subclasses should call ``super()._initialize_impl()`` first to initialize
        the common sensor infrastructure from :class:`SensorBase`.
        """
        super()._initialize_impl()

    @abstractmethod
    def _update_buffers_impl(self, env_mask: wp.array):
        raise NotImplementedError

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        super()._invalidate_initialize_callback(event)

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
