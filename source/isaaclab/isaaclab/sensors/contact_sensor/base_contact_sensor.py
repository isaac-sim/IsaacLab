# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.markers import VisualizationMarkers

from ..sensor_base import SensorBase
from .base_contact_sensor_data import BaseContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class BaseContactSensor(SensorBase):
    """A contact reporting sensor.

    The contact sensor reports the normal contact forces on a rigid body in the world frame.
    It relies on the `PhysX ContactReporter`_ API to be activated on the rigid bodies.

    To enable the contact reporter on a rigid body, please make sure to enable the
    :attr:`isaaclab.sim.spawner.RigidObjectSpawnerCfg.activate_contact_sensors` on your
    asset spawner configuration. This will enable the contact reporter on all the rigid bodies
    in the asset.

    The sensor can be configured to report the contact forces on a set of bodies with a given
    filter pattern using the :attr:`ContactSensorCfg.filter_prim_paths_expr`. This is useful
    when you want to report the contact forces between the sensor bodies and a specific set of
    bodies in the scene. The data can be accessed using the :attr:`ContactSensorData.force_matrix_w`.
    Please check the documentation on `RigidContact`_ for more details.

    The reporting of the filtered contact forces is only possible as one-to-many. This means that only one
    sensor body in an environment can be filtered against multiple bodies in that environment. If you need to
    filter multiple sensor bodies against multiple bodies, you need to create separate sensors for each sensor
    body.

    As an example, suppose you want to report the contact forces for all the feet of a robot against an object
    exclusively. In that case, setting the :attr:`ContactSensorCfg.prim_path` and
    :attr:`ContactSensorCfg.filter_prim_paths_expr` with ``{ENV_REGEX_NS}/Robot/.*_FOOT`` and ``{ENV_REGEX_NS}/Object``
    respectively will not work. Instead, you need to create a separate sensor for each foot and filter
    it against the object.

    .. _PhysX ContactReporter: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_contact_report_a_p_i.html
    .. _RigidContact: https://docs.omniverse.nvidia.com/py/isaacsim/source/isaacsim.core/docs/index.html#isaacsim.core.prims.RigidContact
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

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self.num_bodies}\n"
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
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors attached."""
        raise NotImplementedError(f"Num bodies is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def body_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies with contact sensors attached."""
        raise NotImplementedError(f"Body names is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def contact_partner_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies that are selected as contact partners."""
        raise NotImplementedError(f"Contact partner names is not implemented for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def contact_view(self) -> None:
        """View for the contact forces captured.

        Note:
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
        super().reset(env_ids)

    @abstractmethod
    def find_bodies(
        self, name_keys: str | Sequence[str], preserve_order: bool = False
    ) -> tuple[wp.array, list[str], list[int]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple containing the body mask (wp.array), names (list[str]), and indices (list[int]).
        """
        raise NotImplementedError(f"Find bodies is not implemented for {self.__class__.__name__}.")

    @abstractmethod
    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> wp.array:
        """Checks if bodies that have established contact within the last :attr:`dt` seconds.

        This function checks if the bodies have established contact within the last :attr:`dt` seconds
        by comparing the current contact time with the given time period. If the contact time is less
        than the given time period, then the bodies are considered to be in contact.

        Note:
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

        Note:
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
    def _update_buffers_impl(self, env_ids: Sequence[int], masks: wp.array(dtype=wp.bool) | None = None):
        """Fills the buffers of the sensor data.

        Args:
            env_ids: The indices of the environments to update. Defaults to None: all the environments are updated.
            masks: The masks of the environments to update. Defaults to None: all the environments are updated.
        """
        raise NotImplementedError(f"Update buffers is not implemented for {self.__class__.__name__}.")

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

    @abstractmethod
    def _debug_vis_callback(self, event):
        """Callback for debug visualization."""
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        # TODO: invalidate NewtonManager if necessary
