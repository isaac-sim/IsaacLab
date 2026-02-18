# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import logging
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp
from newton.sensors import MatchKind
from newton.sensors import SensorContact as NewtonContactSensor

import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.contact_sensor.base_contact_sensor import BaseContactSensor
from isaaclab.sim._impl.newton_manager import NewtonManager

from .contact_sensor_data import ContactSensorData

if TYPE_CHECKING:
    from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

logger = logging.getLogger(__name__)


class ContactSensor(BaseContactSensor):
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

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()

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
    def num_instances(self) -> int | None:
        return self._num_bodies

    @property
    def data(self) -> ContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors attached."""
        return self._num_bodies

    @property
    def body_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies with contact sensors attached."""
        return self._body_names

    @property
    def contact_partner_names(self) -> list[str] | None:
        """Ordered names of shapes or bodies that are selected as contact partners."""
        return self._contact_partner_names

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
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids] = 0.0
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = 0.0
        # reset force matrix
        if self.cfg.filter_prim_paths_expr is not None or self.cfg.filter_shape_paths_expr is not None:
            self._data.force_matrix_w[env_ids] = 0.0
        # reset the current air time
        if self.cfg.track_air_time:
            self._data.current_air_time[env_ids] = 0.0
            self._data.last_air_time[env_ids] = 0.0
            self._data.current_contact_time[env_ids] = 0.0
            self._data.last_contact_time[env_ids] = 0.0

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
        indices, names = string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)
        mask = wp.array([name in names for name in self.body_names], dtype=wp.bool, device=self._device)
        return mask, names, indices

    def compute_first_contact(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
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
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the bodies are in contact
        currently_in_contact = self.data.current_contact_time > 0.0
        less_than_dt_in_contact = self.data.current_contact_time < (dt + abs_tol)
        return currently_in_contact * less_than_dt_in_contact

    def compute_first_air(self, dt: float, abs_tol: float = 1.0e-8) -> torch.Tensor:
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
        # check if the sensor is configured to track contact time
        if not self.cfg.track_air_time:
            raise RuntimeError(
                "The contact sensor is not configured to track contact time."
                "Please enable the 'track_air_time' in the sensor configuration."
            )
        # check if the sensor is configured to track contact time
        currently_detached = self.data.current_air_time > 0.0
        less_than_dt_detached = self.data.current_air_time < (dt + abs_tol)
        return currently_detached * less_than_dt_detached

    """
    Implementation.
    """

    def _initialize_impl(self):
        super()._initialize_impl()
        """Initializes the sensor-related handles and internal buffers."""
        # construct regex expression for the body names

        if self.cfg.filter_prim_paths_expr is not None or self.cfg.filter_shape_paths_expr is not None:
            self._generate_force_matrix = True
        else:
            self._generate_force_matrix = False

        body_names_regex = self.cfg.prim_path
        if self.cfg.shape_path is not None:
            shape_names_regex = "(" + "|".join(self.cfg.shape_path) + ")"
        else:
            shape_names_regex = None
        if self.cfg.filter_prim_paths_expr is not None:
            contact_partners_body_regex = "(" + "|".join(self.cfg.filter_prim_paths_expr) + ")"
        else:
            contact_partners_body_regex = None
        if self.cfg.filter_shape_paths_expr is not None:
            contact_partners_shape_regex = "(" + "|".join(self.cfg.filter_shape_paths_expr) + ")"
        else:
            contact_partners_shape_regex = None

        # Store the sensor key for later lookup
        self._sensor_key = (
            body_names_regex,
            shape_names_regex,
            contact_partners_body_regex,
            contact_partners_shape_regex,
        )

        NewtonManager.add_contact_sensor(
            body_names_expr=body_names_regex,
            shape_names_expr=shape_names_regex,
            contact_partners_body_expr=contact_partners_body_regex,
            contact_partners_shape_expr=contact_partners_shape_regex,
        )
        self._create_buffers()

    def _create_buffers(self):
        # resolve the true count of bodies
        self._num_bodies = self.contact_view.shape[0] // self._num_envs

        # Check that number of bodies is an integer
        if self.contact_view.shape[0] % self._num_envs != 0:
            raise RuntimeError(
                "Number of bodies is not an integer multiple of the number of environments. Received:"
                f" {self._num_bodies} bodies and {self._num_envs} environments."
            )
        logger.info(f"Contact sensor initialized with {self._num_bodies} bodies.")

        # Assume homogeneous envs, i.e. all envs have the same number of bodies / shapes
        # Only get the names for the first env. Expected structure: /World/envs/env_.*/...
        def get_name(idx, match_kind):
            if match_kind == MatchKind.BODY:
                return NewtonManager._model.body_key[idx].split("/")[-1]
            if match_kind == MatchKind.SHAPE:
                return NewtonManager._model.shape_key[idx].split("/")[-1]
            return "MATCH_ANY"

        self._body_names = [get_name(idx, kind) for idx, kind in self.contact_view.sensing_objs]
        # Assumes the environments are processed in order.
        self._body_names = self._body_names[: self._num_bodies]
        self._contact_partner_names = [get_name(idx, kind) for idx, kind in self.contact_view.counterparts]

        # Number of filtered bodies
        num_filters = max(self.contact_view.shape[1] - 1, 0)

        # prepare data buffers
        logger.info(
            f"Creating buffers for contact sensor data with num_envs: {self._num_envs}, num_bodies: {self._num_bodies},"
            f" num_filters: {num_filters}, history_length: {self.cfg.history_length}, generate_force_matrix:"
            f" {self._generate_force_matrix}, track_air_time: {self.cfg.track_air_time}, track_pose:"
            f" {self.cfg.track_pose}, device: {self._device}"
        )
        self._data.create_buffers(
            self._num_envs,
            self._num_bodies,
            num_filters,
            self.cfg.history_length,
            self._generate_force_matrix,
            self.cfg.track_air_time,
            self.cfg.track_pose,
            self._device,
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # net_force is a matrix of shape (num_bodies * num_envs, num_filters, 3)
        net_forces_w = wp.to_torch(self.contact_view.net_force).clone()
        self._data.net_forces_w[env_ids, :, :] = net_forces_w[:, 0, :].reshape(self._num_envs, self._num_bodies, 3)[
            env_ids
        ]

        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids, 1:] = self._data.net_forces_w_history[env_ids, :-1].clone()
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if self._generate_force_matrix:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self.contact_view.shape[1] - 1  # -1 for the total force
            # acquire and shape the force matrix
            self._data.force_matrix_w[env_ids] = net_forces_w[:, 1:, :].reshape(
                self._num_envs, self._num_bodies, num_filters, 3
            )[env_ids]

        # FIXME: Re-enable this when we have a non-physx rigid body view?
        # obtain the pose of the sensor origin
        # if self.cfg.track_pose:
        #    pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
        #    pose[..., 3:] = convert_quat(pose[..., 3:], to="wxyz")
        #    self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

        # obtain the air time
        if self.cfg.track_air_time:
            # -- time elapsed since last update
            # since this function is called every frame, we can use the difference to get the elapsed time
            elapsed_time = self._timestamp[env_ids] - self._timestamp_last_update[env_ids]
            # -- check contact state of bodies
            is_contact = torch.norm(self._data.net_forces_w[env_ids, :, :], dim=-1) > self.cfg.force_threshold
            is_first_contact = (self._data.current_air_time[env_ids] > 0) * is_contact
            is_first_detached = (self._data.current_contact_time[env_ids] > 0) * ~is_contact
            # -- update the last contact time if body has just become in contact
            self._data.last_air_time[env_ids] = torch.where(
                is_first_contact,
                self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_air_time[env_ids],
            )
            # -- increment time for bodies that are not in contact
            self._data.current_air_time[env_ids] = torch.where(
                ~is_contact, self._data.current_air_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )
            # -- update the last contact time if body has just detached
            self._data.last_contact_time[env_ids] = torch.where(
                is_first_detached,
                self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1),
                self._data.last_contact_time[env_ids],
            )
            # -- increment time for bodies that are in contact
            self._data.current_contact_time[env_ids] = torch.where(
                is_contact, self._data.current_contact_time[env_ids] + elapsed_time.unsqueeze(-1), 0.0
            )

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
        #    frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
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
