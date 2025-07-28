# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
from isaacsim.core.simulation_manager import SimulationManager
from isaaclab.sim._impl.newton_manager import NewtonManager
from newton.utils.contact_sensor import ContactView
from pxr import PhysxSchema
import warp as wp
import omni.kit.app
import weakref
import omni.timeline

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import convert_quat

from ..sensor_base import SensorBase
from .contact_sensor_data import ContactSensorData

if TYPE_CHECKING:
    from .contact_sensor_cfg import ContactSensorCfg


class ContactSensor(SensorBase):
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
    _contact_newton_view: ContactView
    """The contact view for the sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        # check that config is valid
        if cfg.history_length < 0:
            raise ValueError(f"History length must be greater than 0! Received: {cfg.history_length}")
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg.copy()
        # flag for whether the sensor is initialized
        self._is_initialized = False
        # flag for whether the sensor is in visualization mode
        self._is_visualizing = False

        # note: Use weakref on callbacks to ensure that this object can be deleted when its destructor is called.
        # add callbacks for stage play/stop
        # The order is set to 10 which is arbitrary but should be lower priority than the default order of 0

        NewtonManager.add_on_init_callback(self._initialize_impl)

        timeline_event_stream = omni.timeline.get_timeline_interface().get_timeline_event_stream()
        self._invalidate_initialize_handle = timeline_event_stream.create_subscription_to_pop_by_type(
            int(omni.timeline.TimelineEventType.STOP),
            lambda event, obj=weakref.proxy(self): obj._invalidate_initialize_callback(event),
            order=10,
        )
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self.contact_newton_view.__class__}\n"
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
    def contact_newton_view(self) -> ContactView:
        """View for the contact forces captured (Newton)."""
        return self._contact_newton_view

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids] = 0.0
        self._data.net_forces_w_history[env_ids] = 0.0
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

    def find_bodies(self, name_keys: str | Sequence[str], preserve_order: bool = False) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.
            preserve_order: Whether to preserve the order of the name keys in the output. Defaults to False.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names, preserve_order)

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
        body_names_regex = self.cfg.prim_path
        if self.cfg.shape_path is not None:
            shape_names_regex = r"(" + "|".join(self.cfg.shape_path) + r")"
        else:
            shape_names_regex = None
        if self.cfg.filter_prim_paths_expr is not None:
            contact_partners_body_regex = r"(" + "|".join(self.cfg.filter_prim_paths_expr) + r")"
        else:
            contact_partners_body_regex = None
        if self.cfg.filter_shape_paths_expr is not None:
            contact_partners_shape_regex = r"(" + "|".join(self.cfg.filter_shape_paths_expr) + r")"
        else:
            contact_partners_shape_regex = None

        if self.cfg.filter_prim_paths_expr is not None or self.cfg.filter_shape_paths_expr is not None:
            self._generate_force_matrix = True
        else:
            self._generate_force_matrix = False
        
        self._contact_newton_view = NewtonManager.add_contact_view(
            body_names_expr = body_names_regex,
            shape_names_expr = shape_names_regex,
            contact_partners_body_expr = contact_partners_body_regex,
            contact_partners_shape_expr = contact_partners_shape_regex,
        )
        NewtonManager.add_on_start_callback(self._create_buffers)

    def _create_buffers(self):
        # resolve the true count of bodies
        self._num_bodies = self._contact_newton_view.shape[0] // self._num_envs

        # Check that number of bodies is an integer
        if self._contact_newton_view.shape[0] % self._num_envs != 0:
            raise RuntimeError(f"Number of bodies is not an integer multiple of the number of environments. Received: {self._num_bodies} bodies and {self._num_envs} environments.")
        print(f"[INFO] Contact sensor initialized with {self._num_bodies} bodies.")
        
        self._body_names = [entity[1].split("/")[-1] for entity in self._contact_newton_view.sensor_keys[:self._num_bodies]]
        self._contact_partner_names = [entity[1].split("/")[-1] for entity in self._contact_newton_view.contact_partner_keys[1:]]

        # prepare data buffers
        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        # optional buffers
        # -- history of net forces
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history = torch.zeros(
                self._num_envs, self.cfg.history_length, self._num_bodies, 3, device=self._device
            )
        else:
            self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
        # -- pose of sensor origins
        if self.cfg.track_pose:
            self._data.pos_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
            self._data.quat_w = torch.zeros(self._num_envs, self._num_bodies, 4, device=self._device)
        # -- air/contact time between contacts
        if self.cfg.track_air_time:
            self._data.last_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_air_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.last_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
            self._data.current_contact_time = torch.zeros(self._num_envs, self._num_bodies, device=self._device)
        # force matrix: (num_envs, num_bodies, num_filter_shapes, 3)
        if self._generate_force_matrix:
            num_filters = self._contact_newton_view.shape[1]
            self._data.force_matrix_w = torch.zeros(
                self._num_envs, self._num_bodies, num_filters, 3, device=self._device
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        #net_force is a matrix of shape (num_bodies * num_envs, num_filters, 3)
        net_forces_w = wp.to_torch(self._contact_newton_view.net_force).clone()
        self._data.net_forces_w[env_ids, :, :] = net_forces_w[:, 0, :].reshape(self._num_envs, self._num_bodies, 3)[env_ids]
        
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids, 1:] = self._data.net_forces_w_history[env_ids, :-1].clone()
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if self._generate_force_matrix:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self._contact_newton_view.shape[1] - 1 # -1 for the total force
            # acquire and shape the force matrix
            self._data.force_matrix_w[env_ids] = net_forces_w[:, 1:, :].reshape(self._num_envs, self._num_bodies, num_filters, 3)[env_ids]

        # FIXME: Re-enable this when we have a non-physx rigid body view?
        # obtain the pose of the sensor origin
        #if self.cfg.track_pose:
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
        #if self.body_physx_view is None:
        #    return
        # marker indices
        # 0: contact, 1: no contact
        #net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        #marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        #if self.cfg.track_pose:
        #    frame_origins: torch.Tensor = self._data.pos_w
        #else:
        #    pose = self.body_physx_view.get_transforms()
        #    frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
        # visualize
        #self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._contact_newton_view = None
