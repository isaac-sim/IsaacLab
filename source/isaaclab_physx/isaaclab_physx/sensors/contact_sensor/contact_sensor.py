# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Ignore optional memory usage warning globally
# pyright: reportOptionalSubscript=false

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.contact_sensor import BaseContactSensor

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .contact_sensor_data import ContactSensorData
from .kernels import (
    compute_first_transition_kernel,
    reset_contact_sensor_kernel,
    split_flat_pose_to_pos_quat,
    unpack_contact_buffer_data,
    update_net_forces_kernel,
)

if TYPE_CHECKING:
    from isaaclab.sensors.contact_sensor import ContactSensorCfg


class ContactSensor(BaseContactSensor):
    """A PhysX contact reporting sensor.

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
    .. _RigidContact: https://docs.isaacsim.omniverse.nvidia.com/latest/py/source/extensions/isaacsim.core.api/docs/index.html#isaacsim.core.api.sensors.RigidContactView
    """

    cfg: ContactSensorCfg
    """The configuration parameters."""

    __backend_name__: str = "physx"
    """The name of the backend for the contact sensor."""

    def __init__(self, cfg: ContactSensorCfg):
        """Initializes the contact sensor object.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

        # Enable contact processing
        get_settings_manager().set_bool("/physics/disableContactProcessing", False)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # initialize self._body_physx_view for running in extension mode
        self._body_physx_view = None
        # Warp env index array (set in _initialize_impl)
        self._ALL_ENV_INDICES: wp.array | None = None
        self._ALL_ENV_MASK: wp.array | None = None
        self._reset_mask: wp.array | None = None

        # check if max_contact_data_count_per_prim is set
        if self.cfg.max_contact_data_count_per_prim is None:
            self.cfg.max_contact_data_count_per_prim = 4

        # check if force_threshold is set
        if self.cfg.force_threshold is None:
            self.cfg.force_threshold = 1.0

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Contact sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self.body_physx_view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of bodies  : {self.num_bodies}\n"
            f"\tbody names        : {self.body_names}\n"
        )

    """
    Properties
    """

    @property
    def num_instances(self) -> int:
        return self.body_physx_view.count

    @property
    def data(self) -> ContactSensorData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_sensors(self) -> int:
        """Number of bodies with contact sensors attached."""
        return self._num_sensors

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies with contact sensors attached."""
        prim_paths = self.body_physx_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def body_physx_view(self) -> physx.RigidBodyView:
        """View for the rigid bodies captured (PhysX).

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._body_physx_view

    @property
    def contact_view(self) -> physx.RigidContactView:
        """Contact reporter view for the bodies (PhysX).

        .. note::
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._contact_view

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None, env_mask: wp.array | None = None) -> None:
        # resolve env_ids to warp array
        env_mask = self._resolve_indices_and_mask(env_ids, env_mask)
        # reset the timers and counters
        super().reset(None, env_mask)

        wp.launch(
            reset_contact_sensor_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                self._history_length,
                self._num_filter_shapes,
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
                self._data._friction_forces_w,
                self._data._contact_pos_w,
            ],
            device=self._device,
        )

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
            The tensor returned by this function is only valid when called. If compute_first_air is called after
            compute_first_contact, the tensor returned by this method will be have changed values. To avoid this,
            either consume the results of this call immediately or clone the output of this tensor.

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
        wp.launch(
            compute_first_transition_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[float(dt + abs_tol), self._data._current_contact_time],
            outputs=[self._data._first_transition],
            device=self._device,
        )
        return self._data._first_transition

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
            The tensor returned by this function is only valid when called. If compute_first_contact is called after
            compute_first_air, the tensor returned by this method will be have changed values. To avoid this,
            either consume the results of this call immediately or clone the output of this tensor.

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
                "The contact sensor is not configured to track air time."
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
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check that only rigid bodies are selected
        leaf_pattern = self.cfg.prim_path.rsplit("/", 1)[-1]
        template_prim_path = self._parent_prims[0].GetPath().pathString
        body_names = list()
        for prim in sim_utils.find_matching_prims(template_prim_path + "/" + leaf_pattern):
            # check if prim has contact reporter API
            if "PhysxContactReportAPI" in prim.GetAppliedSchemas():
                prim_path = prim.GetPath().pathString
                body_names.append(prim_path.rsplit("/", 1)[-1])
        # check that there is at least one body with contact reporter API
        if not body_names:
            raise RuntimeError(
                f"Sensor at path '{self.cfg.prim_path}' could not find any bodies with contact reporter API."
                "\nHINT: Make sure to enable 'activate_contact_sensors' in the corresponding asset spawn configuration."
            )

        # construct regex expression for the body names
        body_names_regex = r"(" + "|".join(body_names) + r")"
        body_names_regex = f"{self.cfg.prim_path.rsplit('/', 1)[0]}/{body_names_regex}"
        # convert regex expressions to glob expressions for PhysX
        body_names_glob = body_names_regex.replace(".*", "*")
        filter_prim_paths_glob = [expr.replace(".*", "*") for expr in self.cfg.filter_prim_paths_expr]

        # create a rigid prim view for the sensor
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
        self._contact_view = self._physics_sim_view.create_rigid_contact_view(
            body_names_glob,
            filter_patterns=filter_prim_paths_glob,
            max_contact_data_count=self.cfg.max_contact_data_count_per_prim * len(body_names) * self._num_envs,
        )
        # resolve the true count of bodies
        self._num_sensors = self.body_physx_view.count // self._num_envs
        # check that contact reporter succeeded
        if self._num_sensors != len(body_names):
            raise RuntimeError(
                "Failed to initialize contact reporter for specified bodies."
                f"\n\tInput prim path    : {self.cfg.prim_path}"
                f"\n\tResolved prim paths: {body_names_regex}"
            )

        # check if filter paths are valid
        if self.cfg.track_contact_points or self.cfg.track_friction_forces:
            if len(self.cfg.filter_prim_paths_expr) == 0:
                raise ValueError(
                    "The 'filter_prim_paths_expr' is empty. Please specify a valid filter pattern to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )
            if self.cfg.max_contact_data_count_per_prim < 1:
                raise ValueError(
                    f"The 'max_contact_data_count_per_prim' is {self.cfg.max_contact_data_count_per_prim}. "
                    "Please set it to a value greater than 0 to track"
                    f" {'contact points' if self.cfg.track_contact_points else 'friction forces'}."
                )

        self._create_buffers()

    def _create_buffers(self) -> None:
        # Store filter shapes count
        self._num_filter_shapes = self.contact_view.filter_count if len(self.cfg.filter_prim_paths_expr) != 0 else 0
        # Store effective history length (always >= 1 for consistent buffer shapes)
        self._history_length = max(self.cfg.history_length, 1)

        # prepare data buffers
        self._data.create_buffers(
            num_envs=self._num_envs,
            num_sensors=self._num_sensors,
            num_filter_shapes=self._num_filter_shapes,
            history_length=self.cfg.history_length,
            track_pose=self.cfg.track_pose,
            track_air_time=self.cfg.track_air_time,
            track_contact_points=self.cfg.track_contact_points,
            track_friction_forces=self.cfg.track_friction_forces,
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array | None = None):
        """Fills the buffers of the sensor data."""
        # Convert env_mask to warp array
        env_mask = self._resolve_indices_and_mask(None, env_mask)

        # PhysX returns (N*B, 3) float32 -> (N*B,) vec3f
        net_forces_flat = self.contact_view.get_net_contact_forces(dt=self._sim_physics_dt).view(wp.vec3f)
        # PhysX returns (N*B, M, 3) float32 -> (N*B, M) vec3f
        if len(self.cfg.filter_prim_paths_expr) != 0:
            force_matrix_flat = self.contact_view.get_contact_force_matrix(dt=self._sim_physics_dt).view(wp.vec3f)
        else:
            force_matrix_flat = None
        #
        wp.launch(
            update_net_forces_kernel,
            dim=(self._num_envs, self._num_sensors),
            inputs=[
                net_forces_flat,
                force_matrix_flat,
                env_mask,
                self._num_sensors,
                self._num_filter_shapes,
                self._history_length,
                self.cfg.force_threshold,
                self._timestamp,
                self._timestamp_last_update,
            ],
            outputs=[
                self._data._net_forces_w,
                self._data._net_forces_w_history,
                self._data._force_matrix_w,
                self._data._force_matrix_w_history,
                self._data._current_air_time,
                self._data._current_contact_time,
                self._data._last_air_time,
                self._data._last_contact_time,
            ],
            device=self._device,
        )

        # -- Pose --
        if self.cfg.track_pose:
            # PhysX returns (N*B, 7) float32 -> (N*B,) transformf
            poses_flat = self.body_physx_view.get_transforms().view(wp.transformf)
            wp.launch(
                split_flat_pose_to_pos_quat,
                dim=(self._num_envs, self._num_sensors),
                inputs=[poses_flat, env_mask, self._num_sensors],
                outputs=[self._data._pos_w, self._data._quat_w],
                device=self.device,
            )

        # -- Contact points --
        if self.cfg.track_contact_points:
            _, buffer_contact_points, _, _, buffer_count, buffer_start_indices = self.contact_view.get_contact_data(
                dt=self._sim_physics_dt
            )
            # buffer_contact_points: (total_contacts, 3) float32 -> (total_contacts,) vec3f
            pts_vec3 = buffer_contact_points.view(wp.vec3f)
            wp.launch(
                unpack_contact_buffer_data,
                dim=(self._num_envs, self._num_sensors, self._num_filter_shapes),
                inputs=[
                    pts_vec3,
                    buffer_count,
                    buffer_start_indices,
                    env_mask,
                    self._num_sensors,
                    True,
                    float("nan"),
                ],
                outputs=[self._data._contact_pos_w],
                device=self.device,
            )

        # -- Friction forces --
        if self.cfg.track_friction_forces:
            friction_forces, _, buffer_count, buffer_start_indices = self.contact_view.get_friction_data(
                dt=self._sim_physics_dt
            )
            friction_vec3 = friction_forces.view(wp.vec3f)
            wp.launch(
                unpack_contact_buffer_data,
                dim=(self._num_envs, self._num_sensors, self._num_filter_shapes),
                inputs=[
                    friction_vec3,
                    buffer_count,
                    buffer_start_indices,
                    env_mask,
                    self._num_sensors,
                    False,
                    0.0,
                ],
                outputs=[self._data._friction_forces_w],
                device=self.device,
            )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "contact_visualizer"):
                self.contact_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "contact_visualizer"):
                self.contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self.body_physx_view is None:
            return
        # Convert warp data to torch at the boundary for visualization
        net_forces_torch = wp.to_torch(self._data._net_forces_w)  # (N, B, 3)
        net_contact_force_w = torch.linalg.norm(net_forces_torch, dim=-1)
        # marker indices: 0 = contact, 1 = no contact
        marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        if self.cfg.track_pose:
            frame_origins = wp.to_torch(self._data._pos_w)  # (N, B, 3)
        else:
            pose = self.body_physx_view.get_transforms()  # (N*B, 7) float32
            pose_torch = wp.to_torch(pose)
            frame_origins = pose_torch.view(-1, self._num_sensors, 7)[:, :, :3]
        # visualize
        self.contact_visualizer.visualize(frame_origins.reshape(-1, 3), marker_indices=marker_indices.reshape(-1))

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._body_physx_view = None
        self._contact_view = None
