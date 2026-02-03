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

import carb
import omni.physics.tensors.impl.api as physx
from isaacsim.core.simulation_manager import SimulationManager
from pxr import PhysxSchema

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.contact_sensor import BaseContactSensor

from .contact_sensor_data import ContactSensorData

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
        carb_settings_iface = carb.settings.get_settings()
        carb_settings_iface.set_bool("/physics/disableContactProcessing", False)

        # Create empty variables for storing output data
        self._data: ContactSensorData = ContactSensorData()
        # initialize self._body_physx_view for running in extension mode
        self._body_physx_view = None

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
    def num_bodies(self) -> int:
        """Number of bodies with contact sensors attached."""
        return self._num_bodies

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

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timers and counters
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.net_forces_w[env_ids] = 0.0
        self._data.net_forces_w_history[env_ids] = 0.0
        # reset force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            self._data.force_matrix_w[env_ids] = 0.0
            self._data.force_matrix_w_history[env_ids] = 0.0
        # reset the current air time
        if self.cfg.track_air_time:
            self._data.current_air_time[env_ids] = 0.0
            self._data.last_air_time[env_ids] = 0.0
            self._data.current_contact_time[env_ids] = 0.0
            self._data.last_contact_time[env_ids] = 0.0
        # reset contact positions
        if self.cfg.track_contact_points:
            self._data.contact_pos_w[env_ids, :] = torch.nan
        # reset friction forces
        if self.cfg.track_friction_forces:
            self._data.friction_forces_w[env_ids, :] = 0.0

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
            if prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
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
        self._num_bodies = self.body_physx_view.count // self._num_envs
        # check that contact reporter succeeded
        if self._num_bodies != len(body_names):
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

        # prepare data buffers
        num_filter_shapes = self.contact_view.filter_count if len(self.cfg.filter_prim_paths_expr) != 0 else 0
        self._data.create_buffers(
            num_envs=self._num_envs,
            num_bodies=self._num_bodies,
            num_filter_shapes=num_filter_shapes,
            history_length=self.cfg.history_length,
            track_pose=self.cfg.track_pose,
            track_air_time=self.cfg.track_air_time,
            track_contact_points=self.cfg.track_contact_points,
            track_friction_forces=self.cfg.track_friction_forces,
            device=self._device,
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        # obtain the contact forces
        # TODO: We are handling the indexing ourself because of the shape; (N, B) vs expected (N * B).
        #   This isn't the most efficient way to do this, but it's the easiest to implement.
        net_forces_w = self.contact_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(-1, self._num_bodies, 3)[env_ids]
        # update contact force history
        if self.cfg.history_length > 0:
            self._data.net_forces_w_history[env_ids] = self._data.net_forces_w_history[env_ids].roll(1, dims=1)
            self._data.net_forces_w_history[env_ids, 0] = self._data.net_forces_w[env_ids]

        # obtain the contact force matrix
        if len(self.cfg.filter_prim_paths_expr) != 0:
            # shape of the filtering matrix: (num_envs, num_bodies, num_filter_shapes, 3)
            num_filters = self.contact_view.filter_count
            # acquire and shape the force matrix
            force_matrix_w = self.contact_view.get_contact_force_matrix(dt=self._sim_physics_dt)
            force_matrix_w = force_matrix_w.view(-1, self._num_bodies, num_filters, 3)
            self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
            if self.cfg.history_length > 0:
                self._data.force_matrix_w_history[env_ids] = self._data.force_matrix_w_history[env_ids].roll(1, dims=1)
                self._data.force_matrix_w_history[env_ids, 0] = self._data.force_matrix_w[env_ids]

        # obtain the pose of the sensor origin
        if self.cfg.track_pose:
            pose = self.body_physx_view.get_transforms().view(-1, self._num_bodies, 7)[env_ids]
            self._data.pos_w[env_ids], self._data.quat_w[env_ids] = pose.split([3, 4], dim=-1)

        # obtain contact points
        if self.cfg.track_contact_points:
            _, buffer_contact_points, _, _, buffer_count, buffer_start_indices = self.contact_view.get_contact_data(
                dt=self._sim_physics_dt
            )
            self._data.contact_pos_w[env_ids] = self._unpack_contact_buffer_data(
                buffer_contact_points, buffer_count, buffer_start_indices
            )[env_ids]

        # obtain friction forces
        if self.cfg.track_friction_forces:
            friction_forces, _, buffer_count, buffer_start_indices = self.contact_view.get_friction_data(
                dt=self._sim_physics_dt
            )
            self._data.friction_forces_w[env_ids] = self._unpack_contact_buffer_data(
                friction_forces, buffer_count, buffer_start_indices, avg=False, default=0.0
            )[env_ids]

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

    def _unpack_contact_buffer_data(
        self,
        contact_data: torch.Tensor,
        buffer_count: torch.Tensor,
        buffer_start_indices: torch.Tensor,
        avg: bool = True,
        default: float = float("nan"),
    ) -> torch.Tensor:
        """
        Unpacks and aggregates contact data for each (env, body, filter) group.

        This function vectorizes the following nested loop:

        for i in range(self._num_bodies * self._num_envs):
            for j in range(self.contact_view.filter_count):
                start_index_ij = buffer_start_indices[i, j]
                count_ij = buffer_count[i, j]
                self._contact_position_aggregate_buffer[i, j, :] = torch.mean(
                    contact_data[start_index_ij : (start_index_ij + count_ij), :], dim=0
                )

        For more details, see the `RigidContactView.get_contact_data() documentation <https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.RigidContactView.get_contact_data>`_.

        Args:
            contact_data: Flat tensor of contact data, shape (N_envs * N_bodies, 3).
            buffer_count: Number of contact points per (env, body, filter), shape (N_envs * N_bodies, N_filters).
            buffer_start_indices: Start indices for each (env, body, filter), shape (N_envs * N_bodies, N_filters).
            avg: If True, average the contact data for each group; if False, sum the data. Defaults to True.
            default: Default value to use for groups with zero contacts. Defaults to NaN.

        Returns:
            Aggregated contact data, shape (N_envs, N_bodies, N_filters, 3).
        """
        counts, starts = buffer_count.view(-1), buffer_start_indices.view(-1)
        n_rows, total = counts.numel(), int(counts.sum())
        agg = torch.full((n_rows, 3), default, device=self._device, dtype=contact_data.dtype)
        if total > 0:
            row_ids = torch.repeat_interleave(torch.arange(n_rows, device=self._device), counts)

            block_starts = counts.cumsum(0) - counts
            deltas = torch.arange(row_ids.numel(), device=counts.device) - block_starts.repeat_interleave(counts)
            flat_idx = starts[row_ids] + deltas

            pts = contact_data.index_select(0, flat_idx)
            agg = agg.zero_().index_add_(0, row_ids, pts)
            agg = agg / counts.clamp_min(1).unsqueeze(-1) if avg else agg
            agg[counts == 0] = default

        return agg.view(self._num_envs * self.num_bodies, -1, 3).view(
            self._num_envs, self._num_bodies, self.contact_view.filter_count, 3
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
        # marker indices
        # 0: contact, 1: no contact
        net_contact_force_w = torch.norm(self._data.net_forces_w, dim=-1)
        marker_indices = torch.where(net_contact_force_w > self.cfg.force_threshold, 0, 1)
        # check if prim is visualized
        if self.cfg.track_pose:
            frame_origins: torch.Tensor = self._data.pos_w
        else:
            pose = self.body_physx_view.get_transforms()
            frame_origins = pose.view(-1, self._num_bodies, 7)[:, :, :3]
        # visualize
        self.contact_visualizer.visualize(frame_origins.view(-1, 3), marker_indices=marker_indices.view(-1))

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
