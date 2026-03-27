# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from ..sensor_base import SensorBase
from .imu_data import ImuData

if TYPE_CHECKING:
    from .imu_cfg import ImuCfg


class Imu(SensorBase):
    """The Inertia Measurement Unit (IMU) sensor.

    The sensor can be attached to any prim path with a rigid ancestor in its tree and produces body-frame
    linear acceleration and angular velocity, along with world-frame pose and body-frame linear and angular
    accelerations/velocities.

    If the provided path is not a rigid body, the closest rigid-body ancestor is used for simulation queries.
    The fixed transform from that ancestor to the target prim is computed once during initialization and
    composed with the configured sensor offset.

    .. note::

        We are computing the accelerations using numerical differentiation from the velocities. Consequently, the
        IMU sensor accuracy depends on the chosen phsyx timestep. For a sufficient accuracy, we recommend to keep the
        timestep at least as 200Hz.

    .. note::

        The user can configure the sensor offset in the configuration file. The offset is applied relative to the
        rigid source prim. If the target prim is not a rigid body, the offset is composed with the fixed transform
        from the rigid ancestor to the target prim. The offset is applied in the body frame of the rigid source prim.
        The offset is defined as a position vector and a quaternion rotation, which
        are applied in the order: position, then rotation. The position is applied as a translation
        in the body frame of the rigid source prim, and the rotation is applied as a rotation
        in the body frame of the rigid source prim.

    """

    cfg: ImuCfg
    """The configuration parameters."""

    def __init__(self, cfg: ImuCfg):
        """Initializes the Imu sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = ImuData()

        # Internal: expression used to build the rigid body view (may be different from cfg.prim_path)
        self._rigid_parent_expr: str | None = None

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"Imu sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> ImuData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    @property
    def num_instances(self) -> int:
        return self._view.count

    """
    Operations
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # reset the timestamps
        super().reset(env_ids)
        # resolve None
        if env_ids is None:
            env_ids = slice(None)
        # reset accumulative data buffers
        self._data.pos_w[env_ids] = 0.0
        self._data.quat_w[env_ids] = 0.0
        self._data.quat_w[env_ids, 0] = 1.0
        self._data.projected_gravity_b[env_ids] = 0.0
        self._data.projected_gravity_b[env_ids, 2] = -1.0
        self._data.lin_vel_b[env_ids] = 0.0
        self._data.ang_vel_b[env_ids] = 0.0
        self._data.lin_acc_b[env_ids] = 0.0
        self._data.ang_acc_b[env_ids] = 0.0
        self._prev_lin_vel_w[env_ids] = 0.0
        self._prev_ang_vel_w[env_ids] = 0.0

    def update(self, dt: float, force_recompute: bool = False):
        # save timestamp
        self._dt = dt
        # execute updating
        super().update(dt, force_recompute)

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles and internal buffers.

        - If the target prim path is a rigid body, build the view directly on it.
        - Otherwise find the closest rigid-body ancestor, cache the fixed transform from that ancestor
          to the target prim, and build the view on the ancestor expression.
        """
        # Initialize parent class
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is a rigid prim
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        # Find the first matching ancestor prim that implements rigid body API
        ancestor_prim = sim_utils.get_first_matching_ancestor_prim(
            prim.GetPath(), predicate=lambda _prim: _prim.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        if ancestor_prim is None:
            raise RuntimeError(f"Failed to find a rigid body ancestor prim at path expression: {self.cfg.prim_path}")
        # Convert ancestor prim path to expression
        if ancestor_prim == prim:
            self._rigid_parent_expr = self.cfg.prim_path
            fixed_pos_b, fixed_quat_b = None, None
        else:
            # Convert ancestor prim path to expression
            relative_path = prim.GetPath().MakeRelativePath(ancestor_prim.GetPath()).pathString
            self._rigid_parent_expr = self.cfg.prim_path.replace(relative_path, "")
            # Resolve the relative pose between the target prim and the ancestor prim
            fixed_pos_b, fixed_quat_b = sim_utils.resolve_prim_pose(prim, ancestor_prim)

        # Create the rigid body view on the ancestor
        self._view = self._physics_sim_view.create_rigid_body_view(self._rigid_parent_expr.replace(".*", "*"))

        # Get world gravity
        gravity = self._physics_sim_view.get_gravity()
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        self.GRAVITY_VEC_W = gravity_dir.repeat(self.num_instances, 1)

        # Create internal buffers
        self._initialize_buffers_impl()

        # Compose the configured offset with the fixed ancestor->target transform (done once)
        # new_offset = fixed * cfg.offset
        # where composition is: p = p_fixed + R_fixed * p_cfg, q = q_fixed * q_cfg
        if fixed_pos_b is not None and fixed_quat_b is not None:
            # Broadcast fixed transform across instances
            fixed_p = torch.tensor(fixed_pos_b, device=self._device).repeat(self._view.count, 1)
            fixed_q = torch.tensor(fixed_quat_b, device=self._device).repeat(self._view.count, 1)

            cfg_p = self._offset_pos_b.clone()
            cfg_q = self._offset_quat_b.clone()

            composed_p = fixed_p + math_utils.quat_apply(fixed_q, cfg_p)
            composed_q = math_utils.quat_mul(fixed_q, cfg_q)

            self._offset_pos_b = composed_p
            self._offset_quat_b = composed_q

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        # world pose of the rigid source (ancestor) from the PhysX view
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = quat_w.roll(1, dims=-1)

        # sensor pose in world: apply composed offset
        self._data.pos_w[env_ids] = pos_w + math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids])
        self._data.quat_w[env_ids] = math_utils.quat_mul(quat_w, self._offset_quat_b[env_ids])

        # COM of rigid source (body frame)
        com_pos_b = self._view.get_coms().to(self.device).split([3, 4], dim=-1)[0]

        # Velocities at rigid source COM
        lin_vel_w, ang_vel_w = self._view.get_velocities()[env_ids].split([3, 3], dim=-1)

        # If sensor offset or COM != link origin, account for angular velocity contribution
        lin_vel_w += torch.linalg.cross(
            ang_vel_w, math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids] - com_pos_b[env_ids]), dim=-1
        )

        # numerical derivative (world frame)
        lin_acc_w = (lin_vel_w - self._prev_lin_vel_w[env_ids]) / self._dt + self._gravity_bias_w[env_ids]
        ang_acc_w = (ang_vel_w - self._prev_ang_vel_w[env_ids]) / self._dt

        # batch rotate world->body using current sensor orientation
        dynamics_data = torch.stack((lin_vel_w, ang_vel_w, lin_acc_w, ang_acc_w, self.GRAVITY_VEC_W[env_ids]), dim=0)
        dynamics_data_rot = math_utils.quat_apply_inverse(self._data.quat_w[env_ids].repeat(5, 1), dynamics_data).chunk(
            5, dim=0
        )
        # store the velocities.
        self._data.lin_vel_b[env_ids] = dynamics_data_rot[0]
        self._data.ang_vel_b[env_ids] = dynamics_data_rot[1]
        # store the accelerations
        self._data.lin_acc_b[env_ids] = dynamics_data_rot[2]
        self._data.ang_acc_b[env_ids] = dynamics_data_rot[3]
        # store projected gravity
        self._data.projected_gravity_b[env_ids] = dynamics_data_rot[4]

        self._prev_lin_vel_w[env_ids] = lin_vel_w
        self._prev_ang_vel_w[env_ids] = ang_vel_w

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.quat_w[:, 0] = 1.0
        self._data.projected_gravity_b = torch.zeros(self._view.count, 3, device=self._device)
        self._data.lin_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_vel_b = torch.zeros_like(self._data.pos_w)
        self._data.lin_acc_b = torch.zeros_like(self._data.pos_w)
        self._data.ang_acc_b = torch.zeros_like(self._data.pos_w)
        self._prev_lin_vel_w = torch.zeros_like(self._data.pos_w)
        self._prev_ang_vel_w = torch.zeros_like(self._data.pos_w)

        # store sensor offset (applied relative to rigid source).
        # This may be composed later with a fixed ancestor->target transform.
        self._offset_pos_b = torch.tensor(list(self.cfg.offset.pos), device=self._device).repeat(self._view.count, 1)
        self._offset_quat_b = torch.tensor(list(self.cfg.offset.rot), device=self._device).repeat(self._view.count, 1)
        # set gravity bias
        self._gravity_bias_w = torch.tensor(list(self.cfg.gravity_bias), device=self._device).repeat(
            self._view.count, 1
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.acceleration_visualizer.set_visibility(True)
        else:
            if hasattr(self, "acceleration_visualizer"):
                self.acceleration_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # safely return if view becomes invalid
        # note: this invalidity happens because of isaac sim view callbacks
        if self._view is None:
            return
        # get marker location
        # -- base state
        base_pos_w = self._data.pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales
        default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_b.shape[0], 1)
        # get up axis of current stage
        up_axis = UsdGeom.GetStageUpAxis(self.stage)
        # arrow-direction
        quat_opengl = math_utils.quat_from_matrix(
            math_utils.create_rotation_matrix_from_view(
                self._data.pos_w,
                self._data.pos_w + math_utils.quat_apply(self._data.quat_w, self._data.lin_acc_b),
                up_axis=up_axis,
                device=self._device,
            )
        )
        quat_w = math_utils.convert_camera_frame_orientation_convention(quat_opengl, "opengl", "world")
        # display markers
        self.acceleration_visualizer.visualize(base_pos_w, quat_w, arrow_scale)
