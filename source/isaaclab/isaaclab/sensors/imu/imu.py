# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers

from ..sensor_base import SensorBase
from .imu_data import ImuData

if TYPE_CHECKING:
    from .imu_cfg import ImuCfg


class Imu(SensorBase):
    """The Inertia Measurement Unit (IMU) sensor.

    The sensor can be attached to any :class:`RigidObject` or :class:`Articulation` in the scene. The sensor provides complete state information.
    The sensor is primarily used to provide the linear acceleration and angular velocity of the object in the body frame. The sensor also provides
    the position and orientation of the object in the world frame and the angular acceleration and linear velocity in the body frame. The extra
    data outputs are useful for simulating with or comparing against "perfect" state estimation.

    .. note::

        We are computing the accelerations using numerical differentiation from the velocities. Consequently, the
        IMU sensor accuracy depends on the chosen phsyx timestep. For a sufficient accuracy, we recommend to keep the
        timestep at least as 200Hz.

    .. note::

        It is suggested to use the OffsetCfg to define an IMU frame relative to a rigid body prim defined at the root of
        a :class:`RigidObject` or  a prim that is defined by a non-fixed joint in an :class:`Articulation` (except for the
        root of a fixed based articulation). The use frames with fixed joints and small mass/inertia to emulate a transform
        relative to a body frame can result in lower performance and accuracy.

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

        This function creates handles and registers the provided data types with the replicator registry to
        be able to access the data from the sensor. It also initializes the internal buffers to store the data.

        Raises:
            RuntimeError: If the imu prim is not a RigidBodyPrim
        """
        # Initialize parent class
        super()._initialize_impl()
        # obtain global simulation view
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        # check if the prim at path is a rigid prim
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # check if it is a RigidBody Prim
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
        else:
            raise RuntimeError(f"Failed to find a RigidBodyAPI for the prim paths: {self.cfg.prim_path}")

        # Get world gravity
        gravity = self._physics_sim_view.get_gravity()
        gravity_dir = torch.tensor((gravity[0], gravity[1], gravity[2]), device=self.device)
        gravity_dir = math_utils.normalize(gravity_dir.unsqueeze(0)).squeeze(0)
        self.GRAVITY_VEC_W = gravity_dir.repeat(self.num_instances, 1)

        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""

        # default to all sensors
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)
        # obtain the poses of the sensors
        pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
        quat_w = quat_w.roll(1, dims=-1)

        # store the poses
        self._data.pos_w[env_ids] = pos_w + math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids])
        self._data.quat_w[env_ids] = math_utils.quat_mul(quat_w, self._offset_quat_b[env_ids])

        # get the offset from COM to link origin
        com_pos_b = self._view.get_coms().to(self.device).split([3, 4], dim=-1)[0]

        # obtain the velocities of the link COM
        lin_vel_w, ang_vel_w = self._view.get_velocities()[env_ids].split([3, 3], dim=-1)
        # if an offset is present or the COM does not agree with the link origin, the linear velocity has to be
        # transformed taking the angular velocity into account
        lin_vel_w += torch.linalg.cross(
            ang_vel_w, math_utils.quat_apply(quat_w, self._offset_pos_b[env_ids] - com_pos_b[env_ids]), dim=-1
        )

        # numerical derivative
        lin_acc_w = (lin_vel_w - self._prev_lin_vel_w[env_ids]) / self._dt + self._gravity_bias_w[env_ids]
        ang_acc_w = (ang_vel_w - self._prev_ang_vel_w[env_ids]) / self._dt
        # stack data in world frame and batch rotate
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

        # store sensor offset transformation
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
        up_axis = stage_utils.get_stage_up_axis()
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
