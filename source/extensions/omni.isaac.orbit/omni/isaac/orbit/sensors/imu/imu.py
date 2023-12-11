# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import carb

import omni.isaac.orbit.sim as sim_utils
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView, XFormPrimView
from pxr import UsdPhysics

from omni.isaac.orbit.markers import VisualizationMarkers
import omni.isaac.orbit.utils.math as math_utils

from ..sensor_base import SensorBase
from .imu_data import IMUData
from ..camera.utils import create_rotation_matrix_from_view, convert_orientation_convention

if TYPE_CHECKING:
    from .imu_cfg import IMUCfg


class IMU(SensorBase):
    """The inertia measurement unit."""

    cfg: IMUCfg
    """The configuration parameters."""

    def __init__(self, cfg: IMUCfg):
        """Initializes the IMU sensor.

        Args:
            cfg: The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)
        # Create empty variables for storing output data
        self._data = IMUData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"IMU sensor @ '{self.cfg.prim_path}': \n"
            f"\tview type         : {self._view.__class__}\n"
            f"\tupdate period (s) : {self.cfg.update_period}\n"
            f"\tnumber of sensors : {self._view.count}\n"
        )

    """
    Properties
    """

    @property
    def data(self) -> IMUData:
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
        self._data.quat_w[env_ids] = 0.0
        self._data.ang_vel_w[env_ids] = 0.0
        self._data.lin_acc_w[env_ids] = 0.0
        # Set all reset sensors to not outdated since their value won't be updated till next sim step.
        self._is_outdated[env_ids] = False
        # resample the drift
        self.drift[env_ids].uniform_(*self.cfg.drift_range)

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
            RuntimeError: If the number of camera prims in the view does not match the number of environments.
        """
        # Initialize parent class
        super()._initialize_impl()
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")
        # check if the prim at path is an articulated or rigid prim
        # we do this since for physics-based view classes we can access their data directly
        # otherwise we need to use the xform view class which is slower
        found_supported_prim_class = False
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
        # check if it is a rigid prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            found_supported_prim_class = True
        else:
            self._view = XFormPrimView(self.cfg.prim_path, reset_xform_properties=False)
            found_supported_prim_class = True
            carb.log_warn(f"The prim at path {prim.GetPath().pathString} is not a physics prim! Using XFormPrimView.")
        # check if prim view class is found
        if not found_supported_prim_class:
            raise RuntimeError(f"Failed to find a valid prim view class for the prim paths: {self.cfg.prim_path}")

        # Create internal buffers
        self._initialize_buffers_impl()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrimView):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # get velocities
        vel = self.body_physx_view.get_velocities()
        self._data.ang_vel_w[env_ids] = vel[env_ids, 3:]
        self._data.lin_acc_w[env_ids] = (vel[env_ids, :3] - self._last_lin_vel_w[env_ids]) / self._dt
        self._last_lin_vel_w[env_ids] = vel[env_ids, :3]

    def _initialize_buffers_impl(self):
        """Create buffers for storing data."""
        # data buffers
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.quat_w[:, 0] = 1.0
        self._data.lin_acc_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.ang_vel_w = torch.zeros(self._view.count, 3, device=self._device)
        # internal buffers
        self._last_lin_vel_w = torch.zeros(self._view.count, 3, device=self._device)
        # prepare drift
        self.drift = torch.zeros(self._view.count, 3, device=self.device)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
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
        # TODO: check if necessary
        if self._view is None:
            return
        # get marker location
        # -- base state
        base_pos_w = self._data.pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales
        default_scale = self.acceleration_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self._data.lin_acc_w.shape[0], 1)
        # arrow-direction
        # -- the function is meant for camera where the image is in -z direction, as here the "look" should be in +z 
        #    direction, we have to switch the sign in front of z target location
        quat_opengl = math_utils.quat_from_matrix(create_rotation_matrix_from_view(self._data.pos_w, self._data.pos_w + self._data.lin_acc_w * torch.tensor([[1, 1, -1]], device=self._device), device=self._device))
        arrow_quat = convert_orientation_convention(quat_opengl, "ros", "world")
        # display markers
        self.acceleration_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)
