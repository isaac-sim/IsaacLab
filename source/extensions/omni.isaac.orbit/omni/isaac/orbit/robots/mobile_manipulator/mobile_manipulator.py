# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import Optional, Sequence

from omni.isaac.orbit.utils.math import quat_rotate_inverse

from ..legged_robot import LeggedRobot
from ..single_arm import SingleArmManipulator
from .mobile_manipulator_cfg import LeggedMobileManipulatorCfg, MobileManipulatorCfg
from .mobile_manipulator_data import LeggedMobileManipulatorData, MobileManipulatorData

__all__ = ["MobileManipulator", "LeggedMobileManipulator"]


class MobileManipulator(SingleArmManipulator):
    """Class for handling a mobile manipulator robot with a tool on it."""

    cfg: MobileManipulatorCfg
    """Configuration for the mobile manipulator."""

    def __init__(self, cfg: MobileManipulatorCfg):
        """Initialize the robot class.

        Args:
            cfg (MobileManipulatorCfg): The configuration instance.
        """
        # initialize parent
        super().__init__(cfg)
        # container for data access
        self._data = MobileManipulatorData()

    """
    Properties
    """

    @property
    def base_num_dof(self) -> int:
        """Number of DOFs in the robot base."""
        return self.cfg.meta_info.base_num_dof

    @property
    def data(self) -> MobileManipulatorData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def update_buffers(self, dt: float):
        # update parent buffers
        super().update_buffers(dt)
        # frame states
        # -- root frame in base
        self._data.root_vel_b[:, 0:3] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_lin_vel_w)
        self._data.root_vel_b[:, 3:6] = quat_rotate_inverse(self._data.root_quat_w, self._data.root_ang_vel_w)
        self._data.projected_gravity_b[:] = quat_rotate_inverse(self._data.root_quat_w, self._GRAVITY_VEC_W)

    """
    Internal helpers - Override.
    """

    def _create_buffers(self):
        """Create buffers for storing data."""
        # process parent buffers
        super()._create_buffers()
        # constants
        self._GRAVITY_VEC_W = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.count, 1)

        # frame states
        # -- base
        self._data.root_vel_b = torch.zeros(self.count, 6, dtype=torch.float, device=self.device)
        self._data.projected_gravity_b = torch.zeros(self.count, 3, dtype=torch.float, device=self.device)
        # dof states (redefined here to include base)
        # ---- base
        self._data.base_dof_pos = self._data.dof_pos[:, : self.base_num_dof]
        self._data.base_dof_vel = self._data.dof_vel[:, : self.base_num_dof]
        self._data.base_dof_acc = self._data.dof_acc[:, : self.base_num_dof]
        # ----- arm
        self._data.arm_dof_pos = self._data.dof_pos[:, self.base_num_dof : self.base_num_dof + self.arm_num_dof]
        self._data.arm_dof_vel = self._data.dof_vel[:, self.base_num_dof : self.base_num_dof + self.arm_num_dof]
        self._data.arm_dof_acc = self._data.dof_acc[:, self.base_num_dof : self.base_num_dof + self.arm_num_dof]
        # ---- tool
        self._data.tool_dof_pos = self._data.dof_pos[:, self.base_num_dof + self.arm_num_dof :]
        self._data.tool_dof_vel = self._data.dof_vel[:, self.base_num_dof + self.arm_num_dof :]
        self._data.tool_dof_acc = self._data.dof_acc[:, self.base_num_dof + self.arm_num_dof :]

    def _update_optional_buffers(self):
        """Update buffers from articulation that are optional."""
        # Note: we implement this function here to allow inherited classes decide whether these
        #   quantities need to be updated similarly or not.
        # -- dynamic state (note: base and tools don't contribute towards these quantities)
        start_index = self.base_num_dof
        end_index = start_index + self.arm_num_dof
        # jacobian
        if self.cfg.data_info.enable_jacobian:
            jacobians = self.articulations.get_jacobians(indices=self._ALL_INDICES, clone=False)
            # Returned jacobian: [batch, body, 6, dof] does not include the base body (i.e. the first link).
            # So we need to subtract 1 from the body index to get the correct jacobian.
            self._data.ee_jacobian[:] = jacobians[:, self.ee_body_index - 1, :, start_index:end_index]
        # mass matrix
        if self.cfg.data_info.enable_mass_matrix:
            mass_matrices = self.articulations.get_mass_matrices(indices=self._ALL_INDICES, clone=False)
            self._data.mass_matrix[:] = mass_matrices[:, start_index:end_index, start_index:end_index]
        # coriolis
        if self.cfg.data_info.enable_coriolis:
            forces = self.articulations.get_coriolis_and_centrifugal_forces(indices=self._ALL_INDICES, clone=False)
            self._data.coriolis[:] = forces[:, start_index:end_index]
        # gravity
        if self.cfg.data_info.enable_gravity:
            gravity = self.articulations.get_generalized_gravity_forces(indices=self._ALL_INDICES, clone=False)
            self._data.gravity[:] = gravity[:, start_index:end_index]


class LeggedMobileManipulator(MobileManipulator, LeggedRobot):
    """Class for handling a legged mobile manipulator with a tool on it."""

    cfg: LeggedMobileManipulatorCfg
    """Configuration for the legged mobile manipulator."""

    def __init__(self, cfg: LeggedMobileManipulatorCfg):
        """Initialize the robot class.

        Args:
            cfg (LeggedMobileManipulatorCfg): The configuration instance.
        """
        # initialize parent
        super().__init__(cfg)
        # container for data access
        self._data = LeggedMobileManipulatorData()

    """
    Properties
    """

    @property
    def data(self) -> LeggedMobileManipulatorData:
        """Data related to articulation."""
        return self._data

    """
    Operations.
    """

    def spawn(self, prim_path: str, translation: Sequence[float] = None, orientation: Sequence[float] = None):
        # spawn using parent
        LeggedRobot.spawn(self, prim_path, translation, orientation)

    def initialize(self, prim_paths_expr: Optional[str] = None):
        # reset parent buffers
        LeggedRobot.initialize(self, prim_paths_expr)
        MobileManipulator.initialize(self, prim_paths_expr)

    def reset_buffers(self, env_ids: Optional[Sequence[int]] = None):
        # reset parent buffers
        LeggedRobot.reset_buffers(self, env_ids)
        MobileManipulator.reset_buffers(self, env_ids)

    def update_buffers(self, dt: float):
        # update parent buffers
        LeggedRobot.update_buffers(self, dt)
        MobileManipulator.update_buffers(self, dt)
