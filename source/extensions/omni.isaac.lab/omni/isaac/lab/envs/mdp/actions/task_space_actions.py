# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.controllers.operational_space import OperationalSpaceController
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

    from . import actions_cfg


class DifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
        else:
            self._jacobi_body_idx = self._body_idx

        # log info for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        carb.log_info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the differential IK controller
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        # set command into controller
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        # obtain quantities from simulation
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        # compute the delta in joint-space
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        # set the joint position command
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        root_pose_w = self._asset.data.root_state_w[:, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._joint_ids]
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian


class OperationalSpaceControllerAction(ActionTerm):
    r"""Operational space controller action term.

    This action term performs pre-processing of the raw actions for operational space control.

    """

    cfg: actions_cfg.OperationalSpaceControllerActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.OperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._sim_dt = env.sim.get_physics_dt()

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_DoF = len(self._joint_ids)
        # parse the ee body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the ee body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first ee body index
        self._ee_body_idx = body_ids[0]
        self._ee_body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_ee_body_idx = self._ee_body_idx - 1
        else:
            self._jacobi_ee_body_idx = self._ee_body_idx

        # log info for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        carb.log_info(
            f"Resolved ee body name for the action term {self.__class__.__name__}:"
            f" {self._ee_body_name} [{self._ee_body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_DoF == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create the operational space controller
        self._opc = OperationalSpaceController(cfg=self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # create tensors for the dynamic-related quantities
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_DoF, device=self.device)
        self._jacobian_w = torch.zeros(self.num_envs, 6, self._num_DoF, device=self.device)
        self._mass_matrix = torch.zeros(self.num_envs, self._num_DoF, self._num_DoF, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create tensors for the kinematic-related quantities
        self._root_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._root_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_force_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_force_b = torch.zeros(self.num_envs, 6, device=self.device)

        # create contact sensor if and of the command is wrench_abs anf if stiffness is provided
        if (
            "wrench_abs" in self.cfg.controller_cfg.target_types
            and self.cfg.controller_cfg.wrench_stiffness is not None
        ):
            self._contact_sensor_cfg = ContactSensorCfg(prim_path=self._asset.cfg.prim_path + "/" + self._ee_body_name)
            self._contact_sensor = ContactSensor(self._contact_sensor_cfg)
        else:
            self._contact_sensor_cfg = None
            self._contact_sensor = None

        # create the joint effort tensor
        self._joint_efforts = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # save the scale as tensors
        self._position_scale = torch.tensor(self.cfg.position_scale, device=self.device)
        self._orientation_scale = torch.tensor(self.cfg.orientation_scale, device=self.device)
        self._wrench_scale = torch.tensor(self.cfg.wrench_scale, device=self.device)
        self._stiffness_scale = torch.tensor(self.cfg.stiffness_scale, device=self.device)
        self._damping_ratio_scale = torch.tensor(self.cfg.damping_ratio_scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # indexes for the various command elements (e.g., pose_rel, stifness, etc.) within the command tensor
        self._pose_abs_idx = None
        self._pose_rel_idx = None
        self._wrench_abs_idx = None
        self._stiffness_idx = None
        self._damping_ratio_idx = None
        self._resolve_command_indexes()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._opc.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):

        # Retrieve the jacobians to correctly refresh the ee_pose_b after a reset, see #993.
        self._compute_ee_jacobian()
        self._compute_ee_pose()

        # Store the raw actions. Please note that the actions contain task space targets
        # (in the order of the target_types), and possibly the impedance parameters depending on impedance_mode.
        self._raw_actions[:] = actions

        # Initialize the processed actions with raw actions.
        self._processed_actions[:] = self._raw_actions

        # Go through the command types one by one, and apply the pre-processing if needed.
        if self._pose_abs_idx is not None:
            self._processed_actions[:, self._pose_abs_idx : self._pose_abs_idx + 3] *= self._position_scale
            self._processed_actions[:, self._pose_abs_idx + 3 : self._pose_abs_idx + 7] *= self._orientation_scale
        if self._pose_rel_idx is not None:
            self._processed_actions[:, self._pose_rel_idx : self._pose_rel_idx + 3] *= self._position_scale
            self._processed_actions[:, self._pose_rel_idx + 3 : self._pose_rel_idx + 6] *= self._orientation_scale
        if self._wrench_abs_idx is not None:
            self._processed_actions[:, self._wrench_abs_idx : self._wrench_abs_idx + 6] *= self._wrench_scale
        if self._stiffness_idx is not None:
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] *= self._stiffness_scale
            self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6] = torch.clamp(
                self._processed_actions[:, self._stiffness_idx : self._stiffness_idx + 6],
                min=self.cfg.controller_cfg.stiffness_limits[0],
                max=self.cfg.controller_cfg.stiffness_limits[1],
            )
        if self._damping_ratio_idx is not None:
            self._processed_actions[
                :, self._damping_ratio_idx : self._damping_ratio_idx + 6
            ] *= self._damping_ratio_scale
            self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6] = torch.clamp(
                self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6],
                min=self.cfg.controller_cfg.damping_ratio_limits[0],
                max=self.cfg.controller_cfg.damping_ratio_limits[1],
            )

        # set command into controller
        self._opc.set_command(command=self._processed_actions, current_ee_pose=self._ee_pose_b)

    def apply_actions(self):

        # Update the relevant states and dynamical quantities
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        # Calculate the joint efforts
        self._joint_efforts[:] = self._opc.compute(
            jacobian=self._jacobian_b,
            current_ee_pose=self._ee_pose_b,
            current_ee_vel=self._ee_vel_b,
            current_ee_force=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
        )
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    """
    Helper functions.

    """

    def _resolve_command_indexes(self):
        """Resolves the indexes for the various command elements within the command tensor."""
        # First iterate over the target types to find the indexes of the different command elements
        cmd_idx = 0
        for target_type in self.cfg.controller_cfg.target_types:
            if target_type == "pose_abs":
                self._pose_abs_idx = cmd_idx
                cmd_idx += 7
            elif target_type == "pose_rel":
                self._pose_rel_idx = cmd_idx
                cmd_idx += 6
            elif target_type == "wrench_abs":
                self._wrench_abs_idx = cmd_idx
                cmd_idx += 6
            else:
                raise ValueError("Undefined target_type for OPC within OperationalSpaceControllerAction.")
        # Then iterate over the impedance parameters depending on the impedance mode
        if (
            self.cfg.controller_cfg.impedance_mode == "variable_kp"
            or self.cfg.controller_cfg.impedance_mode == "variable"
        ):
            self._stiffness_idx = cmd_idx
            cmd_idx += 6
            if self.cfg.controller_cfg.impedance_mode == "variable":
                self._damping_ratio_idx = cmd_idx
                cmd_idx += 6

        # Check if any command is left unresolved
        if self.action_dim != cmd_idx:
            raise ValueError("Not all command indexes have been resolved.")

    def _compute_dynamic_quantities(self):
        """Computes the dynamic quantities for operational space control."""

        self._mass_matrix[:] = self._asset.root_physx_view.get_mass_matrices()[:, self._joint_ids, :][
            :, :, self._joint_ids
        ]
        self._gravity[:] = self._asset.root_physx_view.get_generalized_gravity_forces()[:, self._joint_ids]

    def _compute_ee_jacobian(self):
        """Computes the geometric Jacobian of the ee body frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        self._jacobian_w[:] = self._asset.root_physx_view.get_jacobians()[
            :, self._jacobi_ee_body_idx, :, self._joint_ids
        ]

        # Convert the Jacobian from world to robot base frame
        self._jacobian_b[:] = self._jacobian_w
        root_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_state_w[:, 3:7]))
        self._jacobian_b[:, :3, :] = torch.bmm(root_rot_matrix, self._jacobian_b[:, :3, :])
        self._jacobian_b[:, 3:, :] = torch.bmm(root_rot_matrix, self._jacobian_b[:, 3:, :])

        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            self._jacobian_b[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), self._jacobian_b[:, 3:, :])  # type: ignore
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            self._jacobian_b[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), self._jacobian_b[:, 3:, :])  # type: ignore

    def _compute_ee_pose(self):
        """Computes the pose of the ee frame in the root frame."""
        # obtain quantities from simulation
        self._root_pose_w[:] = self._asset.data.root_state_w[:, :7]
        self._ee_pose_w[:] = self._asset.data.body_state_w[:, self._ee_body_idx, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            self._root_pose_w[:, 0:3], self._root_pose_w[:, 3:7], self._ee_pose_w[:, 0:3], self._ee_pose_w[:, 3:7]
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        # fill in the pose tensor
        self._ee_pose_b[:, 0:3] = ee_pose_b
        self._ee_pose_b[:, 3:7] = ee_quat_b

    def _compute_ee_velocity(self):
        """Computes the velocity of the ee frame in the root frame."""
        self._root_vel_w[:] = self._asset.data.root_vel_w  # Extract root velocity in the world frame
        self._ee_vel_w[:] = self._asset.data.body_vel_w[
            :, self._ee_body_idx, :
        ]  # Extract end-effector velocity in the world frame
        relative_vel_w = self._ee_vel_w - self._root_vel_w  # Compute the relative velocity in the world frame
        ee_lin_vel_b = math_utils.quat_rotate_inverse(
            self._asset.data.root_quat_w, relative_vel_w[:, 0:3]
        )  # From world to root frame
        ee_ang_vel_b = math_utils.quat_rotate_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 3:6])
        # fill in the velocity tensor
        self._ee_vel_b[:, 0:3] = ee_lin_vel_b
        self._ee_vel_b[:, 3:6] = ee_ang_vel_b

        # FIXME Account for the body offset

    def _compute_ee_force(self):
        """Computes the contact forces on the ee frame in the root frame."""
        if self._contact_sensor is not None:
            self._contact_sensor.update(self._sim_dt)
            self._ee_force_w[:] = self._contact_sensor.data.net_forces_w[:, self._ee_body_idx, :]  # type: ignore

            self._ee_force_b[:] = (
                self._ee_force_w
            )  # FIXME: Rotate the vector to body frame and account for the body offset
