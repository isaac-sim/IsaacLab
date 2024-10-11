# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.assets.articulation import Articulation
from omni.isaac.lab.assets.rigid_object import RigidObject
from omni.isaac.lab.controllers.differential_ik import DifferentialIKController
from omni.isaac.lab.managers.action_manager import ActionTerm

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
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
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
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
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


class RigidObjectActionTerm(ActionTerm):
    """Simple action term that implements a PD controller to track a target position.

    The action term is applied to the cube asset. It involves two steps:

    1. **Process the raw actions**: Typically, this includes any transformations of the raw actions
       that are required to map them to the desired space. This is called once per environment step.
    2. **Apply the processed actions**: This step applies the processed actions to the asset.
       It is called once per simulation step.

    In this case, the action term simply applies the raw actions to the cube asset. The raw actions
    are the desired target positions of the cube in the environment frame. The pre-processing step
    simply copies the raw actions to the processed actions as no additional processing is required.
    The processed actions are then applied to the cube asset by implementing a PD controller to
    track the target position.
    """

    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.RigidObjectPoseActionTermCfg, env: ManagerBasedEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        self._vel_command = torch.zeros(self.num_envs, 6, device=self.device)
        
        self.obj_pos_des = torch.zeros(self.num_envs, 3, device=self.device)
        self.obj_quat_des = torch.zeros(self.num_envs, 4, device=self.device)
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        
        # gains of controller
        self.p_gain = cfg.p_gain
        self.d_gain = cfg.d_gain
        self.initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.act_lows = torch.tensor(cfg.lows, device=self.device)[None]
        self.act_highs = torch.tensor(cfg.highs, device=self.device)[None]
        
    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qw, qx, qy, qz)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self.initialized[env_ids] = False
        
    """
    Operations
    """
    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        root_pose_w = self._asset.data.root_state_w[..., :7]
        obj_pose_w = root_pose_w[:, :3] - self._env.scene.env_origins
        obj_quat_w = root_pose_w[:, 3:]
        # ee_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        # root_pose_w = self._asset.data.root_state_w[:, :7]
        # # compute the pose of the body in the root frame
        # ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
        #     root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        # )
        # # account for the offset
        # if self.cfg.body_offset is not None:
        #     ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
        #         ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
        #     )
        return obj_pose_w, obj_quat_w

    def set_command(
        self, command: torch.Tensor, obj_pos: torch.Tensor | None = None, obj_quat: torch.Tensor | None = None
    ):
        """Set target object pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (w, x, y, z) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if obj_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if obj_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self.obj_pos_des[:] = obj_pos + self._command
                self.obj_quat_des[:] = obj_quat
            else:
                self.obj_pos_des[:] = self._command
                self.obj_quat_des[:] = obj_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if obj_pos is None or obj_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self.obj_pos_des, self.obj_quat_des = math_utils.apply_delta_pose(obj_pos, obj_quat, self._command)
            else:
                self.obj_pos_des = self._command[:, 0:3]
                self.obj_quat_des = self._command[:, 3:7]
                
    def process_actions(self, actions: torch.Tensor):
        #         ----------------------------------------------
        # Toggle gripper (open/close): K
        # Move arm along x-axis: W/S
        # Move arm along y-axis: A/D
        # Move arm along z-axis: Q/E
        # Rotate arm along x-axis: Z/X
        # Rotate arm along y-axis: T/G
        # Rotate arm along z-axis: C/V
        # print("------------------------------------")
        # print(actions)
        # store the raw actions
        # actions[:] = 0
        # actions[:, 2] = -0.00001
        # actions[:, 5] = -0.2
        
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]
        obj_pos_curr, obj_quat_curr = self._compute_frame_pose()
     
        # set command into controller
        if self.cfg.is_accumulate_action:
            self.obj_pos_des[~self.initialized] = obj_pos_curr[~self.initialized]
            self.obj_quat_des[~self.initialized] = obj_quat_curr[~self.initialized]
            self.initialized[~self.initialized] = True
            self.set_command(self._processed_actions, self.obj_pos_des, self.obj_quat_des)
        else:
            self.set_command(self._processed_actions, obj_pos_curr, obj_quat_curr)
        # print("Current ", obj_pos_curr, obj_quat_curr)
        # print("Destination ", self.obj_pos_des, self.obj_quat_des)

    def apply_actions(self):
        # implement a PD controller to track the target pose
        obj_pos_curr, obj_quat_curr = self._compute_frame_pose()
        pos_error, rot_error = math_utils.compute_pose_error(obj_pos_curr, obj_quat_curr,
                                                             self.obj_pos_des, self.obj_quat_des)
        pos_error = torch.clamp(pos_error, self.act_lows[:, :3], self.act_highs[:, :3])
        rot_error = torch.clamp(rot_error, self.act_lows[:, 3:], self.act_highs[:, 3:])                                                   
        pos_vel_error = -self._asset.data.root_lin_vel_w
        rot_vel_error = -self._asset.data.root_ang_vel_w
        
        # set velocity targets
        self._vel_command[:, :3] = self.p_gain * pos_error + self.d_gain * pos_vel_error
        self._vel_command[:, 3:] = self.p_gain * rot_error + self.d_gain * rot_vel_error
        self._asset.write_root_velocity_to_sim(self._vel_command)
        
