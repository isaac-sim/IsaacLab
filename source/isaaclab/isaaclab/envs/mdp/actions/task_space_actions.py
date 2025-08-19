# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.operational_space import OperationalSpaceController
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg, FrameTransformer, FrameTransformerCfg
from isaaclab.sim.utils import find_matching_prims

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

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
    _clip: torch.Tensor
    """The clip applied to the input action."""

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

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

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

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - body_name: The name of the body.
        - joint_names: The names of the joints.
        - scale: The scale of the action term.
        - clip: The clip of the action term.
        - controller_cfg: The configuration of the controller.
        - body_offset: The offset of the body.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TaskSpaceAction"
        self._IO_descriptor.body_name = self._body_name
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.scale = self._scale
        if self.cfg.clip is not None:
            self._IO_descriptor.clip = self.cfg.clip
        else:
            self._IO_descriptor.clip = None
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        self._IO_descriptor.extras["body_offset"] = self.cfg.body_offset.__dict__
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
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
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        root_pos_w = self._asset.data.root_pos_w
        root_quat_w = self._asset.data.root_quat_w
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
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
        jacobian = self.jacobian_b
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
    _contact_sensor: ContactSensor = None
    """The contact sensor for the end-effector body."""
    _task_frame_transformer: FrameTransformer = None
    """The frame transformer for the task frame."""

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
            self._jacobi_joint_idx = self._joint_ids
        else:
            self._jacobi_ee_body_idx = self._ee_body_idx
            self._jacobi_joint_idx = [i + 6 for i in self._joint_ids]

        # log info for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved ee body name for the action term {self.__class__.__name__}:"
            f" {self._ee_body_name} [{self._ee_body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_DoF == self._asset.num_joints:
            self._joint_ids = slice(None)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        # create contact sensor if any of the command is wrench_abs, and if stiffness is provided
        if (
            "wrench_abs" in self.cfg.controller_cfg.target_types
            and self.cfg.controller_cfg.contact_wrench_stiffness_task is not None
        ):
            self._contact_sensor_cfg = ContactSensorCfg(prim_path=self._asset.cfg.prim_path + "/" + self._ee_body_name)
            self._contact_sensor = ContactSensor(self._contact_sensor_cfg)
            if not self._contact_sensor.is_initialized:
                self._contact_sensor._initialize_impl()
                self._contact_sensor._is_initialized = True

        # Initialize the task frame transformer if a relative path for the RigidObject, representing the task frame,
        # is provided.
        if self.cfg.task_frame_rel_path is not None:
            # The source RigidObject can be any child of the articulation asset (we will not use it),
            # hence, we will use the first RigidObject child.
            root_rigidbody_path = self._first_RigidObject_child_path()
            task_frame_transformer_path = "/World/envs/env_.*/" + self.cfg.task_frame_rel_path
            task_frame_transformer_cfg = FrameTransformerCfg(
                prim_path=root_rigidbody_path,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        name="task_frame",
                        prim_path=task_frame_transformer_path,
                    ),
                ],
            )
            self._task_frame_transformer = FrameTransformer(task_frame_transformer_cfg)
            if not self._task_frame_transformer.is_initialized:
                self._task_frame_transformer._initialize_impl()
                self._task_frame_transformer._is_initialized = True
            # create tensor for task frame pose in the root frame
            self._task_frame_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        else:
            # create an empty reference for task frame pose
            self._task_frame_pose_b = None

        # create the operational space controller
        self._osc = OperationalSpaceController(cfg=self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # create tensors for the dynamic-related quantities
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_DoF, device=self.device)
        self._mass_matrix = torch.zeros(self.num_envs, self._num_DoF, self._num_DoF, device=self.device)
        self._gravity = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create tensors for the ee states
        self._ee_pose_w = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_pose_b_no_offset = torch.zeros(self.num_envs, 7, device=self.device)  # The original ee without offset
        self._ee_vel_w = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_force_w = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now
        self._ee_force_b = torch.zeros(self.num_envs, 3, device=self.device)  # Only the forces are used for now

        # create tensors for the joint states
        self._joint_pos = torch.zeros(self.num_envs, self._num_DoF, device=self.device)
        self._joint_vel = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # create the joint effort tensor
        self._joint_efforts = torch.zeros(self.num_envs, self._num_DoF, device=self.device)

        # save the scale as tensors
        self._position_scale = torch.tensor(self.cfg.position_scale, device=self.device)
        self._orientation_scale = torch.tensor(self.cfg.orientation_scale, device=self.device)
        self._wrench_scale = torch.tensor(self.cfg.wrench_scale, device=self.device)
        self._stiffness_scale = torch.tensor(self.cfg.stiffness_scale, device=self.device)
        self._damping_ratio_scale = torch.tensor(self.cfg.damping_ratio_scale, device=self.device)

        # indexes for the various command elements (e.g., pose_rel, stifness, etc.) within the command tensor
        self._pose_abs_idx = None
        self._pose_rel_idx = None
        self._wrench_abs_idx = None
        self._stiffness_idx = None
        self._damping_ratio_idx = None
        self._resolve_command_indexes()

        # Nullspace position control joint targets
        self._nullspace_joint_pos_target = None
        self._resolve_nullspace_joint_pos_targets()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of operational space control."""
        return self._osc.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Raw actions for operational space control."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Processed actions for operational space control."""
        return self._processed_actions

    @property
    def jacobian_w(self) -> torch.Tensor:
        return self._asset.root_physx_view.get_jacobians()[:, self._jacobi_ee_body_idx, :, self._jacobi_joint_idx]

    @property
    def jacobian_b(self) -> torch.Tensor:
        jacobian = self.jacobian_w
        base_rot = self._asset.data.root_quat_w
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - body_name: The name of the body.
        - joint_names: The names of the joints.
        - position_scale: The scale of the position.
        - orientation_scale: The scale of the orientation.
        - wrench_scale: The scale of the wrench.
        - stiffness_scale: The scale of the stiffness.
        - damping_ratio_scale: The scale of the damping ratio.
        - nullspace_joint_pos_target: The nullspace joint pos target.
        - clip: The clip of the action term.
        - controller_cfg: The configuration of the controller.
        - body_offset: The offset of the body.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "TaskSpaceAction"
        self._IO_descriptor.body_name = self._ee_body_name
        self._IO_descriptor.joint_names = self._joint_names
        self._IO_descriptor.position_scale = self.cfg.position_scale
        self._IO_descriptor.orientation_scale = self.cfg.orientation_scale
        self._IO_descriptor.wrench_scale = self.cfg.wrench_scale
        self._IO_descriptor.stiffness_scale = self.cfg.stiffness_scale
        self._IO_descriptor.damping_ratio_scale = self.cfg.damping_ratio_scale
        self._IO_descriptor.nullspace_joint_pos_target = self.cfg.nullspace_joint_pos_target
        if self.cfg.clip is not None:
            self._IO_descriptor.clip = self.cfg.clip
        else:
            self._IO_descriptor.clip = None
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller_cfg.__dict__
        self._IO_descriptor.extras["body_offset"] = self.cfg.body_offset.__dict__
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions and sets them as commands for for operational space control.

        Args:
            actions (torch.Tensor): The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """

        # Update ee pose, which would be used by relative targets (i.e., pose_rel)
        self._compute_ee_pose()

        # Update task frame pose w.r.t. the root frame.
        self._compute_task_frame_pose()

        # Pre-process the raw actions for operational space control.
        self._preprocess_actions(actions)

        # set command into controller
        self._osc.set_command(
            command=self._processed_actions,
            current_ee_pose_b=self._ee_pose_b,
            current_task_frame_pose_b=self._task_frame_pose_b,
        )

    def apply_actions(self):
        """Computes the joint efforts for operational space control and applies them to the articulation."""

        # Update the relevant states and dynamical quantities
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()
        # Calculate the joint efforts
        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )
        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Resets the raw actions and the sensors if available.

        Args:
            env_ids (Sequence[int] | None): The environment indices to reset. If ``None``, all environments are reset.
        """
        self._raw_actions[env_ids] = 0.0
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)
        if self._task_frame_transformer is not None:
            self._task_frame_transformer.reset(env_ids)

    """
    Helper functions.

    """

    def _first_RigidObject_child_path(self):
        """Finds the first ``RigidObject`` child under the articulation asset.

        Raises:
            ValueError: If no child ``RigidObject`` is found under the articulation asset.

        Returns:
            str: The path to the first ``RigidObject`` child under the articulation asset.
        """
        child_prims = find_matching_prims(self._asset.cfg.prim_path + "/.*")
        rigid_child_prim = None
        # Loop through the list and stop at the first RigidObject found
        for prim in child_prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_child_prim = prim
                break
        if rigid_child_prim is None:
            raise ValueError("No child rigid body found under the expression: '{self._asset.cfg.prim_path}'/.")
        rigid_child_prim_path = rigid_child_prim.GetPath().pathString
        # Remove the specific env index from the path string
        rigid_child_prim_path = self._asset.cfg.prim_path + "/" + rigid_child_prim_path.split("/")[-1]
        return rigid_child_prim_path

    def _resolve_command_indexes(self):
        """Resolves the indexes for the various command elements within the command tensor.

        Raises:
            ValueError: If any command index is left unresolved.
        """
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
                raise ValueError("Undefined target_type for OSC within OperationalSpaceControllerAction.")
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

    def _resolve_nullspace_joint_pos_targets(self):
        """Resolves the nullspace joint pos targets for the operational space controller.

        Raises:
            ValueError: If the nullspace joint pos targets are set when null space control is not set to 'position'.
            ValueError: If the nullspace joint pos targets are not set when null space control is set to 'position'.
            ValueError: If an invalid value is set for nullspace joint pos targets.
        """

        if self.cfg.nullspace_joint_pos_target != "none" and self.cfg.controller_cfg.nullspace_control != "position":
            raise ValueError("Nullspace joint targets can only be set when null space control is set to 'position'.")

        if self.cfg.nullspace_joint_pos_target == "none" and self.cfg.controller_cfg.nullspace_control == "position":
            raise ValueError("Nullspace joint targets must be set when null space control is set to 'position'.")

        if self.cfg.nullspace_joint_pos_target == "zero" or self.cfg.nullspace_joint_pos_target == "none":
            # Keep the nullspace joint targets as None as this is later processed as zero in the controller
            self._nullspace_joint_pos_target = None
        elif self.cfg.nullspace_joint_pos_target == "center":
            # Get the center of the robot soft joint limits
            self._nullspace_joint_pos_target = torch.mean(
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :], dim=-1
            )
        elif self.cfg.nullspace_joint_pos_target == "default":
            # Get the default joint positions
            self._nullspace_joint_pos_target = self._asset.data.default_joint_pos[:, self._joint_ids]
        else:
            raise ValueError("Invalid value for nullspace joint pos targets.")

    def _compute_dynamic_quantities(self):
        """Computes the dynamic quantities for operational space control."""

        self._mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[:, self._joint_ids, :][
            :, :, self._joint_ids
        ]
        self._gravity[:] = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]

    def _compute_ee_jacobian(self):
        """Computes the geometric Jacobian of the ee body frame in root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # Get the Jacobian in root frame
        self._jacobian_b[:] = self.jacobian_b

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
        """Computes the pose of the ee frame in root frame."""
        # Obtain quantities from simulation
        self._ee_pose_w[:, 0:3] = self._asset.data.body_pos_w[:, self._ee_body_idx]
        self._ee_pose_w[:, 3:7] = self._asset.data.body_quat_w[:, self._ee_body_idx]
        # Compute the pose of the ee body in the root frame
        self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7] = math_utils.subtract_frame_transforms(
            self._asset.data.root_pos_w,
            self._asset.data.root_quat_w,
            self._ee_pose_w[:, 0:3],
            self._ee_pose_w[:, 3:7],
        )
        # Account for the offset
        if self.cfg.body_offset is not None:
            self._ee_pose_b[:, 0:3], self._ee_pose_b[:, 3:7] = math_utils.combine_frame_transforms(
                self._ee_pose_b_no_offset[:, 0:3], self._ee_pose_b_no_offset[:, 3:7], self._offset_pos, self._offset_rot
            )
        else:
            self._ee_pose_b[:] = self._ee_pose_b_no_offset

    def _compute_ee_velocity(self):
        """Computes the velocity of the ee frame in root frame."""
        # Extract end-effector velocity in the world frame
        self._ee_vel_w[:] = self._asset.data.body_vel_w[:, self._ee_body_idx, :]
        # Compute the relative velocity in the world frame
        relative_vel_w = self._ee_vel_w - self._asset.data.root_vel_w

        # Convert ee velocities from world to root frame
        self._ee_vel_b[:, 0:3] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 0:3])
        self._ee_vel_b[:, 3:6] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, relative_vel_w[:, 3:6])

        # Account for the offset
        if self.cfg.body_offset is not None:
            # Compute offset vector in root frame
            r_offset_b = math_utils.quat_apply(self._ee_pose_b_no_offset[:, 3:7], self._offset_pos)
            # Adjust the linear velocity to account for the offset
            self._ee_vel_b[:, :3] += torch.cross(self._ee_vel_b[:, 3:], r_offset_b, dim=-1)
            # Angular velocity is not affected by the offset

    def _compute_ee_force(self):
        """Computes the contact forces on the ee frame in root frame."""
        # Obtain contact forces only if the contact sensor is available
        if self._contact_sensor is not None:
            self._contact_sensor.update(self._sim_dt)
            self._ee_force_w[:] = self._contact_sensor.data.net_forces_w[:, 0, :]  # type: ignore
            # Rotate forces and torques into root frame
            self._ee_force_b[:] = math_utils.quat_apply_inverse(self._asset.data.root_quat_w, self._ee_force_w)

    def _compute_joint_states(self):
        """Computes the joint states for operational space control."""
        # Extract joint positions and velocities
        self._joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_vel[:] = self._asset.data.joint_vel[:, self._joint_ids]

    def _compute_task_frame_pose(self):
        """Computes the pose of the task frame in root frame."""
        # Update task frame pose if task frame rigidbody is provided
        if self._task_frame_transformer is not None and self._task_frame_pose_b is not None:
            self._task_frame_transformer.update(self._sim_dt)
            # Calculate the pose of the task frame in the root frame
            self._task_frame_pose_b[:, :3], self._task_frame_pose_b[:, 3:] = math_utils.subtract_frame_transforms(
                self._asset.data.root_pos_w,
                self._asset.data.root_quat_w,
                self._task_frame_transformer.data.target_pos_w[:, 0, :],
                self._task_frame_transformer.data.target_quat_w[:, 0, :],
            )

    def _preprocess_actions(self, actions: torch.Tensor):
        """Pre-processes the raw actions for operational space control.

        Args:
            actions (torch.Tensor): The raw actions for operational space control. It is a tensor of
                shape (``num_envs``, ``action_dim``).
        """
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
                min=self.cfg.controller_cfg.motion_stiffness_limits_task[0],
                max=self.cfg.controller_cfg.motion_stiffness_limits_task[1],
            )
        if self._damping_ratio_idx is not None:
            self._processed_actions[
                :, self._damping_ratio_idx : self._damping_ratio_idx + 6
            ] *= self._damping_ratio_scale
            self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6] = torch.clamp(
                self._processed_actions[:, self._damping_ratio_idx : self._damping_ratio_idx + 6],
                min=self.cfg.controller_cfg.motion_damping_ratio_limits_task[0],
                max=self.cfg.controller_cfg.motion_damping_ratio_limits_task[1],
            )
