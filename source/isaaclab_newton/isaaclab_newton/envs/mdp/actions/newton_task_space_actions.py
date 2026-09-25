# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from isaaclab_newton.controllers.differential_ik import NewtonDifferentialIKController
from isaaclab_newton.controllers.operational_space import NewtonOperationalSpaceController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .newton_task_space_actions_cfg import (
        NewtonDifferentialInverseKinematicsActionCfg,
        NewtonOperationalSpaceControllerActionCfg,
    )

logger = logging.getLogger(__name__)


class _NewtonTaskSpaceAction(ActionTerm):
    """Shared joint, body, and root-frame kinematics for the Newton task-space action terms."""

    _asset: Articulation

    def __init__(
        self,
        cfg: NewtonDifferentialInverseKinematicsActionCfg | NewtonOperationalSpaceControllerActionCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        self._jacobi_body_idx = self._body_idx - 1 if self._asset.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [j + self._asset.num_base_dofs for j in self._joint_ids]
        logger.info(
            f"Resolved joints {self._joint_names} [{self._joint_ids}] and body {self._body_name} [{self._body_idx}]"
            f" for the action term {self.__class__.__name__}."
        )
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        self._physics_dt = env.physics_dt
        self._ee_pose_b = torch.zeros(self.num_envs, 7, device=self.device)
        self._jacobian_b = torch.zeros(self.num_envs, 6, self._num_joints, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def _joint_pos_target(self, target: str) -> torch.Tensor:
        """Resolve a named joint posture target for the controlled joints."""
        if target == "default":
            return self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()
        if target == "center":
            return self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids].mean(dim=-1)
        if target == "zero":
            return torch.zeros(self.num_envs, self._num_joints, device=self.device)
        raise ValueError(f"Invalid null-space joint position target: {target}.")

    def _compute_ee_pose(self) -> torch.Tensor:
        """Compute the target-frame pose in the root frame, shape (num_envs, 7)."""
        data = self._asset.data
        pos_b, quat_b = math_utils.subtract_frame_transforms(
            data.root_pos_w.torch,
            data.root_quat_w.torch,
            data.body_pos_w.torch[:, self._body_idx],
            data.body_quat_w.torch[:, self._body_idx],
        )
        if self._offset_pos is not None:
            pos_b, quat_b = math_utils.combine_frame_transforms(pos_b, quat_b, self._offset_pos, self._offset_rot)
        self._ee_pose_b[:, :3], self._ee_pose_b[:, 3:] = pos_b, quat_b
        return self._ee_pose_b

    def _compute_ee_jacobian(self) -> torch.Tensor:
        """Compute the target-frame Jacobian in the root frame, shape (num_envs, 6, num_joints)."""
        data = self._asset.data
        jacobian_w = data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        root_rot = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w.torch))
        self._jacobian_b[:, :3] = torch.bmm(root_rot, jacobian_w[:, :3])
        self._jacobian_b[:, 3:] = torch.bmm(root_rot, jacobian_w[:, 3:])
        if self._offset_pos is not None:
            # v_ee = v_link + w_link x r_link_ee, then rotate into the target frame
            self._jacobian_b[:, :3] += torch.bmm(
                -math_utils.skew_symmetric_matrix(self._offset_pos), self._jacobian_b[:, 3:]
            )
            self._jacobian_b[:, 3:] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), self._jacobian_b[:, 3:])
        return self._jacobian_b


class NewtonDifferentialInverseKinematicsAction(_NewtonTaskSpaceAction):
    """Differential inverse-kinematics action term using :class:`NewtonDifferentialIKController`.

    The action is a target position, a target pose ``(x, y, z, qx, qy, qz, qw)``, or, in relative mode, a delta
    position or pose ``(x, y, z, rx, ry, rz)`` in the robot root frame. It sets joint position targets.
    """

    cfg: NewtonDifferentialInverseKinematicsActionCfg

    def __init__(self, cfg: NewtonDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        controller_cfg = self.cfg.controller
        if self.cfg.command_type == "position" and controller_cfg.axis_weight is None:
            controller_cfg = controller_cfg.replace(axis_weight=(1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
        self._controller = NewtonDifferentialIKController(controller_cfg, self.num_envs, self._num_joints, self.device)

        if self.cfg.command_type == "position":
            self._coordinate_names = ["x", "y", "z"]
        elif self.cfg.use_relative_mode:
            self._coordinate_names = ["x", "y", "z", "rx", "ry", "rz"]
        else:
            self._coordinate_names = ["x", "y", "z", "qx", "qy", "qz", "qw"]
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)
        self._clip = None
        if self.cfg.clip is not None:
            self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                self.num_envs, self.action_dim, 1
            )
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.clip, self._coordinate_names
            )
            self._clip[:, index_list] = torch.tensor(value_list, device=self.device)

        self._ee_pose_des = torch.zeros(self.num_envs, 7, device=self.device)
        self._null_space_target = None
        if controller_cfg.use_null_space_posture_control:
            self._null_space_target = self._joint_pos_target(self.cfg.null_space_joint_pos_target)
        # joint limits are set on the first apply, once the asset data is populated
        self._limits_set = not controller_cfg.use_joint_limit_avoidance

    @property
    def action_dim(self) -> int:
        return len(self._coordinate_names)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        if self._clip is not None:
            self._processed_actions[:] = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        # the target is held for the whole control step
        command = self._processed_actions
        ee_pose = self._compute_ee_pose()
        if self.cfg.command_type == "position":
            self._ee_pose_des[:, :3] = command + ee_pose[:, :3] if self.cfg.use_relative_mode else command
            self._ee_pose_des[:, 3:] = ee_pose[:, 3:]
        elif self.cfg.use_relative_mode:
            self._ee_pose_des[:, :3], self._ee_pose_des[:, 3:] = math_utils.apply_delta_pose(
                ee_pose[:, :3], ee_pose[:, 3:], command
            )
        else:
            # hold the current orientation when the commanded quaternion cannot be normalized
            quat = command[:, 3:7] / torch.linalg.norm(command[:, 3:7], dim=-1, keepdim=True)
            is_valid = torch.isfinite(quat).all(dim=-1, keepdim=True)
            self._ee_pose_des[:, :3] = command[:, :3]
            self._ee_pose_des[:, 3:] = torch.where(is_valid, quat, ee_pose[:, 3:])

    def apply_actions(self):
        if not self._limits_set:
            limits = self._asset.data.soft_joint_pos_limits.torch[:, self._joint_ids]
            self._controller.set_joint_pos_limits(limits[..., 0], limits[..., 1])
            self._limits_set = True
        joint_pos_des = self._controller.compute(
            self._compute_ee_pose(),
            self._ee_pose_des,
            self._compute_ee_jacobian(),
            self._asset.data.joint_pos.torch[:, self._joint_ids],
            self._physics_dt,
            null_space_joint_pos_target=self._null_space_target,
        )
        self._asset.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids)


class NewtonOperationalSpaceControllerAction(_NewtonTaskSpaceAction):
    """Operational-space control action term using :class:`NewtonOperationalSpaceController`.

    See :class:`~isaaclab_newton.envs.mdp.actions.NewtonOperationalSpaceControllerActionCfg` for the action
    layout. It sets joint effort targets.
    """

    cfg: NewtonOperationalSpaceControllerActionCfg

    def __init__(self, cfg: NewtonOperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        controller_cfg = self.cfg.controller
        for name in ("operational_frame_pose", "linear_selection_frame", "angular_selection_frame"):
            if getattr(controller_cfg, name) is None:
                raise ValueError(f"NewtonOperationalSpaceControllerAction requires a fixed controller '{name}'.")
        self._controller = NewtonOperationalSpaceController(
            controller_cfg, self.num_envs, self._num_joints, self.device
        )

        # action layout: pose, then optional wrench, stiffness, and damping
        pose_dim = 7 if self.cfg.target_type == "pose_abs" else 6
        self._use_wrench = controller_cfg.use_wrench_feedforward or controller_cfg.use_wrench_feedback
        self._slices: dict[str, slice] = {"pose": slice(0, pose_dim)}
        dim = pose_dim
        for name, enabled in (
            ("wrench", self._use_wrench),
            ("stiffness", controller_cfg.motion_stiffness is None),
            ("damping", controller_cfg.motion_damping is None),
        ):
            if enabled:
                self._slices[name] = slice(dim, dim + 6)
                dim += 6
        self._action_dim = dim
        self._raw_actions = torch.zeros(self.num_envs, dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.ones(self.num_envs, dim, device=self.device)
        self._scale[:, :3] = self.cfg.position_scale
        self._scale[:, 3:pose_dim] = self.cfg.orientation_scale
        for name, scale in (
            ("wrench", self.cfg.wrench_scale),
            ("stiffness", self.cfg.stiffness_scale),
            ("damping", self.cfg.damping_scale),
        ):
            if name in self._slices:
                self._scale[:, self._slices[name]] = scale

        frame = torch.tensor(controller_cfg.operational_frame_pose, device=self.device).repeat(self.num_envs, 1)
        self._frame_pos, self._frame_quat = frame[:, :3], frame[:, 3:]
        self._ee_pose_des = torch.zeros(self.num_envs, 7, device=self.device)
        self._ee_vel_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._ee_wrench_b = torch.zeros(self.num_envs, 6, device=self.device)
        self._null_space_target = None
        if controller_cfg.use_null_space_control:
            self._null_space_target = self._joint_pos_target(self.cfg.null_space_joint_pos_target)

        # the contact sensor measures forces only; moments stay open loop
        self._contact_sensor = None
        if controller_cfg.use_wrench_feedback:
            self._contact_sensor = ContactSensor(
                ContactSensorCfg(prim_path=self._asset.cfg.prim_path + "/" + self._body_name)
            )
            if not self._contact_sensor.is_initialized:
                self._contact_sensor._initialize_impl()
                self._contact_sensor._is_initialized = True

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale
        command = self._processed_actions[:, self._slices["pose"]]
        if self.cfg.target_type == "pose_rel":
            ee_pose = self._compute_ee_pose()
            ee_pos_op, ee_quat_op = math_utils.subtract_frame_transforms(
                self._frame_pos, self._frame_quat, ee_pose[:, :3], ee_pose[:, 3:]
            )
            self._ee_pose_des[:, :3], self._ee_pose_des[:, 3:] = math_utils.apply_delta_pose(
                ee_pos_op, ee_quat_op, command
            )
        else:
            self._ee_pose_des[:, :3] = command[:, :3]
            self._ee_pose_des[:, 3:] = math_utils.normalize(command[:, 3:])

    def apply_actions(self):
        cfg = self._controller.cfg
        data = self._asset.data
        ee_pose = self._compute_ee_pose()
        ee_wrench_des = self._processed_actions[:, self._slices["wrench"]] if self._use_wrench else None
        ee_wrench = None
        if self._contact_sensor is not None:
            self._contact_sensor.update(self._physics_dt)
            force_w = self._contact_sensor.data.net_normal_forces_w.torch[:, 0, :]
            self._ee_wrench_b[:, :3] = math_utils.quat_apply_inverse(data.root_quat_w.torch, force_w)
            self._ee_wrench_b[:, 3:] = ee_wrench_des[:, 3:]
            ee_wrench = self._ee_wrench_b
        joint_ids = self._jacobi_joint_ids
        efforts = self._controller.compute(
            self._compute_ee_jacobian(),
            ee_pose,
            self._compute_ee_velocity(),
            self._ee_pose_des,
            mass_matrix=data.mass_matrix.torch[:, joint_ids][:, :, joint_ids] if cfg.use_inertia_decoupling else None,
            gravity=data.gravity_compensation_forces.torch[:, joint_ids] if cfg.use_gravity_compensation else None,
            ee_wrench_des=ee_wrench_des,
            ee_wrench=ee_wrench,
            joint_pos=data.joint_pos.torch[:, self._joint_ids] if cfg.use_null_space_control else None,
            joint_vel=data.joint_vel.torch[:, self._joint_ids] if cfg.use_null_space_control else None,
            null_space_joint_pos_target=self._null_space_target,
            motion_stiffness=self._action_slice("stiffness"),
            motion_damping=self._action_slice("damping"),
        )
        self._asset.set_joint_effort_target_index(target=efforts, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self._contact_sensor is not None:
            self._contact_sensor.reset(env_ids)

    def _action_slice(self, name: str) -> torch.Tensor | None:
        return self._processed_actions[:, self._slices[name]] if name in self._slices else None

    def _compute_ee_velocity(self) -> torch.Tensor:
        """Compute the target-frame twist relative to the root, in the root frame, shape (num_envs, 6)."""
        data = self._asset.data
        root_quat_w = data.root_quat_w.torch
        relative_vel_w = data.body_link_vel_w.torch[:, self._body_idx] - data.root_link_vel_w.torch
        self._ee_vel_b[:, :3] = math_utils.quat_apply_inverse(root_quat_w, relative_vel_w[:, :3])
        self._ee_vel_b[:, 3:] = math_utils.quat_apply_inverse(root_quat_w, relative_vel_w[:, 3:])
        if self._offset_pos is not None:
            body_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(root_quat_w), data.body_quat_w.torch[:, self._body_idx]
            )
            offset_b = math_utils.quat_apply(body_quat_b, self._offset_pos)
            self._ee_vel_b[:, :3] += torch.cross(self._ee_vel_b[:, 3:], offset_b, dim=-1)
        return self._ee_vel_b
