# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics
import ipdb

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
    from . import actions_cfg


class DifferentialInverseKinematicsAction(ActionTerm):
    r"""Inverse Kinematics action term."""

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    _asset: Articulation
    _scale: torch.Tensor
    _clip: torch.Tensor

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)

        # end-effector body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]

        # base_link body for root frame
        base_ids, base_names = self._asset.find_bodies("base_link")
        if len(base_ids) != 1:
            raise ValueError(f"Could not find a unique base_link in the articulation: {base_names}")
        self._base_body_idx = base_ids[0]

        # jacobian indexing
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
            self._jacobi_joint_ids = self._joint_ids
        else:
            self._jacobi_body_idx = self._body_idx
            self._jacobi_joint_ids = [i + 6 for i in self._joint_ids]

        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved body name for {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )

        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device
        )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None

        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

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
        base_rot = self._asset.data.body_quat_w[:, self._base_body_idx]
        base_rot_matrix = math_utils.matrix_from_quat(math_utils.quat_inv(base_rot))
        jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
        else:
            joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)
        # ipdb.set_trace()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx]
        base_pos_w = self._asset.data.body_pos_w[:, self._base_body_idx]
        if ee_pos_w[0][1] > 0.0:
            offset = torch.tensor([-0.0896, 0.051, -0.009], device=base_pos_w.device).unsqueeze(0)
        else:
            offset = torch.tensor([-0.0896, -0.051, -0.009], device=base_pos_w.device).unsqueeze(0)
        base_pos_w = base_pos_w + offset
        base_quat_w = self._asset.data.body_quat_w[:, self._base_body_idx]
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            base_pos_w, base_quat_w, ee_pos_w, ee_quat_w
        )
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )
        return ee_pose_b, ee_quat_b

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b
        if self.cfg.body_offset is not None:
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])
        return jacobian


class OperationalSpaceControllerAction(ActionTerm):
    """Operational space controller action term."""

    cfg: actions_cfg.OperationalSpaceControllerActionCfg
    _asset: Articulation
    _contact_sensor: ContactSensor = None
    _task_frame_transformer: FrameTransformer = None

    def __init__(self, cfg: actions_cfg.OperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._sim_dt = env.sim.get_physics_dt()

        # joints
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_DoF = len(self._joint_ids)

        # ee body
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the ee body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        self._ee_body_idx = body_ids[0]
        self._ee_body_name = body_names[0]

        # base_link body for root frame
        base_ids, base_names = self._asset.find_bodies("base_link")
        if len(base_ids) != 1:
            raise ValueError(f"Could not find a unique base_link in the articulation: {base_names}")
        self._base_body_idx = base_ids[0]

        if self._asset.is_fixed_base:
            self._jacobi_ee_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_idx = self._joint_ids
        else:
            self._jacobi_ee_body_idx = self._ee_body_idx
            self._jacobi_joint_idx = [i + 6 for i in self._joint_ids]

        omni.log.info(
            f"Resolved joint names for {self.__class__.__name__}: {self._joint_names} [{self._joint_ids}]"
        )
        omni.log.info(
            f"Resolved ee body name for {self.__class__.__name__}: {self._ee_body_name} [{self._ee_body_idx}]"
        )

        if self._num_DoF == self._asset.num_joints:
            self._joint_ids = slice(None)

        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self
