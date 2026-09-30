# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative IK and a scalar, continuous gripper opening."""

import torch

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass


class BerryIKAction(DifferentialInverseKinematicsAction):
    """Advance contact at the robot physics rate [120 Hz]."""

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite IK command")
        # Keep carried tissue inside the explicitly allocated MPM work volume.
        pos, _ = self._compute_frame_pose()
        offset = torch.tensor(self._env.cfg.berry_position, device=self.device)
        # All local grids must cover the commanded workspace, including edge berries.
        reach = 0.065 if self._env.cfg.berry == "all" else 0.09
        lower = offset + torch.tensor([-reach, -reach, 0.005], device=self.device)
        upper_y = 0.205 if self._env.cfg.background == "ebc" else reach
        upper = offset + torch.tensor([reach, upper_y, 0.18], device=self.device)
        limited = actions.clone()
        limited[:, :3] = (pos + actions[:, :3]).clamp(lower, upper) - pos
        super().process_actions(limited)
        self._ik_pending = True

    def apply_actions(self):
        # The teleop pose target changes at 30 Hz. Hold its joint solution while
        # the 120 Hz drives and two-way tissue contact continue to run.
        if getattr(self, "_ik_pending", True):
            super().apply_actions()
            self._ik_pending = False
        berry = getattr(self._env, "berry", None)
        if berry is not None:
            self._env.advance_berries()

    def _compute_frame_jacobian(self):
        self._jacobian_b[:] = self.jacobian_b
        body_quat = self._asset.data.body_quat_w.torch[:, self._body_idx]
        base_quat = self._asset.data.root_quat_w.torch
        hand_in_base = math_utils.quat_mul(math_utils.quat_inv(base_quat), body_quat)
        offset_in_base = math_utils.quat_apply(hand_in_base, self._offset_pos)
        self._jacobian_b[:, :3, :] -= torch.bmm(
            math_utils.skew_symmetric_matrix(offset_in_base), self._jacobian_b[:, 3:, :]
        )
        return self._jacobian_b


class BerryGraspAction(ActionTerm):
    """Map [-1, 1] continuously to total finger aperture [0, 0.08 m]."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints("panda_finger_joint.*")
        self._raw = torch.ones((self.num_envs, 1), device=self.device)
        self._target = torch.full((self.num_envs, 2), 0.04, device=self.device)

    @property
    def action_dim(self):
        return 1

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._target

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite grasp command")
        self._raw[:] = actions.clamp(-1, 1)
        self._target[:] = (self._raw + 1) * 0.02

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self._raw[env_ids] = 1
        self._target[env_ids] = 0.04


@configclass
class BerryGraspActionCfg(ActionTermCfg):
    class_type: type = BerryGraspAction
