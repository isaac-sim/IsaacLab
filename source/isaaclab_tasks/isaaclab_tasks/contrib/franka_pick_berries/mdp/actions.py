# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative IK and a scalar, continuous gripper opening."""

import torch
import warp as wp

from isaaclab.envs.mdp.actions.task_space_actions import (
    DifferentialInverseKinematicsAction,
)
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from ..physics.coupling import set_grasping

HOLD_TOLERANCE = 0.0005
"""Distance [m] between commanded and current finger opening within which the fingers hold their aperture."""

HOLD_MAX_OPENING = 0.036
"""Commanded finger opening [m] from which the gripper is open, not holding (fully open is 0.04)."""

HOLD_MAX_SPEED = 0.001
"""Finger speed [m/s] above which the fingers are moving, not holding."""


class BerryIKAction(DifferentialInverseKinematicsAction):
    """Differential IK that holds each command's joint solution across the physics steps of one control step."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._ik_pending = True
        # Keep carried tissue inside the explicitly allocated MPM work volume.
        offset = torch.tensor(env.cfg.berry_position, device=self.device)
        # All local grids must cover the commanded workspace, including edge berries.
        # Several berries of one species stay inside the explicit grid at full reach, which the sort mode needs
        # to bring a berry to the reject dish.
        reach = 0.065 if env.cfg.berry == "all" else 0.09
        upper_y = 0.205 if env.cfg.background == "ebc" else reach
        self._lower = offset + torch.tensor([-reach, -reach, 0.005], device=self.device)
        self._upper = offset + torch.tensor([reach, upper_y, 0.18], device=self.device)

    def process_actions(self, actions):
        if not torch.isfinite(actions).all():
            raise ValueError("Nonfinite IK command")
        pos, _ = self._compute_frame_pose()
        limited = actions.clone()
        limited[:, :3] = (pos + actions[:, :3]).clamp(self._lower, self._upper) - pos
        super().process_actions(limited)
        self._ik_pending = True

    def apply_actions(self):
        # The teleop pose target changes at 30 Hz. Hold its joint solution while
        # the 120 Hz drives and two-way tissue contact continue to run.
        if self._ik_pending:
            super().apply_actions()
            self._ik_pending = False


class BerryGraspAction(ActionTerm):
    """Map [-1, 1] continuously to total finger aperture [0, 0.08 m], and grasp the tissue the fingers hold.

    The fingers hold an aperture while they are commanded close to their current opening, short of fully open, and
    are still. With the implicit tissue solver, the tissue they press is then clamped to them (see
    :class:`~..physics.grasp_implicit_mpm.SolverGraspImplicitMPM`). Moving the fingers releases it: closing
    compresses the tissue through contact alone, and opening lets go, where two clamped fingers would crush or tear
    it without limit.
    """

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
        opening = self._asset.data.joint_pos.torch[:, self._joint_ids]
        speed = self._asset.data.joint_vel.torch[:, self._joint_ids]
        holding = (
            ((self._target - opening).abs() <= HOLD_TOLERANCE)
            & (self._target < HOLD_MAX_OPENING)
            & (speed.abs() <= HOLD_MAX_SPEED)
        )
        set_grasping(wp.from_torch(holding.all().to(torch.int32).reshape(1)))

    def apply_actions(self):
        self._asset.set_joint_position_target(self._target, joint_ids=self._joint_ids)

    def reset(self, env_ids=None):
        self._raw[env_ids] = 1
        self._target[env_ids] = 0.04


@configclass
class BerryGraspActionCfg(ActionTermCfg):
    class_type: type = BerryGraspAction
