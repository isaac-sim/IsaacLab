# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime action term for the SO-101 full-pose SE3 IK controller.

This module hosts the live :class:`SO101PoseIKAction` term.  It is intentionally **separate**
from :mod:`.pose_ik_action` (which holds the pure-data :class:`~.pose_ik_action.SO101PoseIKActionCfg`)
because subclassing :class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction`
forces an eager ``from pxr import UsdPhysics`` at import time.  The env cfg must be constructable
without Kit (see ``test/core/test_env_cfg_no_forbidden_imports.py``), so the cfg only references
this term lazily via a ``"{DIR}.pose_ik_action_term:SO101PoseIKAction"`` string ``class_type`` that
:meth:`cfg.validate` resolves after Kit has launched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

from .pose_ik_controller import SO101PoseIKController

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .pose_ik_action import SO101PoseIKActionCfg


class SO101PoseIKAction(DifferentialInverseKinematicsAction):
    """IK action term that uses the SO-101 full-pose SE3 controller."""

    cfg: SO101PoseIKActionCfg

    def __init__(self, cfg: SO101PoseIKActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Swap the base differential-IK controller for the SO-101 one. The two share the same 7D
        # ``[pos, quat_xyzw]`` action layout, so the base-allocated buffers/scale stay valid; the
        # SO-101 controller only adds the orientation joint mask on top of the core adaptive-DLS +
        # orientation-weighting + JLA controller. Null-space joint-limit injection now comes from
        # the base term (gated on ``joint_limit_avoidance_gain > 0``), so it is not repeated here.
        self._ik_controller = SO101PoseIKController(cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device)
        # Restrict the orientation task to a subset of joints if configured (e.g. the SO-101 wrist),
        # so the other joints (shoulder_pan, ...) serve position only and the base does not swing to
        # track orientation. Resolve the names against ``self._joint_names`` (asset-ordered, matching
        # the Jacobian columns the controller receives) so the pushed mask is order-proof.
        ori_names = self.cfg.controller.orientation_joint_names
        if ori_names is not None:
            missing = set(ori_names) - set(self._joint_names)
            if missing:
                raise ValueError(
                    f"orientation_joint_names {sorted(missing)} are not among this action's joints {self._joint_names}."
                )
            ori_mask = torch.tensor(
                [1.0 if name in ori_names else 0.0 for name in self._joint_names], device=self.device
            )
            self._ik_controller.set_orientation_joint_mask(ori_mask)
        # Action clipping is not supported for this term: the base clips by action-dim axis using
        # joint-name keys, but this term's action layout is the task-space pose
        # [pos_xyz, quat_xyzw], which does not map to joint names. Refuse rather than silently
        # clip the wrong axis.
        if self.cfg.clip is not None:
            raise NotImplementedError(
                "clip is not supported for SO101PoseIKAction (task-space [pos, quat] action "
                "does not map to joint-name clip keys)."
            )
