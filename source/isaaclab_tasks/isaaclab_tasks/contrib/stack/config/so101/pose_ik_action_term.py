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
        # Replace the base position/pose controller with the SO-101 full-pose one.  Re-create the
        # raw/processed action buffers and scale (action_dim is the controller's 7).
        self._ik_controller = SO101PoseIKController(cfg=self.cfg.controller, num_envs=self.num_envs, device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)
        # Action clipping is not supported for this term: the base clips by action-dim axis using
        # joint-name keys, but this term's action layout is the task-space pose
        # [pos_xyz, quat_xyzw], which does not map to joint names. Refuse rather than silently
        # clip the wrong axis.
        if self.cfg.clip is not None:
            raise NotImplementedError(
                "clip is not supported for SO101PoseIKAction (task-space [pos, quat] action "
                "does not map to joint-name clip keys)."
            )
        # Joint limits are injected lazily on the first apply (asset data is populated by then)
        # so the controller can do null-space joint-limit avoidance.
        self._limits_injected = False

    def apply_actions(self) -> None:
        if not self._limits_injected:
            # Limits are uniform across envs for the SO-101; env 0 is representative.
            limits = self._asset.data.soft_joint_pos_limits.torch[0, self._joint_ids, :]
            self._ik_controller.set_joint_pos_limits(limits[:, 0].clone(), limits[:, 1].clone())
            self._limits_injected = True
        super().apply_actions()
