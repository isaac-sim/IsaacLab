# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the SO-101 full-pose SE3 IK action term.

This module is pure data: it must stay importable without Kit so ``load_cfg_from_registry`` can
construct the env cfg before deciding whether to launch the simulator (see
``test/core/test_env_cfg_no_forbidden_imports.py``).  The live term lives in
:mod:`.pose_ik_action_term` because subclassing
:class:`~isaaclab.envs.mdp.actions.task_space_actions.DifferentialInverseKinematicsAction` forces an
eager ``from pxr import UsdPhysics`` import.  We therefore reference the term lazily via a
``"{DIR}.pose_ik_action_term:SO101PoseIKAction"`` string ``class_type`` (resolved by
:meth:`cfg.validate` after Kit has launched), mirroring the base
:class:`~isaaclab.envs.mdp.actions.actions_cfg.DifferentialInverseKinematicsActionCfg`.

.. note::
    :attr:`~isaaclab.envs.mdp.actions.actions_cfg.DifferentialInverseKinematicsActionCfg.clip`
    is not supported for this subclass.  The base clips by action-dim axis using joint-name
    keys, but this term's action layout is the task-space pose ``[pos_xyz, quat_xyzw]``, which
    does not map to joint names.  Setting ``clip`` raises :exc:`NotImplementedError`.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from .pose_ik_controller import SO101PoseIKControllerCfg

if TYPE_CHECKING:
    from .pose_ik_action_term import SO101PoseIKAction


@configclass
class SO101PoseIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for :class:`~.pose_ik_action_term.SO101PoseIKAction`."""

    class_type: type[SO101PoseIKAction] | str = "{DIR}.pose_ik_action_term:SO101PoseIKAction"
    controller: SO101PoseIKControllerCfg = MISSING
