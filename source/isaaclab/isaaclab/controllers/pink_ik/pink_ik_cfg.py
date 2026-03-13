# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Pink IK controller."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .pink_task_cfg import PinkIKTaskCfg

if TYPE_CHECKING:
    from pink.tasks import Task


@configclass
class PinkIKControllerCfg:
    """Configuration settings for the Pink IK Controller.

    The Pink IK controller can be found at: https://github.com/stephane-caron/pink
    """

    usd_path: str | None = None
    """Path to the robot's USD file. When set and ``urdf_path`` is None, the controller will automatically
    convert the USD to URDF at runtime using ``convert_usd_to_urdf``. Requires Isaac Sim at runtime.
    """

    urdf_output_dir: str | None = None
    """Output directory for the USD-to-URDF conversion. Only used when ``usd_path`` is set and
    ``urdf_path`` is None. Defaults to ``tempfile.gettempdir()`` if not provided.
    """

    urdf_path: str | None = None
    """Path to the robot's URDF file. This file is used by Pinocchio's ``robot_wrapper.BuildFromURDF``
    to load the robot model. If not provided, the URDF is generated from ``usd_path`` at runtime.
    """

    mesh_path: str | None = None
    """Path to the mesh files associated with the robot. These files are also loaded by Pinocchio's
    ``robot_wrapper.BuildFromURDF``.
    """

    num_hand_joints: int = 0
    """The number of hand joints in the robot.

    The action space for the controller contains the ``pose_dim(7) * num_controlled_frames + num_hand_joints``.
    The last ``num_hand_joints`` values of the action are the hand joint angles.
    """

    variable_input_tasks: list[Task | PinkIKTaskCfg] = field(default_factory=list)
    """A list of tasks for the Pink IK controller.

    These tasks are controllable by the environment action.

    These tasks can be used to control the pose of a frame or the angles of joints.
    For more details, visit: https://github.com/stephane-caron/pink
    """

    fixed_input_tasks: list[Task | PinkIKTaskCfg] = field(default_factory=list)
    """
    A list of tasks for the Pink IK controller. These tasks are fixed and not controllable by the env action.

    These tasks can be used to fix the pose of a frame or the angles of joints to a desired configuration.
    For more details, visit: https://github.com/stephane-caron/pink
    """

    joint_names: list[str] | None = None
    """A list of joint names in the USD asset controlled by the Pink IK controller.

    This is required because the joint naming conventions differ between USD and URDF files. This value is
    currently designed to be automatically populated by the action term in a manager based environment.
    """

    all_joint_names: list[str] | None = None
    """A list of joint names in the USD asset.

    This is required because the joint naming conventions differ between USD and URDF files. This value is
    currently designed to be automatically populated by the action term in a manager based environment.
    """

    articulation_name: str = "robot"
    """The name of the articulation USD asset in the scene."""

    base_link_name: str = "base_link"
    """The name of the base link in the USD asset."""

    show_ik_warnings: bool = True
    """Show warning if IK solver fails to find a solution."""

    fail_on_joint_limit_violation: bool = True
    """Whether to fail on joint limit violation.

    If True, the Pink IK solver will fail and raise an error if any joint limit is violated during optimization.
    The PinkIKController will handle the error by setting the last joint positions.

    If False, the solver will ignore joint limit violations and return the closest solution found.
    """

    xr_enabled: bool = False
    """If True, the Pink IK controller will send information to the XRVisualization."""

    qp_solver: str = "daqp"
    """The quadratic programming solver to use for inverse kinematics.

    Common solvers include "daqp", "quadprog", "osqp", but only "daqp" and "quadprog" are installed dependencies.
    The solver must be installed and available in the environment.
    """
