# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various action terms that can be used in the environment."""

import lazy_loader as lazy

from .actions_cfg import *  # noqa: F401, F403

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "binary_joint_actions": ["BinaryJointAction", "BinaryJointPositionAction", "BinaryJointVelocityAction", "AbsBinaryJointPositionAction"],
        "joint_actions": ["JointAction", "JointPositionAction", "RelativeJointPositionAction", "JointVelocityAction", "JointEffortAction"],
        "joint_actions_to_limits": ["JointPositionToLimitsAction", "EMAJointPositionToLimitsAction"],
        "non_holonomic_actions": ["NonHolonomicAction"],
        "surface_gripper_actions": ["SurfaceGripperBinaryAction"],
        "task_space_actions": ["DifferentialInverseKinematicsAction", "OperationalSpaceControllerAction"],
    },
)
