# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Retargeters for mapping input device data to robot commands."""

from .humanoid.fourier.gr1t2_retargeter import GR1T2Retargeter, GR1T2RetargeterCfg
from .humanoid.unitree.g1_lower_body_standing import G1LowerBodyStandingRetargeter, G1LowerBodyStandingRetargeterCfg
from .humanoid.unitree.g1_motion_controller_locomotion import (
    G1LowerBodyStandingMotionControllerRetargeter,
    G1LowerBodyStandingMotionControllerRetargeterCfg,
)
from .humanoid.unitree.inspire.g1_upper_body_retargeter import UnitreeG1Retargeter, UnitreeG1RetargeterCfg
from .humanoid.unitree.trihand.g1_upper_body_motion_ctrl_gripper import (
    G1TriHandUpperBodyMotionControllerGripperRetargeter,
    G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
)
from .humanoid.unitree.trihand.g1_upper_body_motion_ctrl_retargeter import (
    G1TriHandUpperBodyMotionControllerRetargeter,
    G1TriHandUpperBodyMotionControllerRetargeterCfg,
)
from .humanoid.unitree.trihand.g1_upper_body_retargeter import (
    G1TriHandUpperBodyRetargeter,
    G1TriHandUpperBodyRetargeterCfg,
)
from .manipulator.gripper_retargeter import GripperRetargeter, GripperRetargeterCfg
from .manipulator.se3_abs_retargeter import Se3AbsRetargeter, Se3AbsRetargeterCfg
from .manipulator.se3_rel_retargeter import Se3RelRetargeter, Se3RelRetargeterCfg
