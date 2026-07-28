# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka robot configurations for differential-IK reach tasks."""

from isaaclab_newton.sim.schemas import MujocoJointDrivePropertiesCfg

from isaaclab_tasks.utils import preset

from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


FRANKA_PANDA_HIGH_PD_DIFF_IK_CFG = FRANKA_PANDA_HIGH_PD_CFG.copy()
FRANKA_PANDA_HIGH_PD_DIFF_IK_CFG.spawn.joint_drive_props = preset(
    default=None,
    newton_mjwarp=MujocoJointDrivePropertiesCfg(actuatorgravcomp=True),
)
"""High-PD Franka configuration with backend-specific differential-IK drive properties."""
