# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-local collision-aware arm control for the fixed-base G1."""

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass


@configclass
class G1CollisionAwareIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Bound DLS updates using arm-sphere distances to a torso box.

    The geometry protects the distal upper arm, forearm, wrist and palm against the
    torso. Shoulder attachment geometry and inter-arm collisions are excluded.
    This is a kinematic target guard, not a replacement for physical contact response.
    """

    class_type: str = "{DIR}.collision_ik:G1CollisionAwareIKAction"
    side: str = "left"
    torso_body: str = "torso_link"
    torso_center: tuple[float, float, float] = (0.0075, 0.0, 0.2388)
    torso_half_extents: tuple[float, float, float] = (0.077, 0.108, 0.249)
    """Conservative bounds of the source torso visuals in the torso frame [m]."""
    minimum_clearance: float = 0.005
    """Required sphere-to-torso surface clearance for initially safe targets [m]."""
    max_joint_step: float = 0.015
    """Maximum target offset from measured joint position per physics update [rad]."""
    recovery_step: float = 0.001
    """Maximum requested outward correction per distance constraint per update [m]."""
    projection_iterations: int = 12
    backtrack_steps: int = 5
