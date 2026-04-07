# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hardcoded deterministic inference configuration for Flexiv Rizon 4s with Grav gripper.

Edit the constants below to match your real-world setup. All randomization is
disabled and observation noise is turned off so the simulation is fully
deterministic across resets.

The IK grasp-pose solver (``set_robot_to_grasp_pose``) is kept so the robot
properly grasps the gear. With all randomization zeroed out the IK solution
is identical on every reset.

To additionally override the robot's joint positions *after* IK, set
``OVERRIDE_ROBOT_JOINT_POS`` in ``play_hardcoded.py``.
"""

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .ros_inference_env_cfg import Rizon4sGearAssemblyROSInferenceEnvCfg

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  HARDCODED VALUES — Edit these to match your desired setup                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Which gear the robot is grasping: "gear_small", "gear_medium", or "gear_large"
GEAR_TYPE = "gear_large"

# Gear base position (x, y, z) in meters, relative to env origin
GEAR_BASE_POS = (0.481, -0.073, -0.005)

# Gear base orientation as quaternion (x, y, z, w)
GEAR_BASE_ROT = (0.0, 0.0, 0.70711, -0.70711)

# Height of the grasped gear above the gear base (world-Z offset in meters).
# Training midpoint is 0.0675 from range [0.0575, 0.0775].
GEAR_Z_OFFSET = 0.0675

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  Environment configuration                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝


@configclass
class Rizon4sGearAssemblyHardcodedInferenceEnvCfg(Rizon4sGearAssemblyROSInferenceEnvCfg):
    """Deterministic inference config — hardcoded poses, no randomization, no obs noise.

    Inherits the full ROS-inference configuration and then:
      1. Fixes the gear type (no random selection).
      2. Zeros out all pose/orientation randomization ranges.
      3. Sets gear-base and individual-gear initial states from the constants above.
      4. Disables observation noise (``enable_corruption = False``).

    The IK grasp-pose solver is kept from the parent so the robot properly
    reaches and grasps the gear each reset.  Since all inputs to IK are
    deterministic the resulting joint positions are the same every time.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── 1. Fix gear type ──────────────────────────────────────────
        self.events.randomize_gear_type.params["gear_types"] = [GEAR_TYPE]

        # ── 2. Zero out all pose randomization ────────────────────────
        self.events.randomize_gears_and_base_pose.params["pose_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [0.0, 0.0],
            "roll": [0.0, 0.0],
            "pitch": [0.0, 0.0],
            "yaw": [0.0, 0.0],
        }
        self.events.randomize_gears_and_base_pose.params["gear_pos_range"] = {
            "x": [0.0, 0.0],
            "y": [0.0, 0.0],
            "z": [GEAR_Z_OFFSET, GEAR_Z_OFFSET],
        }

        # ── 3. Hardcoded gear base and gear positions ─────────────────
        self.scene.factory_gear_base.init_state = RigidObjectCfg.InitialStateCfg(
            pos=GEAR_BASE_POS,
            rot=GEAR_BASE_ROT,
        )
        for attr in ("factory_gear_small", "factory_gear_medium", "factory_gear_large"):
            getattr(self.scene, attr).init_state = RigidObjectCfg.InitialStateCfg(
                pos=GEAR_BASE_POS,
                rot=GEAR_BASE_ROT,
            )

        # ── 4. Disable observation noise ──────────────────────────────
        self.observations.policy.enable_corruption = False
