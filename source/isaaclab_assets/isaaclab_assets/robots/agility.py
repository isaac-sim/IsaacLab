# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

LEG_JOINT_NAMES = [
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_hip_pitch",
    ".*_knee",
    ".*_toe_a",
    ".*_toe_b",
]

ARM_JOINT_NAMES = [".*_arm_.*"]

# On MJWarp these ten joints diverge at the armature the USD authors. With the damping the asset
# carries (c = 57.3) and the velocity tasks' substep (h = sim.dt / num_substeps = 0.0025 s), the
# observed threshold is c*h/I > 2: wrist_yaw at I = 0.01822 [kg m^2] gives 7.86 and grew 6.9x per
# substep, the other eight at 0.05228 give 2.74, and every joint at or above I = 0.0716 was stable.
# 0.10 leaves margin. Re-measure if sim.dt, num_substeps or the asset's damping change.
# Together these cover exactly ``LEG_JOINT_NAMES + ARM_JOINT_NAMES``.
_LOW_ARMATURE_JOINT_NAMES = [".*_arm_wrist_.*", ".*_toe_a", ".*_toe_b"]
_STABLE_ARMATURE_JOINT_NAMES = [
    ".*_hip_roll",
    ".*_hip_yaw",
    ".*_hip_pitch",
    ".*_knee",
    ".*_arm_shoulder_.*",
    ".*_arm_elbow",
]
_MIN_STABLE_ARMATURE = 0.10


DIGIT_V4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
        activate_contact_sensors=True,
        # digit_v4.usd applies CollisionAPI to 32 prims, every one a decoration mesh on a
        # RealSense camera mount -- glass, USB-C, case halves. They are 32 of the robot's 55
        # collision shapes while the arms, hips and rods carry none, and produced 3e7 N contact
        # forces on bodies 1.4 m above the ground. Colliders on a camera's glass are an authoring
        # error on either backend.
        collision_props={"/.*camera_mount/.*/Visual/.*": [UsdPhysicsCollisionCfg(collision_enabled=False)]},
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs_arms": ImplicitActuatorCfg(
            joint_names_expr=_STABLE_ARMATURE_JOINT_NAMES,
            stiffness=None,
            damping=None,
        ),
        "low_armature": ImplicitActuatorCfg(
            joint_names_expr=_LOW_ARMATURE_JOINT_NAMES,
            stiffness=None,
            damping=None,
            armature=_MIN_STABLE_ARMATURE,
        ),
    },
)


DIGIT_V4_ISAACSIM_JOINT_ORDER = (
    "left_leg_hip_roll",
    "left_arm_shoulder_pitch",
    "right_leg_hip_roll",
    "right_arm_shoulder_pitch",
    "left_leg_hip_yaw",
    "left_arm_shoulder_roll",
    "right_leg_hip_yaw",
    "right_arm_shoulder_roll",
    "left_leg_hip_pitch",
    "left_arm_shoulder_yaw",
    "right_leg_hip_pitch",
    "right_arm_shoulder_yaw",
    "left_leg_knee",
    "left_leg_achilles_rod:0",
    "left_leg_achilles_rod:1",
    "left_leg_achilles_rod:2",
    "left_arm_elbow",
    "right_leg_achilles_rod:0",
    "right_leg_achilles_rod:1",
    "right_leg_achilles_rod:2",
    "right_leg_knee",
    "right_arm_elbow",
    "left_leg_tarsus",
    "left_arm_wrist_roll",
    "right_leg_tarsus",
    "right_arm_wrist_roll",
    "left_leg_toe_a",
    "left_leg_toe_b",
    "left_leg_toe_pitch",
    "left_arm_wrist_pitch",
    "right_leg_toe_a",
    "right_leg_toe_b",
    "right_leg_toe_pitch",
    "right_arm_wrist_pitch",
    "left_leg_toe_a_rod:0",
    "left_leg_toe_a_rod:1",
    "left_leg_toe_a_rod:2",
    "left_leg_toe_b_rod:0",
    "left_leg_toe_b_rod:1",
    "left_leg_toe_b_rod:2",
    "left_leg_toe_roll",
    "left_arm_wrist_yaw",
    "right_leg_toe_a_rod:0",
    "right_leg_toe_a_rod:1",
    "right_leg_toe_a_rod:2",
    "right_leg_toe_b_rod:0",
    "right_leg_toe_b_rod:1",
    "right_leg_toe_b_rod:2",
    "right_leg_toe_roll",
    "right_arm_wrist_yaw",
)
"""Joint order recorded from Isaac Sim PhysX 6.2 for legacy checkpoints."""


DIGIT_V4_ISAACSIM_BODY_ORDER = (
    "torso_base",
    "left_leg_hip_roll",
    "left_arm_shoulder_pitch",
    "right_leg_hip_roll",
    "right_arm_shoulder_pitch",
    "forward_chest_camera_mount",
    "backward_pelvis_camera_mount",
    "forward_pelvis_camera_mount",
    "downward_pelvis_camera_mount",
    "left_leg_hip_yaw",
    "left_arm_shoulder_roll",
    "right_leg_hip_yaw",
    "right_arm_shoulder_roll",
    "left_leg_hip_pitch",
    "left_arm_shoulder_yaw",
    "right_leg_hip_pitch",
    "right_arm_shoulder_yaw",
    "left_leg_knee",
    "left_leg_achilles_rod",
    "left_arm_elbow",
    "right_leg_achilles_rod",
    "right_leg_knee",
    "right_arm_elbow",
    "left_leg_tarsus",
    "left_arm_wrist_roll",
    "right_leg_tarsus",
    "right_arm_wrist_roll",
    "left_leg_toe_a",
    "left_leg_toe_b",
    "left_leg_toe_pitch",
    "left_arm_wrist_pitch",
    "right_leg_toe_a",
    "right_leg_toe_b",
    "right_leg_toe_pitch",
    "right_arm_wrist_pitch",
    "left_leg_toe_a_rod",
    "left_leg_toe_b_rod",
    "left_leg_toe_roll",
    "left_arm_wrist_yaw",
    "right_leg_toe_a_rod",
    "right_leg_toe_b_rod",
    "right_leg_toe_roll",
    "right_arm_wrist_yaw",
)
"""Body order recorded from Isaac Sim PhysX 6.2 for legacy checkpoints."""
