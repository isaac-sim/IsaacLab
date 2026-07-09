# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the camera-mount variant of the OpenArm bimanual robot.

This is a SEPARATE robot config from `OPENARM_BI_CFG` (openarm.py) -- it does not modify or
replace anything used by existing tasks. It exists because the real robot used for VLA data
collection has a camera-mount bracket bolted onto each wrist (visible in wrist/body camera
frames), while every existing sim task uses the stock OpenArm USD with no such bracket -- a
visual sim2real gap for camera-conditioned policies (the sim robot's own arm doesn't look like
what the real camera sees).

Source: a SolidWorks-exported URDF (source/isaaclab_assets/data/openarm_bimanual_cammount/urdf/
onlyOpenArm.urdf) with the camera-mount bracket baked into the wrist-link mesh geometry. That
export used entirely different joint names (L_link{i}_joint, R_link{i}_joint, gripper_LL/LR/RL/
RR_link_joint) and had no end-effector/TCP frame and no gripper mimic relationship, so a copy of
it was edited (in place, in the copy under source/isaaclab_assets/data/) to be a structural
drop-in for the existing naming convention before conversion to USD:
  - L_link{i}_joint  -> openarm_left_joint{i}   (i=1..7)
  - R_link{i}_joint  -> openarm_right_joint{i}  (i=1..7)
  - gripper_LR_link_joint -> openarm_left_finger_joint1   (primary/actuated, limits unchanged [0,0.044])
  - gripper_LL_link_joint -> openarm_left_finger_joint2   (axis + limits sign-flipped to match
    finger_joint1's [0,0.044] convention, then a <mimic joint="openarm_left_finger_joint1"
    multiplier="1" offset="0"/> tag added -- matches the real/stock robot's finger2-mimics-
    finger1 setup, and lets every existing gripper action term's open/close regex, which
    commands BOTH finger joints to the identical numeric value, work unchanged)
  - gripper_RR_link_joint / gripper_RL_link_joint -> openarm_right_finger_joint1/2, same pattern
  - Added fixed openarm_left_ee_tcp / openarm_right_ee_tcp frames (offset from L_link7/R_link7
    -- first estimated then confirmed/corrected against the actual finger geometry in Isaac Sim;
    see the conversion validation notes) since this URDF's arms otherwise end at the wrist link
    with no dedicated TCP frame, and every IK-based action term / camera mount needs one.

Link names were NOT renamed (still L_link1..7, R_link1..7, chest_link, base_link, gripper_*_link)
-- only joint names needed to match for actuator/action-term regexes to work. Any task config
that references a LINK name from the stock robot (e.g. "openarm_left_link1", "openarm_body_link")
needs the corresponding new-robot link name substituted in (see openarm/pickup_ik_abs_cammount_
env_cfg.py's own docstring for the exact mapping used there).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

OPENARM_BI_CAMMOUNT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/openarm_bimanual_cammount/openarm_bimanual_cammount.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "openarm_left_joint.*": 0.0,
            "openarm_right_joint.*": 0.0,
            "openarm_left_finger_joint.*": 0.0,
            "openarm_right_finger_joint.*": 0.0,
        },
    ),
    # Actuator gains copied from OPENARM_BI_CFG (openarm.py) -- same physical motors, this is a
    # different mesh/mount on the same underlying arm, not a different robot.
    actuators={
        "openarm_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_joint[1-7]",
                "openarm_right_joint[1-7]",
            ],
            velocity_limit_sim={
                "openarm_left_joint[1-2]": 2.175,
                "openarm_right_joint[1-2]": 2.175,
                "openarm_left_joint[3-4]": 2.175,
                "openarm_right_joint[3-4]": 2.175,
                "openarm_left_joint[5-7]": 2.61,
                "openarm_right_joint[5-7]": 2.61,
            },
            effort_limit_sim={
                "openarm_left_joint[1-2]": 40.0,
                "openarm_right_joint[1-2]": 40.0,
                "openarm_left_joint[3-4]": 27.0,
                "openarm_right_joint[3-4]": 27.0,
                "openarm_left_joint[5-7]": 7.0,
                "openarm_right_joint[5-7]": 7.0,
            },
            stiffness=80.0,
            damping=4.0,
        ),
        "openarm_gripper": ImplicitActuatorCfg(
            joint_names_expr=[
                "openarm_left_finger_joint.*",
                "openarm_right_finger_joint.*",
            ],
            velocity_limit_sim=0.2,
            effort_limit_sim=333.33,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the camera-mount variant of the OpenArm Bimanual robot."""

OPENARM_BI_CAMMOUNT_HIGH_PD_CFG = OPENARM_BI_CAMMOUNT_CFG.copy()
OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.actuators["openarm_arm"].stiffness = 400.0
OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.actuators["openarm_arm"].damping = 80.0
OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.actuators["openarm_gripper"].stiffness = 2e3
OPENARM_BI_CAMMOUNT_HIGH_PD_CFG.actuators["openarm_gripper"].damping = 1e2
"""Configuration of the camera-mount OpenArm Bimanual robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
