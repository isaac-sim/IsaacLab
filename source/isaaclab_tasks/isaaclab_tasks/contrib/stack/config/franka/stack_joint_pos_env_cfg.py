# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.stack import mdp
from isaaclab_tasks.contrib.stack.stack_env_cfg import (
    StackEnvCfg,
    StackEventCfg,
    apply_default_semantics,
    make_ee_frame_cfg,
)

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip

# Default arm + gripper joint pose
_FRANKA_STACK_IK_REL_INIT_JOINT_POS: dict[str, float] = {
    "panda_joint1": 0.0444,
    "panda_joint2": -0.1894,
    "panda_joint3": -0.1107,
    "panda_joint4": -2.5148,
    "panda_joint5": 0.0044,
    "panda_joint6": 2.3775,
    "panda_joint7": 0.6952,
    "panda_finger_joint.*": 0.0400,
}


@configclass
class FrankaCubeStackEnvCfg(StackEnvCfg):
    """Configuration for the Franka Cube Stack Environment.

    Uses the robot-neutral stack scaffolding (cubes, semantics, ee-frame builder, reset events)
    from :mod:`~isaaclab_tasks.contrib.stack.stack_env_cfg`; the default cube transforms and
    workspace there match the Franka layout, so only the robot, actions/gripper, and end-effector
    frame prim paths are set here.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set robot-neutral reset events (default cube workspace matches the Franka layout).
        self.events = StackEventCfg()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ArticulationCfg.InitialStateCfg(joint_pos=_FRANKA_STACK_IK_REL_INIT_JOINT_POS),
        )

        # Tag the table / ground / robot semantic classes.
        apply_default_semantics(self.scene)

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # utilities for gripper status check
        self.gripper_joint_names = ["panda_finger_.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.005

        # End-effector frame (the shared cubes/spawns come from the base scene).
        self.scene.ee_frame = make_ee_frame_cfg(
            base_prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            target_specs=[
                ("{ENV_REGEX_NS}/Robot/panda_hand", "end_effector", (0.0, 0.0, 0.1034)),
                ("{ENV_REGEX_NS}/Robot/panda_rightfinger", "tool_rightfinger", (0.0, 0.0, 0.046)),
                ("{ENV_REGEX_NS}/Robot/panda_leftfinger", "tool_leftfinger", (0.0, 0.0, 0.046)),
            ],
            marker_scale=(0.1, 0.1, 0.1),
        )
