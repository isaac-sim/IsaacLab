# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenArm reaching task: reach the red cube with randomized cube positions and camera observation."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from . import stack_ik_abs_visuomotor_env_cfg


@configclass
class EventCfg:
    """Randomize all cube positions on every episode reset within the robot's workspace."""

    # Reset robot joints to default configuration
    init_robot_pose = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )

    # ── Workspace strip: x=[0.15, 0.25], y=[-0.25, 0.25] ────────────────────────
    # Pad near edge (robot side) is at x=0.125.  All cubes stay within the
    # first 10 cm of the pad so the arms can comfortably reach them.
    # Formula: offset = target_actual − cube_default_pos
    #
    # To widen/narrow the strip, change the x offsets symmetrically:
    #   e.g. x=[0.15, 0.35]  →  cube_1 x=(-0.05,+0.15), cube_2 x=(-0.40,-0.20), cube_3 x=(-0.45,-0.25)
    # To shift the strip further from the robot, increase both x offsets by the same amount.

    # cube_1 (blue)  default [0.20, 0.08]  →  actual x:[0.15,0.25]  y:[-0.25,0.25]
    randomize_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),   # 0.20 + offset → [0.15, 0.25]
                "y": (-0.33, 0.17),   # 0.08 + offset → [-0.25, 0.25]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )

    # cube_2 (red)   default [0.55, 0.05]  →  actual x:[0.15,0.25]  y:[-0.25,0.25]
    randomize_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.40, -0.30),  # 0.55 + offset → [0.15, 0.25]
                "y": (-0.30, 0.20),   # 0.05 + offset → [-0.25, 0.25]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )

    # cube_3 (green) default [0.60, -0.10]  →  actual x:[0.15,0.25]  y:[-0.25,0.25]
    randomize_cube_3 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.45, -0.35),  # 0.60 + offset → [0.15, 0.25]
                "y": (-0.15, 0.35),   # -0.10 + offset → [-0.25, 0.25]
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_3"),
        },
    )


@configclass
class OpenarmReachRedCubeEnvCfg(stack_ik_abs_visuomotor_env_cfg.OpenarmCubeStackVisuomotorEnvCfg):
    """OpenArm teleoperation task: reach the red cube (cube_2).

    Features:
      - All 3 cube positions randomized each reset within robot's reachable workspace
      - Front camera + wrist camera observations (640x480 RGB, concatenate_terms=False)
      - Dual-arm IK control: LEFT arm (arm_action) + RIGHT arm (right_arm_action)
      - Workspace bounds: x=[0.15, 0.55m], y=[-0.25, 0.25m] from robot base

    Record demos (with arm switching via TAB key):
      ./isaaclab.sh -p scripts/tools/record_demos_openarm.py \\
          --task Isaac-Reach-RedCube-OpenArm-IK-Abs-v0 \\
          --dataset_file logs/demos/openarm_reach.hdf5 \\
          --enable_cameras

    Convert to LeRobot:
      conda run -n lerobot python scripts/tools/convert_hdf5_to_lerobot.py \\
          --hdf5 logs/demos/openarm_reach.hdf5 \\
          --output ~/datasets/openarm_reach \\
          --task "Reach red cube" --fps 20 \\
          --cameras front_cam wrist_cam

    Action space (flat, 14D):
      [0:6]   left arm IK delta pose (dx, dy, dz, drx, dry, drz)
      [6:7]   left gripper (±1.0)
      [7:13]  right arm IK delta pose
      [13:14] right gripper (±1.0)
    """

    def __post_init__(self):
        super().__post_init__()

        # Replace events with cube randomization + robot reset
        self.events = EventCfg()

        # Add right arm IK action (appended after left arm in action space → indices [7:14])
        self.actions.right_arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_joint[1-7]"],
            body_name="openarm_right_ee_tcp",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=True, ik_method="dls"
            ),
        )
        self.actions.right_gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["openarm_right_finger_joint.*"],
            open_command_expr={"openarm_right_finger_joint.*": 0.044},
            close_command_expr={"openarm_right_finger_joint.*": 0.0},
        )
