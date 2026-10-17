# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position reach configuration for the OneRobotics A1 bimanual robot."""

import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from isaaclab_assets import ONEROBOTICS_A1_BIMANUAL_CFG  # isort: skip

from .reach_env_cfg import BimanualReachEnvCfg

_A1_RIGHT_JOINT_PATTERNS = [f".*joint_r{index}.*" for index in range(1, 8)]
_A1_LEFT_JOINT_PATTERNS = [f".*joint_l{index}.*" for index in range(1, 8)]
_A1_JOINT_PATTERNS = _A1_RIGHT_JOINT_PATTERNS + _A1_LEFT_JOINT_PATTERNS
_A1_RIGHT_END_EFFECTOR_PATTERN = ".*Link_r7.*"
_A1_LEFT_END_EFFECTOR_PATTERN = ".*Link_l7.*"


@configclass
class OneRoboticsA1BimanualReachEnvCfg(BimanualReachEnvCfg):
    """Reach task for the fixed-base, 14-DoF OneRobotics A1 bimanual source model.

    One articulation exposes a 14-D right-then-left joint-position action. Both
    standard pose commands sample independent Cartesian targets in the robot
    base frame. The source model retains its fixed shoulder mounts and joint axes.
    Reset is centered on the source model's all-zero pose; it is not labeled as a
    physical hardware home pose.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # The source integration is validated with Isaac Sim PhysX; do not expose unvalidated backends.
        self.sim.physics = self.sim.physics.isaacsim_physx
        self.scene.robot = ONEROBOTICS_A1_BIMANUAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {"joint_[rl][1-7]": 0.0}
        # The source URDF retains two fixed shoulder joints. PhysX scene replication
        # drops one cloned fixed-root body relation and destabilizes later environments.
        self.scene.replicate_physics = False

        # Match the A1 control stack: 200 Hz implicit servo and 50 Hz policy actions.
        self.sim.dt = 1.0 / 200.0
        self.decimation = 4
        self.sim.render_interval = self.decimation

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = _A1_JOINT_PATTERNS
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = _A1_JOINT_PATTERNS
        self.rewards.joint_vel.params["asset_cfg"].joint_names = _A1_JOINT_PATTERNS
        self.rewards.joint_vel.params["asset_cfg"].preserve_order = True

        self.rewards.right_end_effector_position_tracking.params["asset_cfg"].body_names = [
            _A1_RIGHT_END_EFFECTOR_PATTERN
        ]
        self.rewards.right_end_effector_orientation_tracking.params["asset_cfg"].body_names = [
            _A1_RIGHT_END_EFFECTOR_PATTERN
        ]
        self.rewards.left_end_effector_position_tracking.params["asset_cfg"].body_names = [
            _A1_LEFT_END_EFFECTOR_PATTERN
        ]
        self.rewards.left_end_effector_orientation_tracking.params["asset_cfg"].body_names = [
            _A1_LEFT_END_EFFECTOR_PATTERN
        ]

        self.actions.right_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_A1_RIGHT_JOINT_PATTERNS,
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
        )
        self.actions.left_arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_A1_LEFT_JOINT_PATTERNS,
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
        )

        # Both Cartesian workspaces lie above the table and are expressed in the robot base frame.
        self.commands.right_ee_pose.body_name = _A1_RIGHT_END_EFFECTOR_PATTERN
        self.commands.right_ee_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.16, 0.28),
            pos_y=(-0.60, -0.48),
            pos_z=(0.25, 0.37),
            roll=(-1.98, -1.58),
            pitch=(0.69, 1.09),
            yaw=(-2.73, -2.33),
        )
        self.commands.left_ee_pose.body_name = _A1_LEFT_END_EFFECTOR_PATTERN
        self.commands.left_ee_pose.ranges = mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.16, 0.28),
            pos_y=(0.48, 0.60),
            pos_z=(0.25, 0.37),
            roll=(-1.98, -1.58),
            pitch=(-1.09, -0.69),
            yaw=(-0.82, -0.42),
        )
        self.commands.right_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/Command/right_goal_pose"
        self.commands.right_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/Command/right_body_pose"
        self.commands.left_ee_pose.goal_pose_visualizer_cfg.prim_path = "/Visuals/Command/left_goal_pose"
        self.commands.left_ee_pose.current_pose_visualizer_cfg.prim_path = "/Visuals/Command/left_body_pose"
