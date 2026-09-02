# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position reach configuration for the OneRobotics A1 bimanual robot."""

import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from isaaclab_assets import ONEROBOTICS_A1_BIMANUAL_CFG  # isort: skip

from .. import mdp as a1_mdp
from .reach_env_cfg import BimanualReachEnvCfg

_A1_RIGHT_JOINT_PATTERNS = [f".*joint_r{index}.*" for index in range(1, 8)]
_A1_LEFT_JOINT_PATTERNS = [f".*joint_l{index}.*" for index in range(1, 8)]
_A1_JOINT_PATTERNS = _A1_RIGHT_JOINT_PATTERNS + _A1_LEFT_JOINT_PATTERNS
_A1_RIGHT_END_EFFECTOR_PATTERN = ".*Link_r7.*"
_A1_LEFT_END_EFFECTOR_PATTERN = ".*Link_l7.*"


def _reachable_command(
    *,
    body_name: str,
    chain: list[a1_mdp.KinematicChainEntry],
    fixed_transform: a1_mdp.FixedTransform,
    marker_prefix: str,
) -> a1_mdp.FkReachablePoseCommandCfg:
    """Create one side's FK-reachable pose command."""
    command = a1_mdp.FkReachablePoseCommandCfg(
        asset_name="robot",
        body_name=body_name,
        chain=chain,
        fixed_transform=fixed_transform,
        joint_range_scale=0.8,
        resampling_time_range=(4.0, 4.0),
        make_quat_unique=False,
        debug_vis=True,
        position_success_threshold=0.05,
        orientation_success_threshold=0.2,
        ranges=a1_mdp.FkReachablePoseCommandCfg.Ranges(
            pos_x=(0.0, 0.0),
            pos_y=(0.0, 0.0),
            pos_z=(0.0, 0.0),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
    command.goal_pose_visualizer_cfg = command.goal_pose_visualizer_cfg.replace(
        prim_path=f"/Visuals/Command/{marker_prefix}_goal_pose"
    )
    command.current_pose_visualizer_cfg = command.current_pose_visualizer_cfg.replace(
        prim_path=f"/Visuals/Command/{marker_prefix}_body_pose"
    )
    return command


@configclass
class OneRoboticsA1BimanualReachEnvCfg(BimanualReachEnvCfg):
    """Reach task for the fixed-base, 14-DoF OneRobotics A1 bimanual source model.

    One articulation exposes a 14-D right-then-left joint-position action. Both
    command generators sample in-limit joints and use the source URDF's fixed
    shoulder mounts, joint origins, and joint axes to construct reachable poses.
    Reset is centered on the source model's all-zero pose; it is not labeled as a
    physical hardware home pose.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # The source integration is validated with Isaac Sim PhysX; do not expose unvalidated backends.
        self.sim.physics = self.sim.physics.isaacsim_physx
        self.scene.robot = ONEROBOTICS_A1_BIMANUAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = {"joint_[rl][1-7]": 0.0}

        # Match the A1 control stack: 200 Hz implicit servo and 50 Hz policy actions.
        self.sim.dt = 1.0 / 200.0
        self.decimation = 4
        self.sim.render_interval = self.decimation

        for term_name in ("joint_pos", "joint_vel"):
            asset_cfg = getattr(self.observations.policy, term_name).params["asset_cfg"]
            asset_cfg.joint_names = _A1_JOINT_PATTERNS
            asset_cfg.preserve_order = True
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

        self.commands.right_ee_pose = _reachable_command(
            body_name=_A1_RIGHT_END_EFFECTOR_PATTERN,
            chain=a1_mdp.A1_BIMANUAL_RIGHT_CHAIN,
            fixed_transform=a1_mdp.A1_BIMANUAL_RIGHT_FIXED_MOUNT,
            marker_prefix="right",
        )
        self.commands.left_ee_pose = _reachable_command(
            body_name=_A1_LEFT_END_EFFECTOR_PATTERN,
            chain=a1_mdp.A1_BIMANUAL_LEFT_CHAIN,
            fixed_transform=a1_mdp.A1_BIMANUAL_LEFT_FIXED_MOUNT,
            marker_prefix="left",
        )
