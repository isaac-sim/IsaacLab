# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-position reach configuration for the OneRobotics A1 right arm."""

import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

from isaaclab_assets import ONEROBOTICS_A1_UNIMANUAL_CFG  # isort: skip

from .. import mdp as a1_mdp

_A1_JOINT_PATTERNS = [f".*joint{index}.*" for index in range(1, 8)]
_A1_END_EFFECTOR_PATTERN = ".*Link7.*"

# The common Reach table occupies x=[0.05, 0.95], y=[-0.65, 0.65], with its top at z=0.
# The canonical A1 base mesh extends 0.01575 m below base_link. Mounting at x=0.1 and
# rotating by pi about +Z keeps both the base and the home-pose arm over the tabletop.
_A1_TABLE_MOUNT_POSITION = (0.1, 0.0, 0.01575)
_A1_TABLE_MOUNT_ROTATION = (0.0, 0.0, 1.0, 0.0)


@configclass
class OneRoboticsA1ReachEnvCfg(ReachEnvCfg):
    """Reach task for the canonical fixed-base, 7-DoF OneRobotics A1 right arm.

    Actions are joint-position offsets from the configured home pose, scaled by 0.5 rad.
    End-effector targets are generated from random in-limit joint configurations, so every
    commanded position and orientation is kinematically reachable by construction.
    """

    def __post_init__(self) -> None:
        super().__post_init__()

        # The source integration is validated with Isaac Sim PhysX; do not expose unvalidated backends.
        self.sim.physics = self.sim.physics.isaacsim_physx

        self.scene.robot = ONEROBOTICS_A1_UNIMANUAL_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=ONEROBOTICS_A1_UNIMANUAL_CFG.init_state.replace(
                pos=_A1_TABLE_MOUNT_POSITION,
                rot=_A1_TABLE_MOUNT_ROTATION,
            ),
        )
        self.sim.default_visualizer_cfg.eye = (1.35, -1.25, 0.85)
        self.sim.default_visualizer_cfg.lookat = (0.42, 0.0, 0.22)

        # Match the confirmed A1 control stack: 200 Hz implicit servo and 50 Hz actions.
        self.sim.dt = 1.0 / 200.0
        self.decimation = 4
        self.sim.render_interval = self.decimation

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [_A1_END_EFFECTOR_PATTERN]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [_A1_END_EFFECTOR_PATTERN]
        self.rewards.joint_vel.params["asset_cfg"].joint_names = _A1_JOINT_PATTERNS
        self.rewards.joint_vel.params["asset_cfg"].preserve_order = True

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_A1_JOINT_PATTERNS,
            scale=0.5,
            use_default_offset=True,
            preserve_order=True,
        )

        self.commands.ee_pose = a1_mdp.FkReachablePoseCommandCfg(
            asset_name="robot",
            body_name=_A1_END_EFFECTOR_PATTERN,
            chain=a1_mdp.A1_RIGHT_CHAIN,
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
