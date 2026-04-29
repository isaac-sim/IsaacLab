# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .joint_pos_env_cfg import Rizon4sGravCableInsertionEnvCfg


@configclass
class Rizon4sGravCableInsertionROSInferenceEnvCfg(Rizon4sGravCableInsertionEnvCfg):
    """Configuration for ROS inference with Flexiv Rizon 4s for cable insertion.

    This configuration:

    - Exposes variables needed for Isaac Manipulator ROS inference.
    - Overrides plug and socket initial poses for a fixed/deterministic setup.
    """

    def __post_init__(self):
        super().__post_init__()

        # Variables used by Isaac Manipulator for on-robot inference
        self.obs_order = ["arm_dof_pos", "arm_dof_vel", "socket_pos", "socket_quat"]
        self.policy_action_space = "joint"
        self.arm_joint_names = self.observations.policy.joint_pos.params["asset_cfg"].joint_names
        self.action_space = self.num_arm_joints
        # State: 7 joint pos + 7 joint vel + 3 socket pos + 4 socket quat + 3 plug pos + 4 plug quat = 28
        self.state_space = 28
        # Observation: 7 joint pos + 7 joint vel + 3 socket pos + 4 socket quat = 21
        self.observation_space = 21

        self.joint_action_scale = self.actions.arm_action.scale
        self.action_scale_joint_space = [self.joint_action_scale] * self.action_space

        # Override robot initial rotation for deterministic setup
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)

        # Override plug and socket initial poses for ROS inference (fixed, deterministic)
        self.scene.gb300_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(-0.6, -0.4, 0.1),
            rot=(0.0, 0.0, 0.0, 1.0),
        )

        self.scene.gb300_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=(-0.6, -0.4, 0.1),
            rot=(0.0, 0.0, 0.0, 1.0),
        )

        # Fixed asset parameters for ROS inference - derived from configuration
        self.fixed_asset_init_pos_center = list(self.scene.gb300_socket.init_state.pos)

        pose_range = self.events.randomize_socket_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],
            pose_range["y"][1],
            pose_range["z"][1],
        ]
        self.fixed_asset_init_orn_deg = [0.0, 0.0, 0.0]
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]

        socket_pos_noise = self.observations.policy.socket_pos.noise.noise_cfg.n_max
        self.fixed_asset_pos_obs_noise_level = [
            socket_pos_noise,
            socket_pos_noise,
            socket_pos_noise,
        ]
