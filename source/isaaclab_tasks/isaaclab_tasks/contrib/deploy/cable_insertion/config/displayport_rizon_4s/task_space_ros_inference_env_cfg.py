# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Task-space ROS inference configuration for DisplayPort insertion with the Flexiv Rizon 4S.

Inherits from the task-space training config and adds the Isaac Manipulator metadata
fields plus a fixed, deterministic socket/plug setup for on-robot inference.
"""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)

from .task_space_env_cfg import _ACTION_SCALE, Rizon4sTaskSpaceDisplayportInsertionEnvCfg

# Deployment socket/plug station pose. CALIBRATE: re-measure for the real DisplayPort
# station (see ros_inference_env_cfg.py for the joint-space equivalent).
_HUBBLE_GEOMETRY_POS = (0.475, 0.125, 0.06)
_HUBBLE_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_HUBBLE_PLUG_CLEARANCE_Z = 0.068

_HUBBLE_SOCKET_ROOT = compute_socket_root(_HUBBLE_GEOMETRY_POS, _HUBBLE_SOCKET_ROT)
_HUBBLE_PLUG_ROOT, _HUBBLE_PLUG_ROT = compute_plug_pose(
    _HUBBLE_GEOMETRY_POS,
    _HUBBLE_SOCKET_ROT,
    z_clearance=_HUBBLE_PLUG_CLEARANCE_Z,
)


@configclass
class Rizon4sTaskSpaceDisplayportInsertionROSInferenceEnvCfg(Rizon4sTaskSpaceDisplayportInsertionEnvCfg):
    """Task-space ROS / Isaac Manipulator inference config.

    Exposes the observation/action metadata Isaac Manipulator needs for on-robot
    inference with task-space (OSC) control and 6D-rotation observations, and pins the
    plug and socket to a fixed deployment pose.
    """

    def __post_init__(self):
        super().__post_init__()

        # Metadata consumed by Isaac Manipulator for on-robot inference.
        self.obs_order = ["eef_pos", "eef_rot_6d", "socket_kp_pos", "socket_kp_rot_6d"]
        self.policy_action_space = "task"
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        # 6-DOF task-space action (3 position + 3 axis-angle rotation).
        self.action_space = 6
        # Critic: 7 jpos + 7 jvel + 3 socket_pos + 6 socket_rot6d + 3 plug_pos + 6 plug_rot6d = 32.
        self.state_space = 32
        # Actor: 3 eef_pos + 6 eef_rot6d + 3 socket_pos + 6 socket_rot6d = 18.
        self.observation_space = 18

        self.action_scale = [_ACTION_SCALE] * self.action_space

        # Flexiv Rizon 4S vertical (table-top) mount. Home joint pose seeds the grasp IK.
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(32.44),
            "joint2": math.radians(-16.71),
            "joint3": math.radians(-5.69),
            "joint4": math.radians(128.38),
            "joint5": math.radians(6.74),
            "joint6": math.radians(55.95),
            "joint7": math.radians(111.54),
        }

        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_SOCKET_ROOT,
            rot=_HUBBLE_SOCKET_ROT,
        )
        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_PLUG_ROOT,
            rot=_HUBBLE_PLUG_ROT,
        )

        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        # Fixed asset parameters for ROS inference (geometry center, not USD root).
        self.fixed_asset_init_pos_center = list(_HUBBLE_GEOMETRY_POS)

        pose_range = self.events.randomize_socket_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],
            pose_range["y"][1],
            pose_range["z"][1],
        ]
        # CALIBRATE: euler orientation of the DP socket at the station.
        self.fixed_asset_init_orn_deg = [0.0, 0.0, 0.0]
        self.fixed_asset_init_orn_deg_range = [
            math.degrees(pose_range["roll"][1]),
            math.degrees(pose_range["pitch"][1]),
            math.degrees(pose_range["yaw"][1]),
        ]

        # No socket-position observation noise in the production configuration.
        self.fixed_asset_pos_obs_noise_level = [0.0, 0.0, 0.0]
