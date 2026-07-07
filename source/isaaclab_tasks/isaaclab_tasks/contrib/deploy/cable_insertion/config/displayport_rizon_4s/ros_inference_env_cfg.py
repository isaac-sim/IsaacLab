# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space ROS inference configuration for DisplayPort insertion with Flexiv Rizon 4S.

Inherits from the joint-space training config and adds Isaac Manipulator
metadata fields plus deployment alignment. Mirrors the GB300
``config/rizon_4s/ros_inference_env_cfg.py``.
"""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.contrib.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)

from .joint_pos_env_cfg import Rizon4sGravDisplayportInsertionEnvCfg

# ---------------------------------------------------------------------------
# Deployment socket/plug positions
# ---------------------------------------------------------------------------
# CALIBRATE: geometry pos seeded from the training-env value (GB300 Hubble table
# mount). Re-measure the real DisplayPort fixture pose for the actual station.
# The socket rotation keeps the verified DP orientation (opening facing +Z).
_HUBBLE_GEOMETRY_POS = (0.476, 0.127, 0.07)
# _HUBBLE_GEOMETRY_POS = (0.481, -0.073, 0.071)
_HUBBLE_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_HUBBLE_PLUG_CLEARANCE_Z = 0.068

_HUBBLE_SOCKET_ROOT = compute_socket_root(_HUBBLE_GEOMETRY_POS, _HUBBLE_SOCKET_ROT)
_HUBBLE_PLUG_ROOT, _HUBBLE_PLUG_ROT = compute_plug_pose(
    _HUBBLE_GEOMETRY_POS,
    _HUBBLE_SOCKET_ROT,
    z_clearance=_HUBBLE_PLUG_CLEARANCE_Z,
)


@configclass
class Rizon4sGravDisplayportInsertionROSInferenceEnvCfg(Rizon4sGravDisplayportInsertionEnvCfg):
    """ROS / Isaac Manipulator inference fields plus deployment alignment.

    This configuration:

    - Exposes variables needed for Isaac Manipulator ROS inference.
    - Aligns robot mounting pose with a vertical (table-top) Flexiv Rizon 4s installation.
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

        # --- Flexiv Rizon 4s mount: vertical (table-top) ---
        # Home joint pose seeds the grasp IK; set to the physical station's pose.
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

        # Wall-mount configuration (GB300 Hubble Lab, rot=(0.5,0.5,0.5,0.5)):
        # self.scene.robot.init_state.joint_pos = {
        #     "joint1": math.radians(-90.0),
        #     "joint2": math.radians(90.0),
        #     "joint3": 0.0,
        #     "joint4": math.radians(90.0),
        #     "joint5": 0.0,
        #     "joint6": 0.0,
        #     "joint7": 0.0,
        # }
        # self.scene.robot.init_state.rot = (0.5, 0.5, 0.5, 0.5)

        # Socket/plug positions account for the DisplayPort USD root-to-geometry offset.
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_SOCKET_ROOT,
            rot=_HUBBLE_SOCKET_ROT,
        )

        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_PLUG_ROOT,
            rot=_HUBBLE_PLUG_ROT,
        )

        # Increase IK iterations for the grasp event as a safety margin.
        self.events.set_robot_to_grasp_pose.params["max_iterations"] = 150

        # Fixed asset parameters for ROS inference (geometry center, not USD root)
        self.fixed_asset_init_pos_center = list(_HUBBLE_GEOMETRY_POS)

        pose_range = self.events.randomize_socket_pose.params["pose_range"]
        self.fixed_asset_init_pos_range = [
            pose_range["x"][1],
            pose_range["y"][1],
            pose_range["z"][1],
        ]
        # CALIBRATE: euler equivalent of the DP socket orientation at the station.
        # Wall-mount was [0.0, 0.0, -90.0]; vertical mount seeds to [0.0, 0.0, 0.0].
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


@configclass
class Rizon4sGravDisplayportInsertionNoJointVelROSInferenceEnvCfg(Rizon4sGravDisplayportInsertionROSInferenceEnvCfg):
    """ROS inference config for the velocity-free joint-space policy.

    Identical deployment setup to
    :class:`Rizon4sGravDisplayportInsertionROSInferenceEnvCfg`, but the actor
    observation drops joint velocity (the critic keeps it). The Isaac
    Manipulator metadata is updated so the deployed observation vector and order
    match the velocity-free actor.
    """

    def __post_init__(self):
        super().__post_init__()

        # Drop joint velocity from the actor group (critic still includes it).
        self.observations.policy.joint_vel = None

        # Isaac Manipulator metadata for the velocity-free actor.
        self.obs_order = ["arm_dof_pos", "socket_pos", "socket_quat"]
        # Observation: 7 joint pos + 3 socket pos + 4 socket quat = 14
        self.observation_space = 14
        # State (critic) is unchanged: 7 jpos + 7 jvel + 3 socket pos + 4 socket
        # quat + 3 plug pos + 4 plug quat = 28.
        self.state_space = 28
