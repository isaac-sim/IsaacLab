# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-space ROS inference configuration for DisplayPort insertion with Flexiv Rizon 4S.

Inherits from the joint-space training config and adds Isaac Manipulator
metadata fields plus Hubble Lab deployment alignment. Mirrors the GB300
``config/rizon_4s/ros_inference_env_cfg.py``.
"""

import math

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.deploy.cable_insertion.displayport_insertion_env_cfg import (
    compute_plug_pose,
    compute_socket_root,
)

from .joint_pos_env_cfg import Rizon4sGravDisplayportInsertionEnvCfg

# ---------------------------------------------------------------------------
# Hubble Lab socket/plug positions
# ---------------------------------------------------------------------------
# CALIBRATE: deployment values seeded from the GB300 cable-insertion ROS config.
# Re-measure the real DisplayPort fixture pose and the Flexiv mount for the
# actual station. The socket rotation keeps the verified DP orientation
# (opening facing +Z) used by the training configs.
_HUBBLE_GEOMETRY_POS = (0.928, 0.129, -0.1)
_HUBBLE_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_HUBBLE_PLUG_CLEARANCE_Z = 0.068

_HUBBLE_SOCKET_ROOT = compute_socket_root(_HUBBLE_GEOMETRY_POS, _HUBBLE_SOCKET_ROT)
_HUBBLE_PLUG_ROOT, _HUBBLE_PLUG_ROT = compute_plug_pose(
    _HUBBLE_GEOMETRY_POS, _HUBBLE_SOCKET_ROT, z_clearance=_HUBBLE_PLUG_CLEARANCE_Z,
)


@configclass
class Rizon4sGravDisplayportInsertionROSInferenceEnvCfg(Rizon4sGravDisplayportInsertionEnvCfg):
    """ROS / Isaac Manipulator inference fields plus deployment alignment for NVIDIA Hubble Lab.

    This configuration:

    - Exposes variables needed for Isaac Manipulator ROS inference.
    - Aligns robot mounting pose with the Flexiv Rizon 4s installation at NVIDIA Hubble Lab
      (wall-mount with 90 deg rotation about negative X-axis).
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

        # --- NVIDIA Hubble Lab: Flexiv Rizon 4s mount ---
        # CALIBRATE: home joint pose / mount seeded from GB300; align with the
        # physical DisplayPort station before on-robot inference.
        self.scene.robot.init_state.joint_pos = {
            "joint1": math.radians(-90.0),
            "joint2": math.radians(90.0),
            "joint3": 0.0,
            "joint4": math.radians(90.0),
            "joint5": 0.0,
            "joint6": 0.0,
            "joint7": 0.0,
        }

        # Orientation of robot is based on the Flexiv Rizon 4s mount in the Hubble Lab
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.0)
        self.scene.robot.init_state.rot = (0.5, 0.5, 0.5, 0.5)

        # Socket/plug positions account for the DisplayPort USD root-to-geometry offset.
        self.scene.dp_socket.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_SOCKET_ROOT,
            rot=_HUBBLE_SOCKET_ROT,
        )

        self.scene.dp_plug.init_state = RigidObjectCfg.InitialStateCfg(
            pos=_HUBBLE_PLUG_ROOT,
            rot=_HUBBLE_PLUG_ROT,
        )

        # Increase IK iterations for the grasp event — the Hubble home pose is
        # further from typical IK solutions than the table-mount default.
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
        self.fixed_asset_init_orn_deg = [0.0, 0.0, -90.0]
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
