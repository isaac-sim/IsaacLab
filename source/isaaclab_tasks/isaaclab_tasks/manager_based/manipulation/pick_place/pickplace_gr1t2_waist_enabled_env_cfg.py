# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import tempfile

try:
    from isaaclab_teleop import IsaacTeleopCfg, XrCfg

    _TELEOP_AVAILABLE = True
except ImportError:
    _TELEOP_AVAILABLE = False
    logging.getLogger(__name__).warning("isaaclab_teleop is not installed. XR teleoperation features will be disabled.")

import isaaclab.controllers.utils as ControllerUtils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .pickplace_gr1t2_env_cfg import (
    ActionsCfg,
    EventCfg,
    ObjectTableSceneCfg,
    ObservationsCfg,
    TerminationsCfg,
    _build_gr1t2_pickplace_pipeline,
)


@configclass
class PickPlaceGR1T2WaistEnabledEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    # Unused managers
    commands = None
    rewards = None
    curriculum = None

    # Temporary directory for URDF files
    temp_urdf_dir = tempfile.gettempdir()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2

        # Add waist joint to pink_ik_cfg
        waist_joint_names = ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
        for joint_name in waist_joint_names:
            self.actions.upper_body_ik.pink_controlled_joint_names.append(joint_name)

        # Convert USD to URDF and change revolute joints to fixed
        temp_urdf_output_path, temp_urdf_meshes_output_path = ControllerUtils.convert_usd_to_urdf(
            self.scene.robot.spawn.usd_path, self.temp_urdf_dir, force_conversion=True
        )

        # Set the URDF and mesh paths for the IK controller
        self.actions.upper_body_ik.controller.urdf_path = temp_urdf_output_path
        self.actions.upper_body_ik.controller.mesh_path = temp_urdf_meshes_output_path

        # IsaacTeleop-based teleoperation pipeline
        if _TELEOP_AVAILABLE:
            self.xr = XrCfg(
                anchor_pos=(0.0, 0.0, 0.0),
                anchor_rot=(0.0, 0.0, 0.0, 1.0),
            )
            pipeline = _build_gr1t2_pickplace_pipeline()
            self.isaac_teleop = IsaacTeleopCfg(
                pipeline_builder=lambda: pipeline,
                sim_device=self.sim.device,
                xr_cfg=self.xr,
            )
