# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import logging
import tempfile

from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.visualizers import HandJointVisualizer
from isaaclab_teleop.xr_cfg import XrCfg

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

logger = logging.getLogger(__name__)


@configclass
class PickPlaceGR1T2WaistEnabledEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the GR1T2 environment."""

    # When True, the teleop pipeline exposes hand_left/hand_right for debugging visualization.
    enable_visualization: bool = True

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

        # Defer USD→URDF conversion to controller initialization (requires Isaac Sim at runtime).
        self.actions.upper_body_ik.controller.usd_path = self.scene.robot.spawn.usd_path
        self.actions.upper_body_ik.controller.urdf_output_dir = self.temp_urdf_dir

        # IsaacTeleop-based teleoperation pipeline.
        self.xr = XrCfg(
            anchor_pos=(0.0, 0.0, 0.0),
            anchor_rot=(0.0, 0.0, 0.0, 1.0),
        )
        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=lambda _s=self: _build_gr1t2_pickplace_pipeline(
                enable_visualization=_s.enable_visualization,
            )[0],
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

    def get_teleop_visualizers(self, teleop_interface):
        """Return teleop visualizers to update each frame (e.g. hand joint markers).

        Call :meth:`update` after each advance(), then call visualizer.update() for each returned object.

        Returns:
            List of visualizer objects with an update() method. Empty if
            enable_visualization is False.
        """
        if not self.enable_visualization:
            return []
        visualizers = []
        if HandJointVisualizer.supports(teleop_interface):
            visualizers.append(HandJointVisualizer(teleop_interface))
        else:
            logger.error(
                "Hand joint visualization enabled but teleop interface is not supported by HandJointVisualizer "
                "(expected IsaacTeleopDevice with session lifecycle)"
            )
        return visualizers
