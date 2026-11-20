# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck whole-body, foot and head tracking with native Newton BAM servos."""

from __future__ import annotations

from typing import Literal

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass, replace
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckSceneCfg, MicroDuckVelocityFlatEnvCfg

from isaaclab_assets.robots.microduck import MICRODUCK_CFG

from ... import mdp
from ...mdp.commands_cfg import MotionCommandCfg
from ...motion_tracking_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    MotionTrackingEnvCfg,
    MotionTrackingSceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .rewards import motion_body_position_error_exp


@configclass
class MicroDuckMotionCommandCfg(MotionCommandCfg):
    """MicroDuck reference bodies, velocity convention and reset perturbations."""

    body_name_map: dict[str, str] = {"neck_pitch_1": "neck_pitch"}
    input_velocity_point: Literal["link", "com"] = "link"

    asset_name: str = "robot"
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    anchor_body_name: str = "trunk_base"
    body_names: list[str] = [
        "trunk_base",
        "yaw2roll",
        "hip_l",
        "upper_leg_left",
        "leg",
        "ankle_left",
        "neck",
        "neck_pitch_1",
        "yaw_roll_motion",
        "jaw_soft",
        "bearing_roll",
        "hip_l_2",
        "upper_leg_right",
        "leg_2",
        "ankle_right",
    ]
    pose_range: dict[str, tuple[float, float]] = {
        "x": (-0.02, 0.02),
        "y": (-0.02, 0.02),
        "z": (-0.003, 0.003),
        "roll": (-0.05, 0.05),
        "pitch": (-0.05, 0.05),
        "yaw": (-0.1, 0.1),
    }
    velocity_range: dict[str, tuple[float, float]] = {
        "x": (-0.05, 0.05),
        "y": (-0.05, 0.05),
        "z": (-0.02, 0.02),
        "roll": (-0.2, 0.2),
        "pitch": (-0.2, 0.2),
        "yaw": (-0.2, 0.2),
    }
    joint_position_range: tuple[float, float] = (-0.03, 0.03)


@configclass
class MicroDuckTrackingSceneCfg(MotionTrackingSceneCfg):
    """MicroDuck's articulated body and foot-to-foot contact sensor."""

    robot = replace(MICRODUCK_CFG, prim_path="{ENV_REGEX_NS}/Robot", soft_joint_pos_limit_factor=1.0)
    self_collision = MicroDuckSceneCfg().self_collision
    contact_forces = MicroDuckSceneCfg().contact_forces


@configclass
class MicroDuckTrackingActionsCfg(ActionsCfg):
    """Servo position targets in the articulation's native joint order."""

    joint_pos = replace(ActionsCfg().joint_pos, scale=0.25)


@configclass
class MicroDuckTrackingRewardsCfg(RewardsCfg):
    """Track each foot and the head in addition to the shared whole-body rewards."""

    motion_feet_pos = RewTerm(
        func=motion_body_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "body_names": ["ankle_left", "ankle_right"], "std": 0.02},
    )
    motion_head_pos = RewTerm(
        func=motion_body_position_error_exp,
        weight=4.0,
        params={"command_name": "motion", "body_names": ["jaw_soft"], "std": 0.02},
    )


@configclass
class MicroDuckTrackingTerminationsCfg(TerminationsCfg):
    """Control foot tracking tolerance separately from head tracking tolerance."""

    feet_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={"command_name": "motion", "body_names": ["ankle_left", "ankle_right"], "threshold": 0.09},
    )


@configclass
class MicroDuckMotionTrackingEnvCfg(MotionTrackingEnvCfg):
    """Whole-body tracking at 50 Hz using MicroDuck's 200 Hz BAM dynamics."""

    scene: MicroDuckTrackingSceneCfg = MicroDuckTrackingSceneCfg(num_envs=4096, env_spacing=1.0)
    actions: MicroDuckTrackingActionsCfg = MicroDuckTrackingActionsCfg()
    commands: CommandsCfg = CommandsCfg(motion=MicroDuckMotionCommandCfg())
    rewards: MicroDuckTrackingRewardsCfg = MicroDuckTrackingRewardsCfg()
    terminations: MicroDuckTrackingTerminationsCfg = MicroDuckTrackingTerminationsCfg()
    sim = MicroDuckVelocityFlatEnvCfg().sim

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 14.0
        self.sim.use_newton_actuators = True
        self.sim.visualizer_cfgs = VisualizerCfg(eye=(0.6, 0.6, 0.35), lookat=(0.0, 0.0, 0.12))
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.self_collision.update_period = self.sim.dt
        self.rewards.motion_global_anchor_pos.params["std"] = 0.05
        self.rewards.motion_body_pos.params["std"] = 0.05
        self.rewards.motion_body_lin_vel.params["std"] = 0.25
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.self_collisions.weight = -1.0
        self.rewards.self_collisions.params["threshold"] = 0.1
        self.terminations.anchor_pos.params["threshold"] = 0.15
        self.terminations.ee_body_pos.params.update(threshold=0.18, body_names=["jaw_soft"])
        self.events.base_mass.params.update(
            asset_cfg=SceneEntityCfg("robot", body_names="trunk_base"),
            mass_distribution_params=(0.95, 1.05),
        )
        self.events.physics_material.params.update(static_friction_range=(0.7, 1.3), dynamic_friction_range=(0.7, 1.3))
        self.events.push_robot = None

    def play_mode(self):
        super().play_mode()
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
