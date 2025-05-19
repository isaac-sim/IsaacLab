# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for a single drive"""

from __future__ import annotations

import math

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from rai.eval_sim.tasks.single_drive.single_drive_scene_cfg import SingleDriveSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_positions = mdp.JointPositionActionCfg(asset_name="robot", joint_names=["Motor_Joint"], scale=1.0)
    joint_velocities = mdp.JointVelocityActionCfg(asset_name="robot", joint_names=["Motor_Joint"], scale=1.0)
    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["Motor_Joint"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObservationTermCfg(func=mdp.joint_pos_rel)
        joint_vel_rel = ObservationTermCfg(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_actuator = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["Motor_Joint"]),
            "position_range": (-math.pi, math.pi),
            "velocity_range": (-0.1 * math.pi, 0.1 * math.pi),
        },
    )


@configclass
class SingleDriveEnvCfg(ManagerBasedEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = SingleDriveSceneCfg()
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [1.2, 0.0, 1.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        # step settings
        self.decimation = 1  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        self.sim.render_interval = 16  # render step every 16 sim steps: 0.002*16 = 0.032 -> 32 Hz
        # simulation settings
        self.sim.dt = 0.002  # sim step every 5ms: 200Hz
