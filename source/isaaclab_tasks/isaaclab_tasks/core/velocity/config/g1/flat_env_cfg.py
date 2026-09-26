# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1 velocity-tracking environment on flat terrain."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ... import mdp
from .rough_env_cfg import G1RoughEnvCfg


@configclass
class G1FlatEnvCfg(G1RoughEnvCfg):
    """Plane-terrain G1 walking with the same 29-action, 286-observation interface."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        # Retain the height scanner for the observation and terrain-relative height terms.
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.joint_deviation_arms.weight = -0.1
        self.rewards.pelvis_height.params["target_height"] = 0.686
        self.rewards.feet_air_time.weight = 1.5
        self.rewards.feet_air_time_variance = RewTerm(
            func=mdp.feet_air_time_variance,
            weight=-24.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            },
        )
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.rewards.feet_flight = RewTerm(
            func=mdp.feet_flight,
            weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
        )
