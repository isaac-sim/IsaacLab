# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp

from .rough_env_cfg import H1Rewards, H1RoughEnvCfg


@configclass
class H1FlatRewards(H1Rewards):
    """Reward terms for the MDP."""

    # fixes the tilted gait
    air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "command_name": "base_velocity",
        },
    )


@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    rewards: H1FlatRewards = H1FlatRewards()

    def __post_init__(self):
        super().__post_init__()

        # physics
        newton_mjwarp = self.sim.physics.newton_mjwarp
        newton_mjwarp.solver_cfg.njmax = 65
        newton_mjwarp.solver_cfg.nconmax = 15
        self.sim.physics.default = newton_mjwarp
        # scene
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        # observations
        self.observations.policy.height_scan = None
        # rewards
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
        # curriculum
        self.curriculum.terrain_levels = None
