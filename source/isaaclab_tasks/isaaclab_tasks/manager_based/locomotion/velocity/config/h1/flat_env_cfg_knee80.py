# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""H1 flat environment with knee joint limit constrained to 80 degrees."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .flat_env_cfg import H1FlatEnvCfg
from .rough_env_cfg import H1Rewards


# Knee limit in radians (80 degrees)
KNEE_LIMIT_RAD = 80.0 * math.pi / 180.0  # ~1.396 rad


@configclass
class H1Knee80Rewards(H1Rewards):
    """Reward terms with additional knee limit penalty at 80 degrees."""

    # Penalize knee joint positions exceeding 80 degrees
    knee_limit_penalty = RewTerm(
        func=mdp.joint_pos_limit_custom,
        weight=-2.0,
        params={
            "upper_limit": KNEE_LIMIT_RAD,
            "lower_limit": -KNEE_LIMIT_RAD,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_knee"),
        },
    )


@configclass
class H1FlatEnvCfg_Knee80(H1FlatEnvCfg):
    """H1 flat terrain environment with knee joint limits constrained to 80 degrees.

    This configuration adds a reward penalty for knee positions exceeding 80 degrees.
    """

    rewards: H1Knee80Rewards = H1Knee80Rewards()

    def __post_init__(self):
        super().__post_init__()

        # Ensure initial knee position is within the 80 degree limit
        # Default is 0.79 rad (~45 deg) which is already within limit
        current_knee_pos = self.scene.robot.init_state.joint_pos.get(".*_knee", 0.79)
        self.scene.robot.init_state.joint_pos[".*_knee"] = min(current_knee_pos, KNEE_LIMIT_RAD)
