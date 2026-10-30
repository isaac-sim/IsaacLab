# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Optional single-GPU MicroDuck recipe with a wider yaw reward and symmetry augmentation."""

import gymnasium as gym

from isaaclab.utils import configclass

from isaaclab_tasks.contrib.microduck.agents.rsl_rl_ppo_cfg import MicroDuckSymmetricPPORunnerCfg
from isaaclab_tasks.contrib.microduck.flat_env_cfg import MicroDuckVelocityFlatEnvCfg


@configclass
class FastEnvCfg(MicroDuckVelocityFlatEnvCfg):
    """Widen the yaw tracking kernel to give early turning behavior more reward."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel.params["std"] = 0.5


@configclass
class FastRunnerCfg(MicroDuckSymmetricPPORunnerCfg):
    """Use the original PPO settings with mirrored samples and no mirror loss."""

    experiment_name = "microduck_velocity_flat_fast"


def register():
    """Register the optional flat-ground recipe through the external callback interface."""
    gym.register(
        id="IsaacContrib-Velocity-Flat-MicroDuck-FastStudy",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": FastEnvCfg, "rsl_rl_cfg_entry_point": FastRunnerCfg},
    )
