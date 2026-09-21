# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO baseline for the contact-rich smoothie task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg


@configclass
class FrankaSmoothiePPORunnerCfg(CabinetPPORunnerCfg):
    """PPO configuration for the rigid-body smoothie task's state observations."""

    experiment_name = "franka_smoothie"
    num_steps_per_env = 32
    max_iterations = 3000
    save_interval = 25
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}

    def __post_init__(self):
        self.actor.obs_normalization = True
        self.critic.obs_normalization = True
