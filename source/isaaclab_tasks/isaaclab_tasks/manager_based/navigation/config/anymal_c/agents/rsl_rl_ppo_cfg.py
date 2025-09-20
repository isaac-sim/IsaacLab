# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.navigation.config.base_rsl_rl_ppo_cfg import NavBasePPORunnerCfg


@configclass
class NavigationEnvPPORunnerCfg(NavBasePPORunnerCfg):
    max_iterations = 1500
    save_interval = 50
    experiment_name = "anymal_c_navigation"
