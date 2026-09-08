# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.cabinet.config.franka.agents.rsl_rl_ppo_cfg import CabinetPPORunnerCfg


@configclass
class FrankaCabinetIKAbsPPORunnerCfg(CabinetPPORunnerCfg):
    experiment_name = "franka_open_drawer_ik_abs"


@configclass
class FrankaCabinetIKRelPPORunnerCfg(CabinetPPORunnerCfg):
    experiment_name = "franka_open_drawer_ik_rel"
