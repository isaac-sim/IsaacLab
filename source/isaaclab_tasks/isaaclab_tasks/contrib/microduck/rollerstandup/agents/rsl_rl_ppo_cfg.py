# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck stand-up-on-skates task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckRollerStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck stand-up-on-skates task.

    Note:
        The environment's wheel-friction curriculum finishes its last stage at iteration 4000, and
        before that the policy leans on a rolling friction the hardware does not have. **Only
        checkpoints from iteration 4000 onward are deployment candidates.**
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_rollerstandup"
    max_iterations = 15000
