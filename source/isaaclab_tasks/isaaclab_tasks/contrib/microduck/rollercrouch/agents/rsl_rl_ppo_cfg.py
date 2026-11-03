# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck crouch-glide trick task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckRollerCrouchPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck crouch-glide trick task.

    Note the exploration bonus is the family's 0.01 and **not** the skating task's 0.03, even though
    this task runs on the same robot: the pose reward pays from the first step, so unlike a stroke
    the trick does not have to be found before there is any gradient at all.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_rollercrouch"
    # Short by the family's standards, and the repo's own guidance calls a simple episodic trick about a thousand
    # iterations at 4096 environments, so this is a generous ceiling rather than a target.
    max_iterations = 8000
