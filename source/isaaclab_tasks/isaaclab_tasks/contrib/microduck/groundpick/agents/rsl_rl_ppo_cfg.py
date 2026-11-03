# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck ground-pick task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckGroundPickPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck ground-pick task."""

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_groundpick"
    # The curricula have all finished ramping by iteration 2000, but the gesture has to hold together across the
    # whole domain-randomization spread -- including the family's widest centre-of-mass range -- and it is the
    # balance rather than the trajectory that takes the iterations.
    max_iterations = 20000
