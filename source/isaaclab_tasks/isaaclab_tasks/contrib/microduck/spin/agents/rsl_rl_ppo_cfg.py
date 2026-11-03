# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck spin trick task."""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckSpinPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck spin trick task.

    Symmetry augmentation is off, and this is one of the two tasks in the batch that gives a correct
    task-level reason for it rather than a stale technical one: a left-right mirror would turn a
    counter-clockwise spin into a clockwise one, which is a different behavior rather than an
    equivalent sample.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_spin"

    max_iterations = 8000
