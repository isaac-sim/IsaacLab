# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck stand-up task.

Every hyper-parameter is upstream's; see ``historical tasks2 source extraction``
section 3.10.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck stand-up task.

    Upstream's stand-up runner differs from its velocity one in exactly two fields, so the rest is
    inherited rather than restated: the network shapes, the optimizer, the asymmetric actor-critic
    wiring and the 24-step rollout are all shared.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_stand"
    # a third of the velocity budget. Standing up is one skill reaching one fixed target, and
    # upstream's schedules have all finished ramping by iteration 4000.
    max_iterations = 15000
