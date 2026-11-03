# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck sit-stand task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 4.10.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckSitStandPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck sit-stand task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_sitstand"
    # the stand-up task's budget, and for the same reason: two transitions between two fixed
    # keyframes, with every schedule finished ramping by iteration 2500.
    max_iterations = 15000
