# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck ground-pick task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 5.9.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckGroundPickPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck ground-pick task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family. Symmetry augmentation is off, as it is on
    every task but the forward roll.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_groundpick"
    # Upstream's budget, matching the velocity-plus-recovery task's. The curricula have all finished
    # ramping by iteration 2000, but the gesture has to hold together across the whole
    # domain-randomization spread -- including the family's widest centre-of-mass range -- and it is
    # the balance rather than the trajectory that takes the iterations.
    max_iterations = 20000
