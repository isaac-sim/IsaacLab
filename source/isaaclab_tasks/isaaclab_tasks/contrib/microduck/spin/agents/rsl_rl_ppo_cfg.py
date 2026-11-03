# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck spin trick task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 10.8.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckSpinPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck spin trick task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family.

    Symmetry augmentation is off, and this is one of the two tasks in the batch that gives a correct
    task-level reason for it rather than a stale technical one: a left-right mirror would turn a
    counter-clockwise spin into a clockwise one, which is a different behavior rather than an
    equivalent sample.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_spin"
    # Upstream's budget, the same ceiling as the crouch-glide trick's.
    max_iterations = 8000
