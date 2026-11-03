# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck stand-up-on-skates task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 9.10.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckRollerStandUpPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck stand-up-on-skates task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family. It declares the family's exploration
    bonus rather than the skating task's tripled one, even though it derives from the skating
    environment.

    Symmetry augmentation is off, as it is on every task but the forward roll. Upstream's stated
    reason -- that the augmentation is wired for a 51-wide observation -- is stale, since it was
    migrated to the 61-wide layout in 2026-08 and the forward-roll task runs with it on; the setting
    is reproduced, the reason is not.

    Note:
        The environment's wheel-friction curriculum finishes its last stage at iteration 4000, and
        before that the policy leans on a rolling friction the hardware does not have. **Only
        checkpoints from iteration 4000 onward are deployment candidates.**
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_rollerstandup"
    max_iterations = 15000
