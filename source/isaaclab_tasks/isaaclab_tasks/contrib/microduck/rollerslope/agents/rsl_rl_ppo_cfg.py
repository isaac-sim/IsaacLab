# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck slope-descent task.

Every hyper-parameter is upstream's; see ``historical tasks4 source extraction``
section 7.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckRollerSlopePPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck slope-descent task.

    Upstream's runner differs from its **velocity** one in exactly two fields, so the rest is
    inherited rather than restated: the network shapes, the optimizer, the asymmetric actor-critic
    wiring and the 24-step rollout are shared across the whole family. Symmetry augmentation is off,
    as it is on every task but the forward roll.

    It derives from the velocity runner and **not** from the skating one, even though it runs the
    skating robot and the skating recipe: that class raises ``entropy_coef`` to 0.03 for the stroke,
    and upstream declares the family's 0.01 here. Deriving from it would silently triple the
    exploration noise -- and unlike the stroke, this task's positive reward pays from the first step,
    because gravity turns the wheels whether or not the policy has found anything.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_rollerslope"
    # Upstream's budget, shorter than the family's 50 000: the objective is to stay upright on a
    # ramp the curriculum makes steeper, not to master a stride.
    max_iterations = 8000
