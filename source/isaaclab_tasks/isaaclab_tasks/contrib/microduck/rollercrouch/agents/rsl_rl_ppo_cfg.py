# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck crouch-glide trick task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 8.10.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckRollerCrouchPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck crouch-glide trick task.

    Upstream's runner differs from its velocity one in exactly two fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are shared across the whole family. Symmetry augmentation is off, as it is on
    every task but the forward roll.

    Note the exploration bonus is the family's 0.01 and **not** the skating task's 0.03, even though
    this task runs on the same robot: the pose reward pays from the first step, so unlike a stroke
    the trick does not have to be found before there is any gradient at all.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_rollercrouch"
    # Upstream's budget. Short by the family's standards, and the repo's own guidance calls a simple
    # episodic trick about a thousand iterations at 4096 environments, so this is a generous ceiling
    # rather than a target.
    max_iterations = 8000
