# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck forward-roll task.

Every hyper-parameter is upstream's; see ``historical tasks2 source extraction``
section 4.11.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_roulade_symmetric_states
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

MICRODUCK_ROULADE_MIRROR_LOSS_COEFF = 0.5
"""Weight upstream gives the symmetry-mirror loss on the one task in the family that uses it."""


@configclass
class MicroDuckRouladePPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck forward-roll task.

    Upstream's roll runner differs from its velocity one in three fields, so the rest is inherited
    rather than restated: the network shapes, the optimizer, the asymmetric actor-critic wiring and
    the 24-step rollout are all shared.

    The third field is the interesting one. A forward roll is left-right symmetric, and upstream
    turned the family's symmetry machinery on here -- for the first time -- specifically to fight
    the sideways collapse its earlier runs kept converging to: the policy would roll over a
    shoulder, which is a lower-energy path than going over the head because it avoids the fully
    inverted configuration. The reward set attacks that structurally, through the accumulator's
    sagittal gate and the head-top latch; the mirror loss attacks it in the policy, by penalizing a
    policy that behaves differently on mirrored observations.
    """

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_roulade"
    # a fifth of the velocity budget. The roll is one short manoeuvre and upstream's schedules have
    # all finished ramping by iteration 6000.
    max_iterations = 10000
    # The mirror **loss**, not data augmentation: the loss is defined on the actor's action means
    # and never touches the critic group, which is what makes it sound with the family's symmetry
    # tables -- they mirror the 61-wide actor observation and leave the privileged group alone. See
    # :mod:`~isaaclab_tasks.contrib.microduck.mdp.symmetry`.
    # ``replace`` on an *instance* of the parent, because a configclass turns a mutable default
    # into a field factory and leaves no class attribute to read; the result is an independent copy.
    algorithm = MicroDuckPPORunnerCfg().algorithm.replace(
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=False,
            use_mirror_loss=True,
            mirror_loss_coeff=MICRODUCK_ROULADE_MIRROR_LOSS_COEFF,
            data_augmentation_func=compute_roulade_symmetric_states,
        )
    )
