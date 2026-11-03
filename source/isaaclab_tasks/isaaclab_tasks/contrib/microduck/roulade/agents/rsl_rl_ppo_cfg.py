# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO configuration for the MicroDuck forward-roll task."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlSymmetryCfg

from isaaclab_tasks.contrib.microduck.mdp.symmetry import compute_roulade_symmetric_states
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg

MICRODUCK_ROULADE_MIRROR_LOSS_COEFF = 0.5
"""Weight of the Roulade actor mirror loss."""


@configclass
class MicroDuckRouladePPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck forward-roll task."""

    # a separate log tree: the MicroDuck policies are trained and deployed independently
    experiment_name = "microduck_roulade"
    # a fifth of the velocity budget.
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
