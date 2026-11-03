# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for the MicroDuck ball-kick task.

Every hyper-parameter is upstream's; see ``historical tasks3 source extraction``
section 6.9.

Source pins and historical extraction labels are indexed in this task family's ``UPSTREAM.md``.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.contrib.microduck.ballkick.ballkick_env_cfg import MICRODUCK_KICK_FOOT
from isaaclab_tasks.contrib.microduck.velocity.agents.rsl_rl_ppo_cfg import MicroDuckPPORunnerCfg


@configclass
class MicroDuckBallKickPPORunnerCfg(MicroDuckPPORunnerCfg):
    """PPO runner for the MicroDuck ball-kick task.

    Upstream's ball-kick runner differs from its velocity one in exactly two fields, so the rest is
    inherited rather than restated: the network shapes, the optimizer, the asymmetric actor-critic
    wiring and the 24-step rollout are all shared across the whole family.
    """

    # A separate log tree per kicking foot, as upstream names it: the mirrored kick is a different
    # policy, not a different episode, so the two must not share a run directory.
    experiment_name = f"microduck_ball_kick_{MICRODUCK_KICK_FOOT}"
    # A fifth of the velocity budget. The kick is one skill against one fixed target and upstream's
    # schedules have all finished ramping by iteration 1500.
    max_iterations = 10000
