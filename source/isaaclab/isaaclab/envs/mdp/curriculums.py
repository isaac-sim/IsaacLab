# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


def modify_environment_parameter(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    get_term_fn: callable,
    modify_term_fn: callable,
    value: float,
    num_steps: int,
):
    """General function to modify termination, reward, or command parameters in an RL environment.
    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        get_term_fn: Function to retrieve the term configuration.
        modify_term_fn: Function to modify and set the retrieved term configuration.
        value: Value for command modifications, how it is used should be defined via modify_term_fn.
        num_steps: The step interval at which the modification is applied
                   (i.e., at steps num_steps, 2*num_steps, 3*num_steps, etc., but not at step 0).
    """
    # Check if it's time to apply the modification
    if env.common_step_counter % num_steps == 0 and env.common_step_counter != 0:
        term_cfg = get_term_fn(env)
        modify_term_fn(env, term_cfg, value)
