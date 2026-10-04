# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking curriculums."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.envs.mdp import modify_term_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def staged_value(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], old_value: Any, stages: Sequence[tuple[int, Any]]
) -> Any:
    """``modify_fn`` for :class:`~isaaclab.envs.mdp.modify_term_cfg` that steps through a schedule.

    Args:
        stages: ``(start_step, value)`` pairs in increasing step order; the last reached value applies.
    """
    value = old_value
    for start_step, stage_value in stages:
        if env.common_step_counter >= start_step:
            value = stage_value
    return modify_term_cfg.NO_CHANGE if value == old_value else value
