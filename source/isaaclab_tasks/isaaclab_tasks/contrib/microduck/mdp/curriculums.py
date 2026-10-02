# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking curriculums."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_stage_index(
    env: ManagerBasedRLEnv, stages: Sequence[dict[str, Any]], key: str, term_name: str, *, inclusive: bool
) -> int | None:
    """Return the index of the last stage the run has passed, or None if it has passed none."""
    if not stages:
        raise ValueError(f"The curriculum term '{term_name}' was given an empty stage table.")
    previous_step: int | None = None
    for index, stage in enumerate(stages):
        if "step" not in stage or key not in stage:
            raise ValueError(f"{term_name} stage {index} requires both 'step' and {key!r}: {stage}.")
        if previous_step is not None and stage["step"] <= previous_step:
            raise ValueError(
                f"{term_name} stages must have strictly increasing steps: {stage['step']} follows {previous_step}."
            )
        previous_step = stage["step"]
    resolved = None
    for index, stage in enumerate(stages):
        passed = env.common_step_counter >= stage["step"] if inclusive else env.common_step_counter > stage["step"]
        if passed:
            resolved = index
    return resolved


def _resolve_stage(
    env: ManagerBasedRLEnv, stages: Sequence[dict[str, Any]], key: str, term_name: str, *, inclusive: bool
) -> Any | None:
    """Return the payload of the last stage the run has passed, or None if it has passed none."""
    index = _resolve_stage_index(env, stages, key, term_name, inclusive=inclusive)
    return None if index is None else stages[index][key]


def reward_weight_stages(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_name: str, weight_stages: Sequence[dict[str, Any]]
) -> float:
    """Ramp a reward weight through a staged schedule."""
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    weight = _resolve_stage(env, weight_stages, "weight", "reward_weight_stages", inclusive=False)
    if weight is not None:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(reward_name, term_cfg)
    return term_cfg.weight


def standing_envs_stages(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, standing_stages: Sequence[dict[str, Any]]
) -> float:
    """Ramp the fraction of environments commanded to stand still."""
    del env_ids
    command_cfg = env.command_manager.get_term(command_name).cfg
    fraction = _resolve_stage(env, standing_stages, "rel_standing_envs", "standing_envs_stages", inclusive=False)
    if fraction is not None:
        command_cfg.rel_standing_envs = fraction
    return command_cfg.rel_standing_envs


def command_range_stages(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], command_name: str, range_stages: Sequence[dict[str, Any]]
) -> float:
    """Widen the per-dimension ranges of a pose-delta command through a staged schedule."""
    del env_ids
    command_cfg = env.command_manager.get_term(command_name).cfg
    ranges = _resolve_stage(env, range_stages, "ranges", "command_range_stages", inclusive=True)
    if ranges is None:
        ranges = range_stages[0]["ranges"]
    command_cfg.ranges = tuple(ranges)
    return max((max(abs(low), abs(high)) for low, high in command_cfg.ranges), default=0.0)


def event_range_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str,
    range_stages: Sequence[dict[str, Any]],
    param_name: str = "com_range",
    range_keys: Sequence[str] = ("x", "y", "z"),
) -> float:
    """Widen a symmetric per-axis range of an event term through a staged schedule."""
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    half_width = _resolve_stage(env, range_stages, "range", "event_range_stages", inclusive=False)
    if half_width is None:
        half_width = range_stages[0]["range"]
    event_cfg.params[param_name] = {key: (-half_width, half_width) for key in range_keys}
    return half_width
