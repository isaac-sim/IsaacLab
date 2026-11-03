# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curricula for command ranges, disturbances, rewards and terrain difficulty."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.envs.mdp import modify_term_cfg

from .commands import MicroDuckRoughVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


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


def microduck_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | slice,
    min_distance: float = 1.0,
    min_walking_time: float = 5.0,
    promote_error_ratio: float = 0.5,
    demote_error_ratio: float = 0.8,
    promote_yaw_error: float = 0.5,
    demote_yaw_error: float = 0.8,
) -> dict[str, torch.Tensor]:
    """Advance terrain after surviving and tracking sufficient commanded walking.

    Accumulated path length allows changing direction without losing progress. Mostly stationary episodes
    hold their level; failures demote.

    Args:
        env: The environment.
        env_ids: Environments about to reset.
        min_distance: Minimum commanded and actual walking distance for promotion [m].
        min_walking_time: Minimum translation-command duration [s], also at least 25% of the episode.
        promote_error_ratio: Maximum integrated XY error divided by commanded distance for promotion.
        demote_error_ratio: Integrated XY error divided by commanded distance above which to demote.
        promote_yaw_error: Maximum episode-mean yaw-rate error for promotion [rad/s].
        demote_yaw_error: Episode-mean yaw-rate error above which to demote [rad/s].
    """
    command: MicroDuckRoughVelocityCommand = env.command_manager.get_term("base_velocity")
    command.record_terrain_progress(env_ids)
    progress = {name: value[env_ids] for name, value in command.terrain_progress.items()}
    error_ratio = progress["error_distance"] / progress["commanded_distance"].clamp_min(1e-6)
    yaw_error = progress["yaw_error"] / progress["time"].clamp_min(env.step_dt)
    eligible = (
        (progress["commanded_distance"] >= min_distance)
        & (progress["walking_time"] >= min_walking_time)
        & (progress["walking_time"] >= 0.25 * progress["time"])
    )
    failed = env.termination_manager.terminated[env_ids]
    move_up = (
        eligible
        & ~failed
        & env.termination_manager.time_outs[env_ids]
        & (progress["walked_distance"] >= min_distance)
        & (error_ratio <= promote_error_ratio)
        & (yaw_error <= promote_yaw_error)
    )
    move_down = (
        (progress["time"] > 0.0)
        & (failed | (eligible & ((error_ratio > demote_error_ratio) | (yaw_error > demote_yaw_error))))
        & ~move_up
    )
    terrain = env.scene.terrain
    terrain.update_env_origins(env_ids, move_up, move_down)
    return {
        "mean": terrain.terrain_levels.float().mean(),
        "promoted": move_up.float().mean(),
        "demoted": move_down.float().mean(),
        "eligible": eligible.float().mean(),
        "walked_distance": progress["walked_distance"].nan_to_num().mean(),
    }


def _resolve_stage_index(
    env: ManagerBasedRLEnv,
    stages: Sequence[dict[str, Any]],
    key: str,
    term_name: str,
    *,
    inclusive: bool,
) -> int | None:
    """Return the index of the last stage the run has passed, or None if it has passed none.

    Args:
        env: The environment instance, read for :attr:`common_step_counter`.
        stages: The stage table, ordered by strictly increasing ``"step"``.
        key: Name of the payload entry every stage must carry.
        term_name: Name of the calling term, used in the error messages.
        inclusive: Whether a stage triggers on the step that equals its boundary.

    Returns:
        The index of the applicable stage, or None if the run is still before the first boundary.

    Raises:
        ValueError: If the stage table is empty, is not ordered by strictly increasing steps, or
            has a stage missing either ``"step"`` or the payload entry.
    """
    if not stages:
        raise ValueError(f"The curriculum term '{term_name}' was given an empty stage table.")
    previous_step: int | None = None
    for index, stage in enumerate(stages):
        if "step" not in stage or key not in stage:
            raise ValueError(
                f"Stage {index} of the curriculum term '{term_name}' must carry both a 'step' and a"
                f" '{key}' entry. Received: {stage}."
            )
        if previous_step is not None and stage["step"] <= previous_step:
            raise ValueError(
                f"The stage table of the curriculum term '{term_name}' must be ordered by strictly"
                f" increasing 'step', so that later stages win. Stage {index} steps at"
                f" {stage['step']} after {previous_step}."
            )
        previous_step = stage["step"]

    resolved = None
    for index, stage in enumerate(stages):
        passed = env.common_step_counter >= stage["step"] if inclusive else env.common_step_counter > stage["step"]
        if passed:
            resolved = index
    return resolved


def _resolve_stage(
    env: ManagerBasedRLEnv,
    stages: Sequence[dict[str, Any]],
    key: str,
    term_name: str,
    *,
    inclusive: bool,
) -> Any | None:
    """Return the payload of the last stage the run has passed, or None if it has passed none.

    See :func:`_resolve_stage_index` for the arguments and the validation it performs.
    """
    index = _resolve_stage_index(env, stages, key, term_name, inclusive=inclusive)
    return None if index is None else stages[index][key]


def reward_weight_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_name: str,
    weight_stages: Sequence[dict[str, Any]],
) -> float:
    """Ramp a reward weight through a staged schedule.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the weight is global.
        reward_name: Name of the reward term to rewrite.
        weight_stages: Stages of the schedule, each a ``{"step": int, "weight": float}`` mapping,
            ordered by strictly increasing ``"step"``. A stage applies once the global step count
            is **strictly greater** than its ``"step"``.

    Returns:
        The weight now in force.
    """
    del env_ids
    term_cfg = env.reward_manager.get_term_cfg(reward_name)
    weight = _resolve_stage(env, weight_stages, "weight", "reward_weight_stages", inclusive=False)
    if weight is not None:
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(reward_name, term_cfg)
    return term_cfg.weight


def standing_envs_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    standing_stages: Sequence[dict[str, Any]],
) -> float:
    """Ramp the fraction of environments commanded to stand still.

    The **live command term's** configuration is rewritten, not ``env.cfg``: the term samples its
    standing bucket from ``self.cfg`` on every resample, and the two objects are distinct.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the fraction is global.
        command_name: Name of the velocity command term to rewrite.
        standing_stages: Stages of the schedule, each a ``{"step": int, "rel_standing_envs": float}``
            mapping, ordered by strictly increasing ``"step"``. A stage applies once the global step
            count is **strictly greater** than its ``"step"``.

    Returns:
        The standing fraction now in force.
    """
    del env_ids
    command_cfg = env.command_manager.get_term(command_name).cfg
    fraction = _resolve_stage(env, standing_stages, "rel_standing_envs", "standing_envs_stages", inclusive=False)
    if fraction is not None:
        command_cfg.rel_standing_envs = fraction
    return command_cfg.rel_standing_envs


def command_range_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    range_stages: Sequence[dict[str, Any]],
) -> float:
    """Widen the per-dimension ranges of a pose-delta command through a staged schedule.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the ranges are global.
        command_name: Name of the
            :class:`~isaaclab_tasks.contrib.microduck.mdp.commands.ZeroablePoseDeltaCommand` term to
            rewrite.
        range_stages: Stages of the schedule, each a
            ``{"step": int, "ranges": tuple[tuple[float, float],...]}`` mapping ordered by strictly
            increasing ``"step"``. The range tuple must stay as wide as the command, which the
            command term asserts.

    Returns:
        The widest half-range currently commanded [m or rad, depending on the dimension].
    """
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
    """Widen a symmetric per-axis range of an event term through a staged schedule.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the range is global.
        event_name: Name of the event term whose parameters are rewritten.
        range_stages: Stages of the schedule, each a ``{"step": int, "range": float}`` mapping
            ordered by strictly increasing ``"step"``. The payload is the **half**-width; the range
            written out is ``(-range, +range)``. A stage applies once the global step count is
            **strictly greater** than its ``"step"``.
        param_name: Entry of the event term's ``params`` to rewrite. Defaults to ``"com_range"``,
            which is what :class:`isaaclab.envs.mdp.randomize_rigid_body_com` reads.
        range_keys: Axis keys the range is written under. Defaults to ``("x", "y", "z")``.

    Returns:
        The half-width now in force [m].
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    half_width = _resolve_stage(env, range_stages, "range", "event_range_stages", inclusive=False)
    if half_width is None:
        half_width = range_stages[0]["range"]
    event_cfg.params[param_name] = {key: (-half_width, half_width) for key in range_keys}
    return half_width


def event_param_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_name: str,
    param_stages: Sequence[dict[str, Any]],
    inclusive: bool = True,
) -> float:
    """Rewrite named parameters of an event term through a staged schedule.

    The merge is shallow, so a stage lists only the parameters it changes and the rest of the event
    term's configuration -- height bands, joint poses -- is left alone.

    :func:`event_range_stages` is the narrower sibling for the common case of one symmetric per-axis
    range; this one takes the parameter mapping itself.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the parameters are global.
        event_name: Name of the event term whose parameters are rewritten.
        param_stages: Stages of the schedule, each a ``{"step": int, "params": dict}`` mapping,
            ordered by strictly increasing ``"step"``. Before the first boundary the first stage's
            parameters apply.
        inclusive: Whether a stage triggers on the step that equals its boundary. Defaults to True,
            which is what ``event_param_curriculum`` does; its ``push_curriculum`` uses
            the exclusive comparison instead, and that inconsistency is reproduced rather than
            smoothed over (see the module note).

    Returns:
        The index of the stage now in force.
    """
    del env_ids
    event_cfg = env.event_manager.get_term_cfg(event_name)
    index = _resolve_stage_index(env, param_stages, "params", "event_param_stages", inclusive=inclusive)
    if index is None:
        index = 0
    event_cfg.params.update(param_stages[index]["params"])
    return float(index)


def termination_param_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    param_stages: Sequence[dict[str, Any]],
    inclusive: bool = True,
) -> float:
    """Rewrite named parameters of a termination term through a staged schedule.

    :func:`event_param_stages` is the same shallow merge over an event term. The two are kept apart
    rather than generalized over a manager name, so that a schedule cannot silently address the
    wrong manager's term of the same name.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the parameters are global.
        term_name: Name of the termination term whose parameters are rewritten.
        param_stages: Stages of the schedule, each a ``{"step": int, "params": dict}`` mapping,
            ordered by strictly increasing ``"step"``. Before the first boundary the first stage's
            parameters apply.
        inclusive: Whether a stage triggers on the step that equals its boundary. Defaults to True,
            which is what compares with here (see the module note).

    Returns:
        The index of the stage now in force.
    """
    del env_ids
    term_cfg = env.termination_manager.get_term_cfg(term_name)
    index = _resolve_stage_index(env, param_stages, "params", "termination_param_stages", inclusive=inclusive)
    if index is None:
        index = 0
    term_cfg.params.update(param_stages[index]["params"])
    return float(index)


def slope_move_masks(distance: torch.Tensor, size_x: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a per-environment descent distance into promotions and demotions.

    Args:
        distance: **Signed** distance [m] travelled down the slope since the reset. Shape is (N,).
        size_x: Length [m] of the sub-terrain tile along the slope.

    Returns:
        The promotion and demotion masks. Each shape is (N,).
    """
    move_up = distance > size_x * 0.4
    # Redundant on its own -- the two bands are disjoint -- but it is how the stock
    # ``terrain_levels_vel`` is written, and keeping the two diffable is worth the extra term.
    move_down = (distance < size_x * 0.2) & (~move_up)
    return move_up, move_down


def terrain_levels_slope(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Promote an environment to a steeper ramp once it has ridden its current one out.

    That distance is the **signed** x displacement, not a planar norm, which is what makes a robot
    that slid backwards up the ramp get an easier slope rather than a harder one.

    Args:
        env: The environment instance.
        env_ids: The environments being reset, whose progress is scored.
        asset_cfg: The articulation whose root link is tracked. Defaults to the robot.

    Returns:
        The mean terrain level over **all** environments, which is what the training log plots.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    distance = asset.data.root_link_pos_w.torch[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up, move_down = slope_move_masks(distance, terrain.cfg.terrain_generator.size[0])
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
