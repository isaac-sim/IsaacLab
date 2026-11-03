# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Staged curriculum terms for the MicroDuck tasks.

Every term here is a **step function over the global environment-step count**: a stage table lists
``{"step": ..., <payload>: ...}`` entries and the term applies the payload of the last stage whose
boundary the run has passed. Nothing is interpolated, exactly as upstream (reference section 6).

The stock :class:`isaaclab.envs.mdp.modify_reward_weight` is not reusable for the weight ramps: it
carries a *single* ``(num_steps, weight)`` pair, so upstream's six-stage ``action_rate_l2`` ramp
would need six curriculum terms racing to write the same reward weight, in an order the manager
does not guarantee. :func:`reward_weight_stages` keeps the whole schedule in one term, which is
also how it is written upstream.

.. note::
    Upstream compares the step counter with ``>`` in its weight, standing-fraction and
    centre-of-mass curricula but with ``>=`` in its pose-range curriculum
    (``microduck_rl/.../mdp.py:5512-5541`` versus ``:3295``, ``:3442``, ``:3464``). That
    inconsistency is reproduced verbatim rather than smoothed over, so a stage table transcribed
    from upstream schedules identically here. It is only observable on the single environment step
    that coincides with a stage boundary, because every MicroDuck stage table restates the
    configured value in its first stage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter


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

    Ported from reference section 6 (``reward_weight``). MicroDuck drives two weights with it: the
    ``action_rate_l2`` penalty from -0.1 to -1.0 over six stages, and the ``head_pose_bias``
    penalty from 0.0 to +3.0 over four.

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

    Ported from reference section 6 (``standing_envs_curriculum``). MicroDuck grows it from 2 % to
    25 % over six stages, so standing is only asked for once walking is established.

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

    Ported from reference section 6 (``pose_command_range_curriculum``). MicroDuck opens the
    head-pose command from a few centiradians to nearly its full joint range over five stages, and
    holds the body-pose command at a single stage.

    Unlike the other terms here this one triggers **inclusively**, and it falls back to the first
    stage before any boundary is reached, so the configured ranges are always replaced by the
    schedule's own first entry. Both behaviours are upstream's; see the module note.

    Args:
        env: The environment instance.
        env_ids: The environments being updated. Unused: the ranges are global.
        command_name: Name of the
            :class:`~isaaclab_tasks.contrib.microduck.pollen_mdp.commands.UniformPoseDeltaCommand` term to
            rewrite.
        range_stages: Stages of the schedule, each a
            ``{"step": int, "ranges": tuple[tuple[float, float], ...]}`` mapping ordered by strictly
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

    Ported from reference section 6 (``com_range_curriculum``). MicroDuck drives its two
    centre-of-mass randomizations with it: the trunk from +/-3 mm to +/-15 mm and the head bodies
    from +/-3 mm to +/-10 mm, so the policy meets the hardware's mass distribution only once it can
    already walk with the nominal one.

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

    Ported from addendum section 3.8 (``event_param_curriculum`` and ``push_curriculum``, which are
    the same shallow merge under two names and two comparison operators). The stand-up task drives
    two schedules with it: the four probabilities of its ground-state reset, which ramp from a mix
    dominated by standing and sitting to one dominated by the prone poses, and the magnitude of its
    push disturbance.

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
            which is what upstream's ``event_param_curriculum`` does; its ``push_curriculum`` uses
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

    Ported from addendum section 3.6 (``termination_param_curriculum``). The hybrid
    walking-and-recovery task drives one schedule with it, and that schedule is the task's central
    design: the tilt termination that ends a fall is active while the walk is learned and is then
    *widened to half a turn*, so a fall stops ending the episode and starts being a recovery to
    train on. Deleting the term instead would change the episode's termination set mid-run;
    widening its bound leaves the manager's shape alone.

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
            which is what upstream compares with here (see the module note).

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

    Ported from addendum section 4.4 (``slope_move_masks``). An environment that travelled more than
    40 percent of the tile has ridden the ramp out and earns a steeper one; one that travelled less
    than 20 percent fell or stalled near the top and gets a gentler one; the band between them holds.

    Note:
        Upstream's docstring calibrates the two fractions against an 8 m tile, and the task it serves
        runs a 15 m one -- so the live thresholds are 6.0 m and 3.0 m, not 3.2 m and 1.6 m (addendum
        section 9.4). The fractions are reproduced because they are what the reference policy trained
        against.

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

    Ported from addendum section 4.4 (``terrain_levels_slope``). The stock
    :func:`~isaaclab_tasks.core.velocity.mdp.terrain_levels_vel` cannot be reused: it demotes against
    the distance a *commanded* velocity should have covered, and this task's twist command is
    neutralized to zero, so every environment would be demoted on every reset. Progress is measured
    instead as the raw distance down the slope.

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
