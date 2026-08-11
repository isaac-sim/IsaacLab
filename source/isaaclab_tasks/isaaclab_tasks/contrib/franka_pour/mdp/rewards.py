# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Bounded progress and outcome rewards for grasping and pouring MPM media."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils

from .terminations import source_grasp_milestones

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def finite_joint_velocity_l2(
    env: FrankaPourEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_velocity: float = 20.0,
) -> torch.Tensor:
    """Return a bounded, finite joint-velocity penalty.

    Non-finite velocities receive the maximum penalty so numerical failures cannot evade this term,
    while clamping keeps the terminal transition consumable by PPO.
    """
    velocity = env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = torch.nan_to_num(
        velocity,
        nan=float(max_velocity),
        posinf=float(max_velocity),
        neginf=-float(max_velocity),
    )
    velocity = torch.clamp(velocity, min=-float(max_velocity), max=float(max_velocity))
    return torch.sum(torch.square(velocity), dim=-1)


def tcp_cup_distance_tanh(env: FrankaPourEnv, std: float = 1.0) -> torch.Tensor:
    """Return bounded TCP-to-source-grasp proximity using distance scale ``std`` [m]."""
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive.")
    distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    distance = torch.where(torch.isfinite(distance), distance, torch.full_like(distance, torch.inf))
    return 1.0 - torch.tanh(distance / float(std))


def media_target_distance_tanh(env: FrankaPourEnv, std: float = 1.0) -> torch.Tensor:
    """Return mean particle proximity to the source-exclusive target set.

    Distance to the receiving cavity supplies transport guidance. For particles still in the
    source, distance to its open rim prevents nesting the loaded source inside the receiver from
    scoring as delivery. Particles that have irreversibly reached the spill plane receive no
    credit.
    """
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError("std must be finite and positive.")
    target_pose = env.target_pose_e()
    source_pose = env.cup_pose_e()
    particle_position = env.particle_pos_e()
    target_quat = target_pose[:, None, 3:7].expand(-1, particle_position.shape[1], -1)
    particle_target = math_utils.quat_apply_inverse(
        target_quat,
        particle_position - target_pose[:, None, :3],
    )
    source_quat = source_pose[:, None, 3:7].expand_as(target_quat)
    particle_source = math_utils.quat_apply_inverse(
        source_quat,
        particle_position - source_pose[:, None, :3],
    )
    margin = float(env.cfg.particle_count_margin)
    lower = env._target_inner_lo_t - margin
    upper = env._target_inner_hi_t + margin
    outside = torch.maximum(lower - particle_target, particle_target - upper).clamp_(min=0.0)
    target_distance = torch.linalg.vector_norm(outside, dim=-1)
    in_source, _, spilled = env.particle_region_masks()
    source_exit_distance = torch.where(
        in_source,
        torch.clamp(env._source_inner_hi_t[2] + margin - particle_source[..., 2], min=0.0),
        torch.zeros_like(target_distance),
    )
    goal_distance = torch.maximum(target_distance, source_exit_distance)
    valid = torch.isfinite(goal_distance) & ~spilled
    quality = torch.where(valid, 1.0 - torch.tanh(goal_distance / float(std)), 0.0)
    return quality.mean(dim=1)


def terminal_failure(env: FrankaPourEnv, include_time_out: bool = True) -> torch.Tensor:
    """Return one unit-integral pulse for an unsuccessful terminal transition."""
    if not isinstance(include_time_out, bool):
        raise TypeError("include_time_out must be a bool.")
    success = _success_terminal(env)
    completed = env.termination_manager.dones if include_time_out else env.termination_manager.terminated
    failed = completed & ~success
    # RewardManager multiplies terms by step_dt. Dividing here makes the configured weight the
    # exact one-time episode penalty, independent of the control frequency.
    return failed.float() / max(float(env.step_dt), 1.0e-6)


def _reach_quality(env: FrankaPourEnv, std: float) -> torch.Tensor:
    distance = torch.linalg.norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    return 1.0 - torch.tanh(distance / max(float(std), 1.0e-6))


def _open_quality(env: FrankaPourEnv) -> torch.Tensor:
    travel = max(float(env.gripper_open_width) - float(env.gripper_grasp_width), 1.0e-4)
    return torch.clamp((env.gripper_width() - float(env.gripper_grasp_width)) / travel, 0.0, 1.0)


def _approach_potential(
    env: FrankaPourEnv,
    position_std: float,
    orientation_std: float,
    open_hand_fraction: float,
) -> torch.Tensor:
    """Return bounded grasp-pose approach with recoverable open-hand coordination."""
    if not math.isfinite(position_std) or position_std <= 0.0:
        raise ValueError("position_std must be finite and positive.")
    if not math.isfinite(orientation_std) or orientation_std <= 0.0:
        raise ValueError("orientation_std must be finite and positive.")
    if not math.isfinite(open_hand_fraction) or not 0.0 <= open_hand_fraction <= 1.0:
        raise ValueError("open_hand_fraction must lie in [0, 1].")

    distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    position_quality = 1.0 - torch.tanh(distance / float(position_std))

    cup_quat = env.cup_pose_e()[:, 3:7]
    desired_quat = math_utils.quat_mul(cup_quat, env.desired_grasp_tcp_quat_c())
    tcp_quat = env.tcp_pose_e()[:, 3:7]
    error_quat = math_utils.quat_mul(math_utils.quat_conjugate(desired_quat), tcp_quat)
    error_quat = math_utils.quat_unique(error_quat)
    orientation_error = torch.linalg.vector_norm(math_utils.axis_angle_from_quat(error_quat), dim=-1)
    orientation_quality = 1.0 - torch.tanh(orientation_error / float(orientation_std))

    # Retain half of the Cartesian gradient even under a poor tool orientation. Multiplying two
    # narrow kernels would make an independently sampled arm receive almost no useful reach signal.
    pose_quality = position_quality * (0.5 + 0.5 * orientation_quality)
    # Opening is useful while approaching, but its bonus fades to zero at the grasp pose. Closing
    # prematurely therefore loses potential without erasing the position/orientation gradient that
    # lets the policy recover and finish the approach.
    potential = pose_quality + float(open_hand_fraction) * (1.0 - pose_quality) * _open_quality(env)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _grasp_width_quality(env: FrankaPourEnv, preload_position: float) -> torch.Tensor:
    """Return grasp quality requiring commanded preload and non-empty physical closure."""
    width = env.gripper_width()
    open_width = float(env.gripper_open_width)
    grasp_width = float(env.gripper_grasp_width)
    travel = max(open_width - grasp_width, 1.0e-4)
    close_progress = torch.clamp((open_width - width) / travel, 0.0, 1.0)
    # Contact can settle slightly inside the nominal geometry, but an empty fully closed hand
    # must not look grasped. This factor stays one throughout the intended open-to-contact path
    # and falls linearly only after the fingers move inside the nominal cup width.
    not_empty = torch.clamp(width / max(grasp_width, 1.0e-4), 0.0, 1.0)
    open_position = 0.5 * open_width
    preload_travel = max(open_position - float(preload_position), 1.0e-4)
    gripper = env.action_manager.get_term("gripper_action")
    command = gripper.commanded_position[:, 0]
    commanded_preload = torch.clamp((open_position - command) / preload_travel, 0.0, 1.0)
    # ``minimum`` is a conjunctive crossfade without the quadratic dead zone produced by a
    # product: coordinated physical closure and commanded preload remain monotonic from open to
    # contact, while either an open command or an empty fully closed hand still yields zero.
    contact_quality = getattr(gripper, "contact_quality", not_empty)
    return torch.minimum(close_progress, commanded_preload) * contact_quality


def _grasp_lift_potential(
    env: FrankaPourEnv,
    target_height: float,
    grasp_reach_std: float,
    grasp_preload_position: float,
    grasp_fraction: float,
) -> torch.Tensor:
    """Return bounded contact-qualified grasp and lift completion in ``[0, 1]``."""
    if not math.isfinite(target_height) or target_height <= 0.0:
        raise ValueError("target_height must be finite and positive.")
    if not math.isfinite(grasp_reach_std) or grasp_reach_std <= 0.0:
        raise ValueError("grasp_reach_std must be finite and positive.")
    if not math.isfinite(grasp_preload_position):
        raise ValueError("grasp_preload_position must be finite.")
    if not math.isfinite(grasp_fraction) or not 0.0 <= grasp_fraction <= 1.0:
        raise ValueError("grasp_fraction must lie in [0, 1].")

    distance = torch.linalg.vector_norm(env.tcp_pos_e() - env.cup_grasp_point_e(), dim=-1)
    proximity = torch.clamp(1.0 - distance / float(grasp_reach_std), 0.0, 1.0)
    # Compact-support smoothstep prevents closing an empty hand at stand-off from earning grasp
    # credit. The broader approach potential supplies the gradient until this contact neighborhood.
    proximity = proximity.square() * (3.0 - 2.0 * proximity)
    grasp = proximity * _grasp_width_quality(env, grasp_preload_position)
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / float(target_height),
        0.0,
        1.0,
    )
    potential = grasp * (float(grasp_fraction) + (1.0 - float(grasp_fraction)) * height)
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _lift_potential(
    env: FrankaPourEnv,
    target_height: float,
    reach_std: float,
    grasp_reach_std: float,
    grasp_preload_position: float,
    approach_fraction: float,
    grasp_fraction: float,
) -> torch.Tensor:
    """Return bounded open-approach, contact-grasp, and lift completion in ``[0, 1]``."""
    if approach_fraction < 0.0 or grasp_fraction < 0.0 or approach_fraction + grasp_fraction > 1.0:
        raise ValueError("approach_fraction and grasp_fraction must be nonnegative and sum to at most one.")
    reach = _reach_quality(env, reach_std)
    # Width alone cannot distinguish a real grasp from an empty hand closed to the cup width.
    # Gate grasp and lift completion much more sharply than the broad approach shaping so that
    # closing at a stand-off pose always loses potential, while closing at the grasp point gains it.
    grasp_reach = _reach_quality(env, grasp_reach_std)
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / max(float(target_height), 1.0e-6),
        0.0,
        1.0,
    )
    grasp = _grasp_width_quality(env, grasp_preload_position)
    lift_fraction = 1.0 - float(approach_fraction) - float(grasp_fraction)
    potential = (
        float(approach_fraction) * reach * _open_quality(env)
        + float(grasp_fraction) * grasp_reach * grasp
        + lift_fraction * height * grasp_reach * grasp
    )
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _align_potential(
    env: FrankaPourEnv,
    lift_height: float,
    std: float,
    source_offset_xy: Sequence[float],
    grasp_reach_std: float,
    grasp_preload_position: float,
) -> torch.Tensor:
    """Return bounded held-source-to-receiver alignment in ``[0, 1]``."""
    if len(source_offset_xy) != 2 or any(not math.isfinite(value) for value in source_offset_xy):
        raise ValueError("source_offset_xy must contain two finite values.")
    cup = env.cup_pose_e()[:, :3]
    grasp_point = env.cup_grasp_point_e()
    target = env.target_pose_e()[:, :3]
    lifted = torch.clamp((cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6), 0.0, 1.0)
    desired_grasp_xy = target[:, :2] + cup.new_tensor(source_offset_xy)
    distance_xy = torch.linalg.norm(grasp_point[:, :2] - desired_grasp_xy, dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(std), 1.0e-6))
    grasp = _grasp_width_quality(env, grasp_preload_position)
    grasp_reach = _reach_quality(env, grasp_reach_std)
    potential = lifted * aligned * grasp_reach * grasp
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _pour_tilt_potential(
    env: FrankaPourEnv,
    target_tilt: float,
    pour_direction_xy: Sequence[float],
    source_mouth_height: float,
    alignment_radius: float,
    active_through_stage: int,
    min_lift_height: float,
    max_tcp_distance: float,
    max_gripper_width_error: float,
    max_gripper_command: float,
) -> torch.Tensor:
    """Return bounded early-stage progress toward the authored physical pour in ``[0, 1]``."""
    if not math.isfinite(target_tilt) or not 0.0 < target_tilt < math.pi:
        raise ValueError(f"target_tilt must lie in (0, pi), got {target_tilt}.")
    if len(pour_direction_xy) != 2 or any(not math.isfinite(value) for value in pour_direction_xy):
        raise ValueError("pour_direction_xy must contain two finite values.")
    direction_norm = math.hypot(float(pour_direction_xy[0]), float(pour_direction_xy[1]))
    if direction_norm <= 0.0:
        raise ValueError("pour_direction_xy must be nonzero.")
    if not math.isfinite(source_mouth_height) or source_mouth_height < 0.0:
        raise ValueError(f"source_mouth_height must be finite and nonnegative, got {source_mouth_height}.")
    if not math.isfinite(alignment_radius) or alignment_radius <= 0.0:
        raise ValueError(f"alignment_radius must be finite and positive, got {alignment_radius}.")
    if active_through_stage < 0:
        raise ValueError(f"active_through_stage must be nonnegative, got {active_through_stage}.")

    cup_pose = env.cup_pose_e()
    target_pose = env.target_pose_e()
    local_open_axis = torch.zeros_like(cup_pose[:, :3])
    local_open_axis[:, 2] = 1.0
    open_axis = math_utils.quat_apply(cup_pose[:, 3:7], local_open_axis)
    direction_x = float(pour_direction_xy[0]) / direction_norm
    direction_y = float(pour_direction_xy[1]) / direction_norm
    directed_opening = open_axis[:, 0] * direction_x + open_axis[:, 1] * direction_y
    # ``atan2`` remains monotonic after the rim passes horizontal, unlike a sine projection. This
    # lets the curriculum teach the validated deep drain while rejecting rotation away from
    # the receiver and sideways inversion.
    directed_angle = torch.atan2(torch.clamp(directed_opening, min=0.0), open_axis[:, 2])
    directed_angle = torch.where(directed_opening > 0.0, directed_angle, 0.0)
    tilt = torch.clamp(directed_angle / float(target_tilt), 0.0, 1.0)

    local_mouth = torch.zeros_like(cup_pose[:, :3])
    local_mouth[:, 2] = float(source_mouth_height)
    mouth_position = cup_pose[:, :3] + math_utils.quat_apply(cup_pose[:, 3:7], local_mouth)
    distance_xy = torch.linalg.vector_norm(mouth_position[:, :2] - target_pose[:, :2], dim=-1)
    proximity = torch.clamp(1.0 - distance_xy / float(alignment_radius), 0.0, 1.0)
    # Compact-support smoothstep prevents the carry stage from earning tilt credit before the cup
    # reaches the receiver, while avoiding a discontinuity at the alignment boundary.
    aligned = proximity.square() * (3.0 - 2.0 * proximity)
    _, _, held_pour = source_grasp_milestones(
        env,
        min_lift_height=min_lift_height,
        max_tcp_distance=max_tcp_distance,
        max_gripper_width_error=max_gripper_width_error,
        max_gripper_command=max_gripper_command,
    )
    active_stage = env.curriculum_stage <= int(active_through_stage)
    potential = tilt * aligned * held_pour.float() * active_stage.float()
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


def _pour_reference_potential(
    env: FrankaPourEnv,
    start_q: Sequence[float],
    target_q: Sequence[float],
    active_stage: int,
    min_lift_height: float,
    max_tcp_distance: float,
    max_gripper_width_error: float,
    max_gripper_command: float,
) -> torch.Tensor:
    """Return held progress along the validated stage-zero arm trajectory in ``[0, 1]``."""
    arm_q = env.arm_joint_pos()
    if len(start_q) != arm_q.shape[1] or len(target_q) != arm_q.shape[1]:
        raise ValueError(f"start_q and target_q must each contain {arm_q.shape[1]} joint positions.")
    if any(not math.isfinite(value) for value in (*start_q, *target_q)):
        raise ValueError("start_q and target_q must contain finite joint positions.")
    reference_distance = math.sqrt(sum((target - start) ** 2 for start, target in zip(start_q, target_q, strict=True)))
    if reference_distance <= 0.0:
        raise ValueError("start_q and target_q must be distinct.")
    target = arm_q.new_tensor(target_q)
    distance = torch.linalg.vector_norm(arm_q - target, dim=-1)
    progress = torch.clamp(1.0 - distance / reference_distance, 0.0, 1.0)
    _, _, held_pour = source_grasp_milestones(
        env,
        min_lift_height=min_lift_height,
        max_tcp_distance=max_tcp_distance,
        max_gripper_width_error=max_gripper_width_error,
        max_gripper_command=max_gripper_command,
    )
    active = env.curriculum_stage == int(active_stage)
    potential = progress * held_pour.float() * active.float()
    return torch.nan_to_num(potential, nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)


class _SignedPotentialProgress(ManagerTermBase):
    """Track independent per-environment potential history."""

    def __init__(self, cfg: RewardTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)
        self._initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    def _reset_potential(
        self,
        potential: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_potential[env_ids] = potential[env_ids]
        self._initialized[env_ids] = True

    def _signed_progress(self, potential: torch.Tensor, step_dt: float) -> torch.Tensor:
        progress = torch.where(self._initialized, potential - self._previous_potential, 0.0)
        self._previous_potential.copy_(potential)
        self._initialized.fill_(True)
        return progress / max(float(step_dt), 1.0e-6)

    def _discounted_progress(
        self,
        potential: torch.Tensor,
        step_dt: float,
        discount_factor: float,
        terminal: torch.Tensor,
    ) -> torch.Tensor:
        """Return policy-invariant discounted potential shaping.

        Treating the post-transition potential as zero on terminal states preserves the standard
        episodic potential-shaping convention. A forward/reverse cycle then has exactly the same
        discounted shaping return as holding the original state for the same duration.
        """
        if not 0.0 < float(discount_factor) <= 1.0:
            raise ValueError("discount_factor must lie in (0, 1].")
        next_potential = torch.where(terminal, torch.zeros_like(potential), potential)
        progress = torch.where(
            self._initialized,
            float(discount_factor) * next_potential - self._previous_potential,
            torch.zeros_like(potential),
        )
        self._previous_potential.copy_(potential)
        self._initialized.fill_(True)
        return progress / max(float(step_dt), 1.0e-6)


class PourTaskProgress(_SignedPotentialProgress):
    """Signed progress through approach, grasp, lift, align, and tilt milestones.

    The single ordered physical potential replaces independently weighted, stage-switched shaping
    terms. Its episode integral telescopes, so holding an intermediate pose earns nothing and
    reversing or dropping the cup repays the corresponding forward progress.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Start progress accounting from the physical state supplied by the reset."""
        params = dict(self.cfg.params or {})
        params.pop("discount_factor", None)
        potential = self._potential(self._env, **params)
        self._reset_potential(potential, env_ids)

    @staticmethod
    def _potential(
        env: FrankaPourEnv,
        target_height: float,
        reach_std: float,
        grasp_reach_std: float,
        grasp_preload_position: float,
        lift_height: float,
        align_std: float,
        source_offset_xy: Sequence[float],
        target_tilt: float,
        pour_direction_xy: Sequence[float],
        source_mouth_height: float,
        alignment_radius: float,
        active_through_stage: int,
        min_lift_height: float,
        max_tcp_distance: float,
        max_gripper_width_error: float,
        max_gripper_command: float,
    ) -> torch.Tensor:
        approach_grasp_lift = _lift_potential(
            env,
            target_height=target_height,
            reach_std=reach_std,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
            approach_fraction=0.20,
            grasp_fraction=0.30,
        )
        align = _align_potential(
            env,
            lift_height=lift_height,
            std=align_std,
            source_offset_xy=source_offset_xy,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
        )
        tilt = _pour_tilt_potential(
            env,
            target_tilt=target_tilt,
            pour_direction_xy=pour_direction_xy,
            source_mouth_height=source_mouth_height,
            alignment_radius=alignment_radius,
            active_through_stage=active_through_stage,
            min_lift_height=min_lift_height,
            max_tcp_distance=max_tcp_distance,
            max_gripper_width_error=max_gripper_width_error,
            max_gripper_command=max_gripper_command,
        )
        return 0.45 * approach_grasp_lift + 0.20 * align + 0.35 * tilt

    def __call__(
        self,
        env: FrankaPourEnv,
        target_height: float,
        reach_std: float,
        grasp_reach_std: float,
        grasp_preload_position: float,
        lift_height: float,
        align_std: float,
        source_offset_xy: Sequence[float],
        target_tilt: float,
        pour_direction_xy: Sequence[float],
        source_mouth_height: float,
        alignment_radius: float,
        active_through_stage: int,
        min_lift_height: float,
        max_tcp_distance: float,
        max_gripper_width_error: float,
        max_gripper_command: float,
        discount_factor: float,
    ) -> torch.Tensor:
        potential = self._potential(
            env,
            target_height=target_height,
            reach_std=reach_std,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
            lift_height=lift_height,
            align_std=align_std,
            source_offset_xy=source_offset_xy,
            target_tilt=target_tilt,
            pour_direction_xy=pour_direction_xy,
            source_mouth_height=source_mouth_height,
            alignment_radius=alignment_radius,
            active_through_stage=active_through_stage,
            min_lift_height=min_lift_height,
            max_tcp_distance=max_tcp_distance,
            max_gripper_width_error=max_gripper_width_error,
            max_gripper_command=max_gripper_command,
        )
        # The five-second manipulation attempt is a finite-horizon episode. Close the shaping
        # potential on both task terminations and the deadline so its discounted return remains
        # policy-invariant and failed timeouts cannot retain bootstrap value.
        terminal = env.termination_manager.dones
        return self._discounted_progress(
            potential,
            env.step_dt,
            discount_factor=discount_factor,
            terminal=terminal,
        )


class ApproachProgress(_SignedPotentialProgress):
    """Discounted progress toward an open, correctly oriented source-cup grasp pose."""

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Start progress accounting from each environment's randomized arm pose."""
        params = dict(self.cfg.params or {})
        params.pop("discount_factor", None)
        active_from_stage = int(params.pop("active_from_stage"))
        potential = _approach_potential(self._env, **params)
        potential *= (self._env.curriculum_stage >= active_from_stage).float()
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        position_std: float,
        orientation_std: float,
        open_hand_fraction: float,
        active_from_stage: int,
        discount_factor: float,
    ) -> torch.Tensor:
        potential = _approach_potential(
            env,
            position_std=position_std,
            orientation_std=orientation_std,
            open_hand_fraction=open_hand_fraction,
        )
        potential *= (env.curriculum_stage >= int(active_from_stage)).float()
        return self._discounted_progress(
            potential,
            env.step_dt,
            discount_factor=discount_factor,
            terminal=env.termination_manager.dones,
        )


class GraspLiftProgress(_SignedPotentialProgress):
    """Discounted progress from near-contact closure through lifting the source cup."""

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Start progress accounting from each environment's reset grasp state."""
        params = dict(self.cfg.params or {})
        params.pop("discount_factor", None)
        active_from_stage = int(params.pop("active_from_stage"))
        potential = _grasp_lift_potential(self._env, **params)
        potential *= (self._env.curriculum_stage >= active_from_stage).float()
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        target_height: float,
        grasp_reach_std: float,
        grasp_preload_position: float,
        grasp_fraction: float,
        active_from_stage: int,
        discount_factor: float,
    ) -> torch.Tensor:
        potential = _grasp_lift_potential(
            env,
            target_height=target_height,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
            grasp_fraction=grasp_fraction,
        )
        potential *= (env.curriculum_stage >= int(active_from_stage)).float()
        return self._discounted_progress(
            potential,
            env.step_dt,
            discount_factor=discount_factor,
            terminal=env.termination_manager.dones,
        )


class LiftProgress(_SignedPotentialProgress):
    """Signed change in bounded grasp-and-lift completion from the lift stage onward.

    The term is divided by the environment step interval because
    :class:`~isaaclab.managers.RewardManager` multiplies every reward by that interval. Its integrated
    episode contribution therefore telescopes: reversing a lift repays the earlier positive reward.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        params = self.cfg.params or {}
        potential = _lift_potential(
            self._env,
            target_height=float(params.get("target_height", 0.12)),
            reach_std=float(params.get("reach_std", 0.07)),
            grasp_reach_std=float(params.get("grasp_reach_std", 0.015)),
            grasp_preload_position=float(params.get("grasp_preload_position", 0.025)),
            approach_fraction=float(params.get("approach_fraction", 0.2)),
            grasp_fraction=float(params.get("grasp_fraction", 0.3)),
        )
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        target_height: float = 0.12,
        reach_std: float = 0.07,
        grasp_reach_std: float = 0.015,
        grasp_preload_position: float = 0.025,
        approach_fraction: float = 0.2,
        grasp_fraction: float = 0.3,
    ) -> torch.Tensor:
        potential = _lift_potential(
            env,
            target_height=target_height,
            reach_std=reach_std,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
            approach_fraction=approach_fraction,
            grasp_fraction=grasp_fraction,
        )
        progress = self._signed_progress(potential, env.step_dt)
        return progress * (env.curriculum_stage >= 2)


class AlignProgress(_SignedPotentialProgress):
    """Signed change in held-source-to-receiver alignment from the carry stage onward.

    Returning to an earlier alignment produces the exact negative of the corresponding forward
    progress, preventing cyclic motion from accumulating reward.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        params = self.cfg.params or {}
        potential = _align_potential(
            self._env,
            lift_height=float(params.get("lift_height", 0.06)),
            std=float(params.get("std", 0.12)),
            source_offset_xy=params.get("source_offset_xy", (0.0, 0.0)),
            grasp_reach_std=float(params.get("grasp_reach_std", 0.015)),
            grasp_preload_position=float(params.get("grasp_preload_position", 0.025)),
        )
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        lift_height: float = 0.06,
        std: float = 0.12,
        source_offset_xy: Sequence[float] = (0.0, 0.0),
        grasp_reach_std: float = 0.015,
        grasp_preload_position: float = 0.025,
    ) -> torch.Tensor:
        potential = _align_potential(
            env,
            lift_height=lift_height,
            std=std,
            source_offset_xy=source_offset_xy,
            grasp_reach_std=grasp_reach_std,
            grasp_preload_position=grasp_preload_position,
        )
        progress = self._signed_progress(potential, env.step_dt)
        return progress * (env.curriculum_stage >= 1)


class PourTiltProgress(_SignedPotentialProgress):
    """Signed early-curriculum progress toward tilting a held cup over the receiver.

    This term is active only through the configured curriculum stage. Its episode integral
    telescopes, so reversing a tilt or releasing the cup repays the corresponding positive reward
    and cyclic wiggling cannot accumulate return.
    """

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        params = self.cfg.params or {}
        potential = _pour_tilt_potential(
            self._env,
            target_tilt=float(params.get("target_tilt", math.radians(150.0))),
            pour_direction_xy=params.get("pour_direction_xy", (0.0, -1.0)),
            source_mouth_height=float(params.get("source_mouth_height", 0.0)),
            alignment_radius=float(params.get("alignment_radius", 0.10)),
            active_through_stage=int(params.get("active_through_stage", 1)),
            min_lift_height=float(params.get("min_lift_height", 0.05)),
            max_tcp_distance=float(params.get("max_tcp_distance", 0.015)),
            max_gripper_width_error=float(params.get("max_gripper_width_error", 0.012)),
            max_gripper_command=float(params.get("max_gripper_command", 0.025)),
        )
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        target_tilt: float = math.radians(150.0),
        pour_direction_xy: Sequence[float] = (0.0, -1.0),
        source_mouth_height: float = 0.0,
        alignment_radius: float = 0.10,
        active_through_stage: int = 1,
        min_lift_height: float = 0.05,
        max_tcp_distance: float = 0.015,
        max_gripper_width_error: float = 0.012,
        max_gripper_command: float = 0.025,
    ) -> torch.Tensor:
        potential = _pour_tilt_potential(
            env,
            target_tilt=target_tilt,
            pour_direction_xy=pour_direction_xy,
            source_mouth_height=source_mouth_height,
            alignment_radius=alignment_radius,
            active_through_stage=active_through_stage,
            min_lift_height=min_lift_height,
            max_tcp_distance=max_tcp_distance,
            max_gripper_width_error=max_gripper_width_error,
            max_gripper_command=max_gripper_command,
        )
        return self._signed_progress(potential, env.step_dt)


class PourReferenceProgress(_SignedPotentialProgress):
    """Signed stage-zero progress along the validated held-pour joint trajectory."""

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        params = self.cfg.params or {}
        potential = _pour_reference_potential(
            self._env,
            start_q=params.get("start_q", ()),
            target_q=params.get("target_q", ()),
            active_stage=int(params.get("active_stage", 0)),
            min_lift_height=float(params.get("min_lift_height", 0.05)),
            max_tcp_distance=float(params.get("max_tcp_distance", 0.015)),
            max_gripper_width_error=float(params.get("max_gripper_width_error", 0.012)),
            max_gripper_command=float(params.get("max_gripper_command", 0.025)),
        )
        self._reset_potential(potential, env_ids)

    def __call__(
        self,
        env: FrankaPourEnv,
        start_q: Sequence[float],
        target_q: Sequence[float],
        active_stage: int = 0,
        min_lift_height: float = 0.05,
        max_tcp_distance: float = 0.015,
        max_gripper_width_error: float = 0.012,
        max_gripper_command: float = 0.025,
    ) -> torch.Tensor:
        potential = _pour_reference_potential(
            env,
            start_q=start_q,
            target_q=target_q,
            active_stage=active_stage,
            min_lift_height=min_lift_height,
            max_tcp_distance=max_tcp_distance,
            max_gripper_width_error=max_gripper_width_error,
            max_gripper_command=max_gripper_command,
        )
        return self._signed_progress(potential, env.step_dt)


def _warn_legacy_reward(name: str) -> None:
    warnings.warn(
        f"mdp.{name} is deprecated; use the bounded progress and outcome terms from FrankaPourEnvCfg instead.",
        DeprecationWarning,
        stacklevel=2,
    )


def _legacy_closure_quality(env: FrankaPourEnv) -> torch.Tensor:
    travel = max(float(env.gripper_open_width) - float(env.gripper_grasp_width), 1.0e-4)
    return torch.clamp((float(env.gripper_open_width) - env.gripper_width()) / travel, 0.0, 1.0)


def reach_cup(env: FrankaPourEnv, std: float = 0.10) -> torch.Tensor:
    """Deprecated Cartesian reach reward retained for downstream compatibility."""
    _warn_legacy_reward("reach_cup")
    return _reach_quality(env, std)


def grasp_cup(env: FrankaPourEnv, reach_std: float = 0.06) -> torch.Tensor:
    """Deprecated grasp reward retained for downstream compatibility."""
    _warn_legacy_reward("grasp_cup")
    return _reach_quality(env, reach_std) * _legacy_closure_quality(env)


def lift_cup(env: FrankaPourEnv, target_height: float = 0.12, reach_std: float = 0.07) -> torch.Tensor:
    """Deprecated lift reward retained for downstream compatibility."""
    _warn_legacy_reward("lift_cup")
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / max(float(target_height), 1.0e-6),
        0.0,
        1.0,
    )
    return height * _reach_quality(env, reach_std) * _legacy_closure_quality(env)


def lift_command_progress(
    env: FrankaPourEnv,
    target_height: float = 0.12,
    reach_std: float = 0.07,
) -> torch.Tensor:
    """Deprecated Cartesian lift-command reward retained for downstream compatibility."""
    _warn_legacy_reward("lift_command_progress")
    height = torch.clamp(
        (env.cup_pose_e()[:, 2] - float(env.cup_reset_height)) / max(float(target_height), 1.0e-6),
        0.0,
        1.0,
    )
    upward = torch.clamp(env.action_manager.action[:, 2], 0.0, 1.0)
    grasp = _reach_quality(env, reach_std) * _legacy_closure_quality(env)
    return grasp * (1.0 - height) * upward


def align_cup_over_target(
    env: FrankaPourEnv,
    lift_height: float = 0.06,
    std: float = 0.12,
) -> torch.Tensor:
    """Deprecated source-alignment reward retained for downstream compatibility."""
    _warn_legacy_reward("align_cup_over_target")
    cup = env.cup_pose_e()[:, :3]
    target = env.target_pose_e()[:, :3]
    lifted = torch.clamp(
        (cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6),
        0.0,
        1.0,
    )
    distance_xy = torch.linalg.vector_norm(cup[:, :2] - target[:, :2], dim=-1)
    return lifted * (1.0 - torch.tanh(distance_xy / max(float(std), 1.0e-6)))


def align_command_progress(
    env: FrankaPourEnv,
    lift_height: float = 0.06,
    std: float = 0.12,
) -> torch.Tensor:
    """Deprecated Cartesian alignment-command reward retained for downstream compatibility."""
    _warn_legacy_reward("align_command_progress")
    cup = env.cup_pose_e()[:, :3]
    target = env.target_pose_e()[:, :3]
    lifted = torch.clamp(
        (cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6),
        0.0,
        1.0,
    )
    delta_xy = target[:, :2] - cup[:, :2]
    distance = torch.linalg.vector_norm(delta_xy, dim=-1)
    direction = delta_xy / torch.clamp(distance[:, None], min=1.0e-6)
    toward = torch.clamp(torch.sum(env.action_manager.action[:, :2] * direction, dim=-1), 0.0, 1.0)
    return lifted * torch.tanh(distance / max(float(std), 1.0e-6)) * toward


def _legacy_cup_up_z(env: FrankaPourEnv) -> torch.Tensor:
    quat = env.cup_pose_e()[:, 3:7]
    up = torch.zeros((quat.shape[0], 3), device=quat.device, dtype=quat.dtype)
    up[:, 2] = 1.0
    xyz = quat[:, :3]
    cross = 2.0 * torch.cross(xyz, up, dim=-1)
    rotated = up + quat[:, 3:4] * cross + torch.cross(xyz, cross, dim=-1)
    return rotated[:, 2]


def tilt_over_target(
    env: FrankaPourEnv,
    lift_height: float = 0.06,
    align_std: float = 0.10,
) -> torch.Tensor:
    """Deprecated source-tilt reward retained for downstream compatibility."""
    _warn_legacy_reward("tilt_over_target")
    cup = env.cup_pose_e()[:, :3]
    lifted = torch.clamp(
        (cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6),
        0.0,
        1.0,
    )
    distance_xy = torch.linalg.vector_norm(cup[:, :2] - env.target_pose_e()[:, :2], dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(align_std), 1.0e-6))
    tilt = torch.clamp(
        (math.cos(math.pi / 3.0) - _legacy_cup_up_z(env)) / math.cos(math.pi / 3.0),
        0.0,
        1.0,
    )
    return lifted * aligned * tilt


def tilt_command_progress(
    env: FrankaPourEnv,
    lift_height: float = 0.06,
    align_std: float = 0.10,
) -> torch.Tensor:
    """Deprecated Cartesian tilt-command reward retained for downstream compatibility."""
    _warn_legacy_reward("tilt_command_progress")
    cup = env.cup_pose_e()[:, :3]
    lifted = torch.clamp(
        (cup[:, 2] - float(env.cup_reset_height)) / max(float(lift_height), 1.0e-6),
        0.0,
        1.0,
    )
    distance_xy = torch.linalg.vector_norm(cup[:, :2] - env.target_pose_e()[:, :2], dim=-1)
    aligned = 1.0 - torch.tanh(distance_xy / max(float(align_std), 1.0e-6))
    tilt = torch.clamp(
        (math.cos(math.pi / 3.0) - _legacy_cup_up_z(env)) / math.cos(math.pi / 3.0),
        0.0,
        1.0,
    )
    rotate_toward_pour = torch.clamp(env.action_manager.action[:, 3], 0.0, 1.0)
    return lifted * aligned * (1.0 - tilt) * rotate_toward_pour


def particles_in_target(env: FrankaPourEnv) -> torch.Tensor:
    return env.count_in_target() / max(env.num_particles, 1)


def particles_in_source(env: FrankaPourEnv) -> torch.Tensor:
    return env.count_in_source() / max(env.num_particles, 1)


def _success_terminal(env: FrankaPourEnv) -> torch.Tensor:
    """Return this step's managed success, or false when a replay tool removed that term."""
    if "success" not in env.termination_manager.active_terms:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    return env.termination_manager.get_term("success")


class HeldDeliveryProgress(ManagerTermBase):
    """Reward signed progress toward the active held-delivery target.

    Credit is capped at each environment's success threshold, decreases when qualified particles
    leave the receiver, and is fully clawed back when an episode ends without success. A successful
    terminal transition retains its final credit because the separate success term rewards the same
    predicate used by curriculum progression.
    """

    def __init__(self, cfg: RewardTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        self._previous_credit = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous_credit[env_ids] = 0.0

    def __call__(
        self,
        env: FrankaPourEnv,
        min_lift_height: float = 0.05,
        max_tcp_distance: float = 0.015,
        max_gripper_width_error: float = 0.012,
        max_gripper_command: float = 0.025,
    ) -> torch.Tensor:
        _, _, held_pour = source_grasp_milestones(
            env,
            min_lift_height=min_lift_height,
            max_tcp_distance=max_tcp_distance,
            max_gripper_width_error=max_gripper_width_error,
            max_gripper_command=max_gripper_command,
        )
        env.update_held_delivery_tracker(held_pour)
        current_fraction = env.current_held_delivered_mask().sum(dim=1).float() / max(env.num_particles, 1)
        credit = torch.minimum(current_fraction, env.pour_target_frac)
        success_terminal = _success_terminal(env)
        unsuccessful_done = env.termination_manager.dones & ~success_terminal
        retained_credit = torch.where(unsuccessful_done, torch.zeros_like(credit), credit)
        progress = retained_credit - self._previous_credit
        self._previous_credit.copy_(retained_credit)
        return progress / max(float(env.step_dt), 1.0e-6)


# Backward-compatible name retained for downstream configurations and imports.
NewlyDeliveredParticles = HeldDeliveryProgress


class NewlySpilledParticles(ManagerTermBase):
    """Penalize each particle's first spill as a time-step-independent pulse."""

    def __init__(self, cfg: RewardTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        self._spilled = torch.zeros(
            (env.num_envs, env.num_particles),
            device=env.device,
            dtype=torch.bool,
        )

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._spilled[env_ids] = False

    def __call__(self, env: FrankaPourEnv) -> torch.Tensor:
        spilled = env.particles_spilled_mask()
        newly_spilled = spilled & ~self._spilled
        self._spilled |= spilled
        fraction = newly_spilled.sum(dim=1).float() / max(env.num_particles, 1)
        return fraction / max(float(env.step_dt), 1.0e-6)


def spilled_particles(env: FrankaPourEnv) -> torch.Tensor:
    """Fraction irrecoverably spilled onto or below the table."""
    return env.spilled_fraction()


def pour_success_bonus(env: FrankaPourEnv) -> torch.Tensor:
    """Reward exactly the stable-success termination used by curriculum progression."""
    success = _success_terminal(env)
    return success.float() / max(float(env.step_dt), 1.0e-6)


def sustained_pour_success(env: FrankaPourEnv, dwell_time_s: float = 0.15) -> torch.Tensor:
    """Reward the current dwell-qualified success state without a terminal pulse."""
    if not math.isfinite(dwell_time_s) or dwell_time_s <= 0.0:
        raise ValueError("dwell_time_s must be finite and positive.")
    dwell_steps = max(1, math.ceil(float(dwell_time_s) / max(float(env.step_dt), 1.0e-6)))
    return (env._success_dwell_count >= dwell_steps).float()
