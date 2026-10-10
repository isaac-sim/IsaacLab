# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the dual-Franka shoelace task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .grasp import bilateral_score, filter_grasps, hamacher_product, shoelace_grasp_quality
from .observations import tails_to_tcp
from .utils import tail_outward_x, tail_x_separation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg, SceneEntityCfg


def arm_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared normalized arm commands, excluding binary gripper commands.

    Args:
        env: Task environment with ``left_arm`` and ``right_arm`` action terms.

    Returns:
        Sum of squared arm commands before Cartesian scaling, shape [N].
    """
    return _arm_action_squared_sum(env, env.action_manager.action)


def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return squared changes in normalized arm commands, excluding gripper changes.

    Args:
        env: Task environment with ``left_arm`` and ``right_arm`` action terms.

    Returns:
        Sum of squared differences from the previous policy step, shape [N]. This is not divided by
        the step duration. The action manager clears history to zero on reset.
    """
    delta = env.action_manager.action - env.action_manager.prev_action
    return _arm_action_squared_sum(env, delta)


def shoelace_success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the configured success flag as an event rate [1/s].

    The reward manager cancels ``step_dt``, so its weight is the reward per successful episode.
    The environment resets after success; timeout alone earns no completion reward.

    Args:
        env: Task environment with a ``success`` termination term.

    Returns:
        Success event rates [1/s], shape [N].
    """
    return env.termination_manager.get_term("success").float() / env.step_dt


class dense_task_reward(ManagerTermBase):
    """Stateful progress reward for acquiring and pulling the two free tails.

    Maintains per-environment grasp filters, initial tail offsets, progress records, and acquisition potential.

    Notes:
        - Phase metrics under ``Metrics/shoelace/`` average only environments with finite reward inputs.
        - ``pull_left_displacement_m`` and ``pull_right_displacement_m`` report signed outward tail
          displacement [m] from the first valid sample after reset, independently of grasp quality.
        - ``pull_left_score`` and ``pull_right_score`` report grasp-gated diagnostic scores
          in [0, 1], not distances or the new-record pull reward.
        - ``valid_fraction`` measures numerical input validity, not grasp quality or task success.
        - ``success_rate`` averages the latest completed result per environment, excluding those with
          no completed episode. It is zero until the first completion.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_per_gripper_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._baseline_outward_x = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._best_pull_progress = torch.zeros((env.num_envs, 3), device=env.device)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._last_episode_success = torch.full((env.num_envs,), torch.nan, device=env.device)
        self._metrics: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear episode state so the next valid evaluation gives no reset credit.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        selected = slice(None) if env_ids is None else env_ids
        self._filtered_per_gripper_grasp[selected] = torch.nan
        self._baseline_outward_x[selected] = torch.nan
        self._best_pull_progress[selected] = 0.0
        self._previous_potential[selected] = torch.nan
        # ManagerBasedRLEnv replaces the log dictionary before resetting reward terms.
        self._env.extras.setdefault("log", {}).update(self._metrics)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reach_std: float,
        contact_std: float,
        relative_speed_std: float,
        grasp_filter_time_constant: float,
        open_position: float,
        closed_position: float,
        success_x_separation: float,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg] | None = None,
        *,
        acquisition_weight: float = 0.6,
        approach_fraction: float = 0.5,
        bilateral_pull_fraction: float = 0.8,
        bilateral_approach_fraction: float = 0.5,
        bilateral_grasp_fraction: float = 0.7,
        contact_penetration_tolerance: float = 0.0,
        pull_grasp_threshold: float = 0.2,
    ) -> torch.Tensor:
        """Return the signed rate of acquisition-and-pull progress [1/s].

        Combines acquisition potential differences with grasp-gated outward-distance records;
        each includes independent-arm credit and a bilateral bonus. Acquisition penalizes moving
        away or losing grasps; physical grasp quality is defined by :func:`shoelace_grasp_quality`.

        Notes:
            - The first finite sample seeds history without reward; invalid samples pay zero and
              preserve history. Regrasping never resets baselines or records.
            - Pull records advance even without grasps, preventing passive-motion and repeated-cycle
              credit. Progress is unbounded; beyond the separation scale, bilateral progress follows
              the trailing arm. Diagnostic pull scores instead start at 0.5 before grasp gating.
            - Acquisition cycles cancel without discounting; discounted policy invariance is not claimed.

        Args:
            env: The task environment.
            reach_std: Tail-to-TCP approach-distance width [m].
            contact_std: Contact-distance width [m]; gaps and deep penetration reduce grasp quality.
            relative_speed_std: Tail-TCP slip-speed width [m/s].
            grasp_filter_time_constant: Grasp-quality low-pass time constant [s].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            success_x_separation: X-separation target for the pull scale [m]; success is checked separately.
            cable_cfgs: Required left and right cable scene entities.
            robot_cfgs: Required left and right robot hand and finger scene entities.
            acquisition_weight: Acquisition fraction in [0, 1]; pulling receives the remainder.
            approach_fraction: Approach share of acquisition in [0, 1]; grasps receive the remainder.
            bilateral_pull_fraction: Bilateral share of pulling in [0, 1]; zero disables the bonus.
            bilateral_approach_fraction: Bilateral share of approach in [0, 1]; zero gives the per-arm mean.
            bilateral_grasp_fraction: Bilateral share of grasp acquisition in [0, 1]; zero gives the per-arm mean.
            contact_penetration_tolerance: Accepted contact-solver penetration [m]. Positive gaps and
                penetration beyond this tolerance still reduce grasp quality.
            pull_grasp_threshold: Minimum current and filtered grasp quality for each arm's record credit;
                both arms must qualify for new bilateral records.

        Returns:
            Signed reward rates [1/s], shape [N]. RewardManager cancels the timestep division
            and applies the term weight. High-water pull has no fixed total reward cap.

        Raises:
            ValueError: If scene entities, reward budgets, or the separation target are invalid.
        """
        if cable_cfgs is None or robot_cfgs is None:
            raise ValueError("dense_task_reward requires cable_cfgs and robot_cfgs")
        if not 0.0 <= acquisition_weight <= 1.0 or not 0.0 <= approach_fraction <= 1.0:
            raise ValueError("acquisition_weight and approach_fraction must be in [0, 1]")
        if not all(
            0.0 <= value <= 1.0
            for value in (bilateral_pull_fraction, bilateral_approach_fraction, bilateral_grasp_fraction)
        ):
            raise ValueError("Bilateral approach, grasp, and pull fractions must be in [0, 1]")
        if not math.isfinite(success_x_separation) or success_x_separation <= 0.0:
            raise ValueError("success_x_separation must be finite and positive")
        if not 0.0 < pull_grasp_threshold <= 1.0:
            raise ValueError("pull_grasp_threshold must be in (0, 1]")

        # Per-arm tensors follow robot order (left, right), which is opposite to the cable naming.
        tail_vectors = tails_to_tcp(env, cable_cfgs, robot_cfgs).reshape(env.num_envs, 2, 3)
        tail_distances = torch.linalg.vector_norm(tail_vectors, dim=-1)
        per_gripper_grasp, grasp_finite = shoelace_grasp_quality(
            env,
            contact_std,
            relative_speed_std,
            open_position,
            closed_position,
            cable_cfgs,
            robot_cfgs,
            contact_penetration_tolerance,
        )
        x_separation = tail_x_separation(env, cable_cfgs)
        outward_x = tail_outward_x(env, cable_cfgs)
        # Per-environment numerical mask for reward inputs; this does not assess grasp or task success.
        finite = (
            torch.isfinite(tail_distances).all(dim=1)
            & grasp_finite
            & torch.isfinite(x_separation)
            & torch.isfinite(outward_x).all(dim=1)
        )

        approach = 1.0 - torch.tanh(tail_distances / max(reach_std, 1.0e-6))
        filtered_per_gripper_grasp = filter_grasps(
            self._filtered_per_gripper_grasp,
            per_gripper_grasp,
            finite,
            env.step_dt,
            max(grasp_filter_time_constant, 1.0e-6),
        )
        bilateral_grasp = hamacher_product(filtered_per_gripper_grasp[:, 0], filtered_per_gripper_grasp[:, 1])
        # Independent credit starts either arm; cooperation makes acquiring the other arm more valuable.
        approach_score = bilateral_score(approach, bilateral_approach_fraction)
        grasp_score = bilateral_score(filtered_per_gripper_grasp, bilateral_grasp_fraction)
        acquire = approach_fraction * approach_score + (1.0 - approach_fraction) * grasp_score

        # Keep each reference fixed until episode reset; releasing/regrasping must not renew pull credit.
        unseeded_baseline = ~torch.isfinite(self._baseline_outward_x)
        self._baseline_outward_x.copy_(
            torch.where(finite.unsqueeze(1) & unseeded_baseline, outward_x, self._baseline_outward_x)
        )
        initial_separation = self._baseline_outward_x.sum(dim=1).abs()
        # Split the remaining target separation between arms; the 1 cm floor avoids a near-zero scale.
        pull_scale = (0.5 * (success_x_separation - initial_separation)).clamp_min(0.01)
        outward_displacement = outward_x - self._baseline_outward_x
        # Start at 0.5 so outward motion below the reset baseline still changes the potential smoothly.
        per_arm_progress = 0.5 * (1.0 + torch.tanh(outward_displacement / pull_scale.unsqueeze(1)))
        # Gate each tail by its own grasp; the bilateral term is a bonus, not a prerequisite for pulling.
        per_arm_pull = hamacher_product(filtered_per_gripper_grasp, per_arm_progress)
        potential = acquisition_weight * acquire
        physical_progress = (outward_displacement / pull_scale.unsqueeze(1)).clamp_min(0.0)
        # Preserve shaping below the scale; extend cooperation with the trailing arm beyond it.
        bounded_progress = physical_progress.clamp_max(1.0)
        bilateral_progress = hamacher_product(bounded_progress[:, 0], bounded_progress[:, 1])
        bilateral_progress += (physical_progress.amin(dim=1) - 1.0).clamp_min(0.0)
        progress_records = torch.cat((physical_progress, bilateral_progress.unsqueeze(1)), dim=1)
        best_progress = torch.maximum(self._best_pull_progress, progress_records)
        new_progress = best_progress - self._best_pull_progress
        # Raw quality prevents the filter's release tail from paying for ungrasped motion.
        eligible = (per_gripper_grasp >= pull_grasp_threshold) & (filtered_per_gripper_grasp >= pull_grasp_threshold)
        pull_increment = (1.0 - bilateral_pull_fraction) * (new_progress[:, :2] * eligible).mean(dim=1)
        pull_increment += bilateral_pull_fraction * new_progress[:, 2] * eligible.all(dim=1)
        # Consume even ungrasped records so closing later cannot collect passive-motion credit.
        self._best_pull_progress.copy_(torch.where(finite.unsqueeze(1), best_progress, self._best_pull_progress))

        metric_values = {
            "approach_distance_m": tail_distances.mean(dim=1),
            "grasp_left": filtered_per_gripper_grasp[:, 0],
            "grasp_right": filtered_per_gripper_grasp[:, 1],
            "grasp_both": bilateral_grasp,
            "pull_x_separation_m": x_separation,
            "pull_left_displacement_m": outward_displacement[:, 0],
            "pull_right_displacement_m": outward_displacement[:, 1],
            "pull_left_score": per_arm_pull[:, 0],
            "pull_right_score": per_arm_pull[:, 1],
        }
        samples = torch.stack(tuple(metric_values.values()), dim=-1)
        means = torch.where(finite.unsqueeze(1), samples, 0.0).sum(dim=0) / finite.sum().clamp_min(1)
        self._metrics = {
            f"Metrics/shoelace/{name}": value.detach() for name, value in zip(metric_values, means, strict=True)
        }
        # Fraction of environments passing the NaN/Inf check this step; normally 1.0.
        self._metrics["Metrics/shoelace/valid_fraction"] = finite.float().mean()
        # Retain each environment's latest completed result; exclude environments with no finished episode.
        self._last_episode_success.copy_(
            torch.where(
                env.termination_manager.dones,
                env.termination_manager.get_term("success").float(),
                self._last_episode_success,
            )
        )
        self._metrics["Metrics/shoelace/success_rate"] = self._last_episode_success.nan_to_num().sum() / (
            torch.isfinite(self._last_episode_success).sum().clamp_min(1)
        )
        # RSL-RL retains each step's dictionary until logging the training iteration.
        env.extras["log"] = {**env.extras.get("log", {}), **self._metrics}

        # Seed without reset credit; preserve acquisition losses independently of earned pull records.
        valid = finite & torch.isfinite(self._previous_potential)
        progress = torch.where(
            valid,
            potential - self._previous_potential + (1.0 - acquisition_weight) * pull_increment,
            torch.zeros_like(potential),
        )
        self._previous_potential.copy_(torch.where(finite, potential, self._previous_potential))
        # RewardManager multiplies by step_dt, leaving weighted progress per policy step.
        return progress / env.step_dt


class grasp_hold_reward(ManagerTermBase):
    """Reward retained physical grasps, with an independent filter for hold-only ablations.

    Unlike the dense potential difference, this score remains positive during stable grasping.
    The manager multiplies it by ``step_dt`` and its weight, making the weight a maximum reward
    per second. An optional episode budget reduces the rate after enough quality-weighted grasp time.
    Invalid contact, closure, or slip inputs earn zero without advancing filters or budgets.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_per_gripper_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._full_rate_time_used = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear the grasp filter and retention budget for the selected environments.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        selected = slice(None) if env_ids is None else env_ids
        self._filtered_per_gripper_grasp[selected] = torch.nan
        self._full_rate_time_used[selected] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        contact_std: float,
        relative_speed_std: float,
        grasp_filter_time_constant: float,
        open_position: float,
        closed_position: float,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        bilateral_grasp_fraction: float = 0.8,
        *,
        contact_penetration_tolerance: float = 0.0,
        full_reward_duration: float | None = None,
        sustained_reward_fraction: float = 0.2,
    ) -> torch.Tensor:
        """Return sustained grasp quality with a bilateral bonus.

        Args:
            env: Task environment.
            contact_std: Finger-tail contact-distance width [m].
            relative_speed_std: Tail-TCP slip-speed width [m/s].
            grasp_filter_time_constant: Grasp-quality low-pass time constant [s].
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            cable_cfgs: Left and right cable scene entities.
            robot_cfgs: Left and right hand and finger scene entities.
            bilateral_grasp_fraction: Bilateral share in [0, 1]; the remainder rewards each arm independently.
            contact_penetration_tolerance: Accepted contact-solver penetration [m]. Positive gaps and
                penetration beyond this tolerance still reduce grasp quality.
            full_reward_duration: Maximum quality-weighted retention time at full reward rate [s] per
                episode. ``None`` disables the budget. Release or regrasp does not renew it.
            sustained_reward_fraction: Remaining reward fraction in [0, 1] after the full-rate budget.

        Returns:
            Hold scores in [0, 1], shape [N]. The first finite sample seeds the filter without a ramp.

        Raises:
            ValueError: If fractions are outside [0, 1] or the optional duration is invalid.
        """
        if not 0.0 <= bilateral_grasp_fraction <= 1.0:
            raise ValueError("bilateral_grasp_fraction must be in [0, 1]")
        if full_reward_duration is not None and (not math.isfinite(full_reward_duration) or full_reward_duration < 0.0):
            raise ValueError("full_reward_duration must be None or finite and nonnegative")
        if not 0.0 <= sustained_reward_fraction <= 1.0:
            raise ValueError("sustained_reward_fraction must be in [0, 1]")
        grasp, finite = shoelace_grasp_quality(
            env,
            contact_std,
            relative_speed_std,
            open_position,
            closed_position,
            cable_cfgs,
            robot_cfgs,
            contact_penetration_tolerance,
        )
        filtered = filter_grasps(
            self._filtered_per_gripper_grasp, grasp, finite, env.step_dt, max(grasp_filter_time_constant, 1.0e-6)
        )
        score = torch.where(finite, bilateral_score(filtered, bilateral_grasp_fraction), 0.0)
        if full_reward_duration is None:
            return score
        remaining = (full_reward_duration - self._full_rate_time_used).clamp_min(0.0)
        full_rate_time = torch.minimum(score * env.step_dt, remaining)
        self._full_rate_time_used.add_(full_rate_time)
        # Split boundary-crossing steps exactly, so changing policy dt does not change the budget.
        return sustained_reward_fraction * score + (1.0 - sustained_reward_fraction) * full_rate_time / env.step_dt


class pregrasp_progress_reward(ManagerTermBase):
    """Bridge coarse approach and physical grasping with fine TCP alignment and gated closure.

    This is a positional pregrasp heuristic, not evidence of contact or tail-orientation alignment.
    Each arm earns progress independently; stationary states pay zero and releases remove closure
    credit. The contact-based grasp and retention terms remain responsible for physical grasp quality.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._previous_potential = torch.full((env.num_envs,), torch.nan, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected baselines so the next valid sample gives no reset credit.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        self._previous_potential[slice(None) if env_ids is None else env_ids] = torch.nan

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        alignment_std: float,
        closure_radius: float,
        open_position: float,
        closed_position: float,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        alignment_weight: float = 0.75,
        closure_weight: float = 0.25,
    ) -> torch.Tensor:
        """Return the signed rate of fine alignment and nearby actual closure [1/s].

        Args:
            env: Task environment.
            alignment_std: Gaussian width for TCP-to-tail-center distance [m].
            closure_radius: TCP-centered radius beyond which closure earns no credit [m].
                The gate is ``max(1 - (distance / radius)**2, 0)**2`` and is smooth at its boundary.
            open_position: Driven finger-joint position when open [m].
            closed_position: Driven finger-joint position when closed [m].
            cable_cfgs: Left and right cable scene entities.
            robot_cfgs: Left and right hand and finger scene entities.
            alignment_weight: Nonnegative fine-alignment potential budget, averaged over the two arms.
            closure_weight: Nonnegative gated-closure potential budget, averaged over the two arms.
                Zero disables closure shaping without changing the alignment budget.

        Returns:
            Signed progress rates [1/s], shape [N]. The manager's ``step_dt`` multiplication cancels
            the timestep division. The potential is bounded by the sum of the two component weights.
            Invalid inputs pay zero and preserve history; the first valid sample after reset pays zero.

        Raises:
            ValueError: If widths are not finite and positive, component weights are not finite and
                nonnegative, or the open position does not exceed the closed position.
        """
        if not all(math.isfinite(value) and value > 0.0 for value in (alignment_std, closure_radius)):
            raise ValueError("alignment_std and closure_radius must be finite and positive")
        if not all(math.isfinite(value) and value >= 0.0 for value in (alignment_weight, closure_weight)):
            raise ValueError("alignment_weight and closure_weight must be finite and nonnegative")
        if (
            not all(math.isfinite(value) for value in (open_position, closed_position))
            or open_position <= closed_position
        ):
            raise ValueError("Gripper positions must be finite with open_position greater than closed_position")

        vectors = tails_to_tcp(env, cable_cfgs, robot_cfgs).reshape(env.num_envs, 2, 3)
        distance = torch.linalg.vector_norm(vectors, dim=-1)
        positions = torch.stack(
            [env.scene[cfg.name].data.joint_pos.torch[:, cfg.joint_ids[0]] for cfg in robot_cfgs], dim=1
        )
        finite = torch.isfinite(distance).all(dim=1) & torch.isfinite(positions).all(dim=1)
        closure = ((open_position - positions) / (open_position - closed_position)).clamp(0.0, 1.0)
        alignment = torch.exp(-torch.square(distance / alignment_std))
        # A compact positional gate cannot reward closing far away, even before contact is observable.
        closure_gate = (1.0 - torch.square(distance / closure_radius)).clamp_min(0.0).square()
        potential = (alignment_weight * alignment + closure_weight * closure_gate * closure).mean(dim=1)

        valid = finite & torch.isfinite(self._previous_potential)
        progress = torch.where(valid, potential - self._previous_potential, 0.0)
        self._previous_potential.copy_(torch.where(finite, potential, self._previous_potential))
        return progress / env.step_dt


def _arm_action_squared_sum(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    """Select arm commands by name so action reordering cannot include grippers."""
    term_slices: dict[str, slice] = {}
    start = 0
    for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim, strict=True):
        term_slices[name] = slice(start, start + dim)
        start += dim
    return actions[:, term_slices["left_arm"]].square().sum(dim=-1) + actions[:, term_slices["right_arm"]].square().sum(
        dim=-1
    )
