# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adaptive curriculum over a validated Franka Pour reset-state dataset."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg
from isaaclab.managers.manager_base import ManagerTermBase

from isaaclab_tasks.utils.adaptive_reset_sampler import AdaptiveResetSampler

if TYPE_CHECKING:
    from ..pour_env import FrankaPourEnv


def reset_dataset_difficulty(
    states: Mapping[str, torch.Tensor],
    task_contract: Mapping[str, object],
) -> torch.Tensor:
    """Return a normalized reverse-curriculum difficulty for every dataset row.

    Grasping rows occupy the easy half of the range and are ordered by their task objective.
    Non-grasping rows occupy the hard half and are graded from information already stored in the
    dataset: normalized arm displacement, source/receiver displacement, and hand closure. This
    avoids the arbitrary ordering that results when every non-grasping objective is fixed at -1.

    Args:
        states: Row-aligned Franka Pour reset-state tensors.
        task_contract: Geometry and robot contract stored with the dataset.

    Returns:
        One difficulty per row in ``[0, 1]``, where lower is easier.
    """
    category = states["category"]
    objective = states["objective"]
    if category.ndim != 1 or objective.shape != category.shape or category.numel() == 0:
        raise ValueError("Reset dataset category and objective must be aligned non-empty vectors.")
    if not bool(torch.isfinite(objective).all()) or not bool(((category == 0) | (category == 1)).all()):
        raise ValueError("Reset dataset categories and objectives must be finite and valid.")

    difficulty = torch.empty_like(objective, dtype=torch.float32)
    grasping = category == 1
    difficulty[grasping] = 0.5 * (1.0 - objective[grasping].float()).clamp(0.0, 1.0)
    non_grasping = ~grasping
    if not bool(non_grasping.any()):
        return difficulty

    arm_position = states["arm_joint_position"][non_grasping].float()
    arm_home = torch.as_tensor(task_contract["arm_home"], device=arm_position.device, dtype=arm_position.dtype)
    arm_limits = torch.as_tensor(
        task_contract["arm_joint_limits"], device=arm_position.device, dtype=arm_position.dtype
    )
    if arm_home.shape != (arm_position.shape[1],) or arm_limits.shape != (arm_position.shape[1], 2):
        raise ValueError("Reset dataset arm contract does not match its joint-state shape.")
    arm_span = (arm_limits[:, 1] - arm_limits[:, 0]).clamp_min(torch.finfo(arm_position.dtype).eps)
    arm_score = (2.0 * torch.abs(arm_position - arm_home) / arm_span).mean(dim=-1).clamp(0.0, 1.0)

    source_xy = states["source_root_pose"][non_grasping, :2].float()
    source_center = torch.as_tensor(
        task_contract["source_region_center"], device=source_xy.device, dtype=source_xy.dtype
    )[:2]
    target_xy = states["target_root_pose"][non_grasping, :2].float()
    target_center = torch.as_tensor(task_contract["target_center_xy"], device=target_xy.device, dtype=target_xy.dtype)
    support_lower = torch.as_tensor(
        task_contract["tabletop_support_lower_xy"], device=source_xy.device, dtype=source_xy.dtype
    )
    support_upper = torch.as_tensor(
        task_contract["tabletop_support_upper_xy"], device=source_xy.device, dtype=source_xy.dtype
    )
    support_diagonal = torch.linalg.vector_norm(support_upper - support_lower).clamp_min(1.0e-6)
    source_score = (torch.linalg.vector_norm(source_xy - source_center, dim=-1) / support_diagonal).clamp(0.0, 1.0)
    target_score = (torch.linalg.vector_norm(target_xy - target_center, dim=-1) / support_diagonal).clamp(0.0, 1.0)

    fingers = states["finger_joint_position"][non_grasping].float().mean(dim=-1)
    gripper_lower, gripper_upper = (
        float(value)
        for value in task_contract["gripper_position_range"]  # type: ignore[arg-type]
    )
    gripper_span = max(gripper_upper - gripper_lower, 1.0e-6)
    closed_score = ((gripper_upper - fingers) / gripper_span).clamp(0.0, 1.0)

    non_grasp_score = 0.45 * arm_score + 0.30 * source_score + 0.15 * target_score + 0.10 * closed_score
    difficulty[non_grasping] = 0.5 + 0.5 * non_grasp_score
    return difficulty


class PourResetDatasetCurriculum(ManagerTermBase):
    """Record reset outcomes and sample a stable easy-to-hard dataset frontier."""

    def __init__(self, cfg: CurriculumTermCfg, env: FrankaPourEnv):
        super().__init__(cfg, env)
        if not getattr(env, "_uses_reset_dataset", False):
            raise RuntimeError("PourResetDatasetCurriculum requires the reset-dataset environment variant.")

        states = env._reset_dataset_states
        self._difficulty = reset_dataset_difficulty(states, env._reset_dataset_metadata["task_contract"])
        difficulty_order = torch.argsort(self._difficulty, stable=True)
        sampler_cfg = env.cfg.reset_dataset_sampler.copy()
        self._sampler = AdaptiveResetSampler(
            difficulty_order,
            sampler_cfg,
        )

        row_count = states["category"].numel()
        self._frozen_rows = torch.arange(row_count, device=env.device, dtype=torch.long)
        top_grasp_count = env.cfg.reset_dataset_top_grasp_count
        if top_grasp_count is not None:
            grasp_rows = torch.nonzero(states["category"] == 1, as_tuple=False).flatten()
            if top_grasp_count > grasp_rows.numel():
                raise ValueError(
                    "reset_dataset_top_grasp_count exceeds the dataset grasp count: "
                    f"{top_grasp_count} > {grasp_rows.numel()}."
                )
            grasp_order = torch.argsort(states["objective"][grasp_rows], descending=True, stable=True)
            self._frozen_rows = grasp_rows[grasp_order[:top_grasp_count]]
        if row_count == 0 or self._frozen_rows.numel() == 0:
            raise RuntimeError("The reset dataset must expose at least one playback row.")

    @staticmethod
    def _env_ids(
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> torch.Tensor:
        """Normalize manager-provided environment IDs on the simulation device."""
        if isinstance(env_ids, slice):
            return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
        return torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()

    def __call__(
        self,
        env: FrankaPourEnv,
        env_ids: Sequence[int] | torch.Tensor | slice,
    ) -> dict[str, float]:
        """Record completed episodes, select exact raw rows, and report compact progress."""
        ids = self._env_ids(env, env_ids)
        if ids.numel() > 0:
            completed = (env.episode_length_buf[ids] > 0) & (env.reset_dataset_row_id[ids] >= 0)
            completed_ids = ids[completed]
            if completed_ids.numel() > 0 and not env.cfg.curriculum_freeze:
                completed_rows = env.reset_dataset_row_id[completed_ids]
                self._sampler.record(completed_rows, env.episode_succeeded[completed_ids])

            forced_rows = env._forced_reset_dataset_row[ids]
            if env.cfg.curriculum_freeze:
                slots = torch.randint(self._frozen_rows.numel(), (ids.numel(),), device=env.device)
                rows = self._frozen_rows[slots]
                use_forced = forced_rows >= 0
                if bool(torch.any(use_forced)):
                    # Reuse the generic sampler's exact-row validation without changing frozen
                    # sampling for the remaining environments.
                    rows = torch.where(
                        use_forced,
                        self._sampler.sample(ids.numel(), forced_row_ids=forced_rows),
                        rows,
                    )
            else:
                rows = self._sampler.sample(ids.numel(), forced_row_ids=forced_rows)
            env.reset_dataset_row_id[ids] = rows
            env.pour_target_frac[ids] = float(env.cfg.pour_target_frac)

        if env.cfg.curriculum_freeze:
            return {
                "frozen_pool_fraction": float(self._frozen_rows.numel() / self._difficulty.numel()),
                "frozen_pool_size": float(self._frozen_rows.numel()),
            }

        metrics = self._sampler.metrics()
        return {
            "predicted_success_rate": metrics["predicted_success_rate"],
            "observed_success_rate": metrics["bounded_success_rate"],
            "dataset_success_rate": metrics["cache_success_rate"],
            "dataset_ever_solved_fraction": metrics["ever_solved_fraction"],
            "frontier_fraction": metrics["frontier_fraction"],
            "effective_pool_size": metrics["effective_pool_size"],
        }
