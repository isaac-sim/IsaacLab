# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the dual-Franka shoelace task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase

from .grasp import filter_grasps, shoelace_grasp_quality
from .utils import tail_outward_x, untying_metrics

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg


class shoelace_bilateral_pull_success(ManagerTermBase):
    """Require geometric completion after both tails have been pulled with bilateral grasps.

    Geometry limits throat occupancy separately for each arm's cable and requires each tail to reach
    its own outward X boundary. One cleared or far-pulled tail cannot compensate for the other.
    Only new outward X records beyond the first finite post-reset sample count, and both raw and
    filtered grasps must qualify at both ends of the policy-step interval. Records advance even without
    grasps, so passive motion, late closure, and repeated excursions cannot create completion credit.
    Completion of the cooperative phase is retained until reset, allowing release before geometric success.
    This term has its own history and works independently of which rewards are enabled.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self._filtered_grasp = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._best_outward_x = torch.full((env.num_envs, 2), torch.nan, device=env.device)
        self._loaded_pull_distance = torch.zeros((env.num_envs, 2), device=env.device)
        self._previous_bilateral_grasp = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._metrics: dict[str, torch.Tensor] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear selected episode histories without losing the terminal step's logged metrics.

        Args:
            env_ids: Environments to reset. ``None`` resets all environments.
        """
        selected = slice(None) if env_ids is None else env_ids
        self._filtered_grasp[selected] = torch.nan
        self._best_outward_x[selected] = torch.nan
        self._loaded_pull_distance[selected] = 0.0
        self._previous_bilateral_grasp[selected] = False
        self._env.extras.setdefault("log", {}).update(self._metrics)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        throat_radius: float,
        minimum_pull_distance: float,
        grasp_threshold: float,
        contact_std: float,
        relative_speed_std: float,
        grasp_filter_time_constant: float,
        open_position: float,
        closed_position: float,
        cable_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        robot_cfgs: tuple[SceneEntityCfg, SceneEntityCfg],
        contact_penetration_tolerance: float = 0.0,
        maximum_throat_segments_per_arm: int = 15,
        minimum_tail_outward_distance: float = 0.09,
    ) -> torch.Tensor:
        """Return success after cooperative loaded pull and current geometric completion.

        Args:
            env: Shoelace environment.
            throat_radius: Radius around the fixed seam midpoint [m].
            minimum_pull_distance: Required new outward record distance under bilateral grasps per tail [m].
            grasp_threshold: Minimum raw and filtered quality for each grasp, in (0, 1].
            contact_std: Finger-tail surface-distance width [m].
            relative_speed_std: Tail-to-TCP relative-speed width [m/s].
            grasp_filter_time_constant: Exponential grasp filter time constant [s].
            open_position: Open finger joint position [m].
            closed_position: Closed finger joint position [m].
            cable_cfgs: Left and right cable scene entities.
            robot_cfgs: Left and right robot entities with resolved finger joint and hand body indices.
            contact_penetration_tolerance: Allowed contact-solver penetration [m].
            maximum_throat_segments_per_arm: Maximum free capsule centers in the throat for either arm's cable.
            minimum_tail_outward_distance: Minimum signed outward X offset of each tail from the seam midpoint [m].

        Returns:
            Success flags for finite states completing both phases, shape [N].

        Raises:
            ValueError: If a pull or geometry limit, grasp threshold, or filter time constant is invalid.
        """
        if not math.isfinite(minimum_pull_distance) or minimum_pull_distance <= 0.0:
            raise ValueError("minimum_pull_distance must be finite and positive")
        if not 0.0 < grasp_threshold <= 1.0:
            raise ValueError("grasp_threshold must be in (0, 1]")
        if not math.isfinite(grasp_filter_time_constant) or grasp_filter_time_constant <= 0.0:
            raise ValueError("grasp_filter_time_constant must be finite and positive")
        if not isinstance(maximum_throat_segments_per_arm, int) or maximum_throat_segments_per_arm < 0:
            raise ValueError("maximum_throat_segments_per_arm must be a nonnegative integer")
        if not math.isfinite(minimum_tail_outward_distance) or minimum_tail_outward_distance <= 0.0:
            raise ValueError("minimum_tail_outward_distance must be finite and positive")

        throat_counts, _, _, geometry_finite = untying_metrics(
            env, cable_cfgs, throat_radius, per_arm_throat_counts=True
        )
        outward_x = tail_outward_x(env, cable_cfgs)
        geometry_success = (
            geometry_finite
            & (throat_counts <= maximum_throat_segments_per_arm).all(dim=1)
            & (outward_x >= minimum_tail_outward_distance).all(dim=1)
        )
        grasp, grasp_finite = shoelace_grasp_quality(
            env,
            contact_std,
            relative_speed_std,
            open_position,
            closed_position,
            cable_cfgs,
            robot_cfgs,
            contact_penetration_tolerance,
        )
        finite = geometry_finite & grasp_finite & torch.isfinite(outward_x).all(dim=1)
        filtered = filter_grasps(self._filtered_grasp, grasp, finite, env.step_dt, grasp_filter_time_constant)
        bilateral_grasp = finite & ((grasp >= grasp_threshold) & (filtered >= grasp_threshold)).all(dim=1)

        best = torch.where(torch.isfinite(self._best_outward_x), self._best_outward_x, outward_x)
        new_distance = (outward_x - best).clamp_min(0.0)
        eligible = bilateral_grasp & self._previous_bilateral_grasp
        self._loaded_pull_distance += torch.where(eligible.unsqueeze(1), new_distance, 0.0)
        self._best_outward_x.copy_(
            torch.where(finite.unsqueeze(1), torch.maximum(best, outward_x), self._best_outward_x)
        )
        # Invalid samples break eligibility so motion across missing observations is not credited.
        self._previous_bilateral_grasp.copy_(bilateral_grasp)
        completed = (self._loaded_pull_distance >= minimum_pull_distance).all(dim=1)
        self._metrics = {
            "Metrics/shoelace/geometry_success": geometry_success.float().mean().detach(),
            "Metrics/shoelace/throat_left_segments": throat_counts[:, 0].float().mean().detach(),
            "Metrics/shoelace/throat_right_segments": throat_counts[:, 1].float().mean().detach(),
            "Metrics/shoelace/bilateral_pull_completed": completed.float().mean().detach(),
            "Metrics/shoelace/loaded_pull_left_m": self._loaded_pull_distance[:, 0].mean().detach(),
            "Metrics/shoelace/loaded_pull_right_m": self._loaded_pull_distance[:, 1].mean().detach(),
        }
        env.extras["log"] = {**env.extras.get("log", {}), **self._metrics}
        return geometry_success & finite & completed
