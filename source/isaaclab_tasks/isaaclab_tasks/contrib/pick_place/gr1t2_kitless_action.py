# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""GR1T2 Pink action conditioning for standalone OVPhysX and Newton."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.pink_task_space_actions import PinkInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .gr1t2_kitless_action_cfg import GR1T2KitlessPinkInverseKinematicsActionCfg


def bound_ik_joint_targets(
    targets: torch.Tensor,
    current: torch.Tensor,
    max_delta_rad: float,
    max_magnitude_rad: float,
) -> torch.Tensor:
    """Hold invalid rows and bound finite IK targets relative to measured state."""
    finite_rows = torch.isfinite(targets).all(dim=1, keepdim=True)
    safe_targets = torch.where(finite_rows, targets, current)
    delta = torch.clamp(safe_targets - current, -max_delta_rad, max_delta_rad)
    bounded = torch.clamp(current + delta, -max_magnitude_rad, max_magnitude_rad)
    return torch.where(finite_rows, bounded, current)


def constrain_hand_joint_targets(
    targets: torch.Tensor,
    mimic_pairs: tuple[tuple[int, int, float, float], ...],
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """Apply software mimic relations and anatomical bounds to hand targets."""
    constrained = targets.clone()
    for mimic_index, reference_index, factor, bias in mimic_pairs:
        constrained[:, mimic_index] = factor * constrained[:, reference_index] + bias
    return torch.maximum(torch.minimum(constrained, upper), lower)


class _HandTargetSlewState:
    """Per-environment hand target history, independent of measured joint state."""

    def __init__(self, prototype: torch.Tensor, max_step_rad: float) -> None:
        self._previous = torch.zeros_like(prototype)
        self._initialized = torch.zeros(prototype.shape[0], dtype=torch.bool, device=prototype.device)
        self._max_step_rad = max_step_rad

    def condition(self, targets: torch.Tensor) -> torch.Tensor:
        slewed = self._previous + torch.clamp(targets - self._previous, -self._max_step_rad, self._max_step_rad)
        result = torch.where(self._initialized[:, None], slewed, targets)
        self._previous.copy_(result)
        self._initialized.fill_(True)
        return result

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._previous[env_ids] = 0.0
        self._initialized[env_ids] = False


class GR1T2KitlessPinkInverseKinematicsAction(PinkInverseKinematicsAction):
    """Pink action that owns the GR1T2 kitless target-stability boundary.

    Kit realizes the hand's PhysX mimic constraints and sanitizes the raw asset.
    The kitless task removes those hard constraints before physics parsing, so
    this term reproduces their target relation in software and bounds hand
    targets. OVPhysX widens physical hand limits; Newton retains them. It also
    bounds cold-start IK demands against measured state to avoid a first-step
    drive impulse. With the feature disabled, behavior is the base Pink action.
    """

    cfg: GR1T2KitlessPinkInverseKinematicsActionCfg

    def __init__(self, cfg: GR1T2KitlessPinkInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._kitless_target_conditioning_enabled = cfg.kitless_target_conditioning_enabled
        self._mimic_pairs: tuple[tuple[int, int, float, float], ...] = ()
        self._hand_lower: torch.Tensor | None = None
        self._hand_upper: torch.Tensor | None = None
        self._hand_target_slew: _HandTargetSlewState | None = None
        if not self._kitless_target_conditioning_enabled:
            return

        from .pickplace_gr1t2_env_cfg import GR1T2KitlessUsdFileCfg

        spawn_cfg = self._asset.cfg.spawn
        if isinstance(spawn_cfg, GR1T2KitlessUsdFileCfg) and spawn_cfg.action_cfg is not None:
            # Asset construction copies spawn settings before conditioning the USD. Read
            # its result here, rather than coupling independently copied environment configs.
            cfg.hand_joint_limits = dict(spawn_cfg.action_cfg.hand_joint_limits)
            cfg.hand_mimic_relations = dict(spawn_cfg.action_cfg.hand_mimic_relations)

        for name, value in (
            ("max_ik_joint_delta_rad", cfg.max_ik_joint_delta_rad),
            ("max_ik_joint_magnitude_rad", cfg.max_ik_joint_magnitude_rad),
            ("max_hand_target_step_rad", cfg.max_hand_target_step_rad),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if not math.isfinite(cfg.hand_target_margin_rad) or cfg.hand_target_margin_rad < 0.0:
            raise ValueError("hand_target_margin_rad must be finite and non-negative.")

        hand_indices = {name: index for index, name in enumerate(self._hand_joint_names)}
        missing_limits = [name for name in self._hand_joint_names if name not in cfg.hand_joint_limits]
        if missing_limits:
            raise ValueError(f"Missing pre-conditioned hand limits for: {missing_limits}")

        pairs = []
        for mimic_name, (reference_name, gearing, offset) in cfg.hand_mimic_relations.items():
            if mimic_name not in hand_indices or reference_name not in hand_indices:
                raise ValueError(
                    f"Mimic relation {mimic_name!r} -> {reference_name!r} is outside the Pink hand joint set."
                )
            if not all(math.isfinite(value) for value in (gearing, offset)):
                raise ValueError(f"Mimic relation for {mimic_name!r} must be finite.")
            # PhysX authors q_mimic + gearing * q_reference + offset = 0.
            pairs.append((hand_indices[mimic_name], hand_indices[reference_name], -gearing, -offset))
        self._mimic_pairs = tuple(pairs)

        lower = []
        upper = []
        for name in self._hand_joint_names:
            joint_lower, joint_upper = cfg.hand_joint_limits[name]
            if not all(math.isfinite(value) for value in (joint_lower, joint_upper)) or joint_lower > joint_upper:
                raise ValueError(f"Invalid hand limits for {name!r}: {(joint_lower, joint_upper)}")
            lower.append(joint_lower - cfg.hand_target_margin_rad)
            upper.append(joint_upper + cfg.hand_target_margin_rad)
        self._hand_lower = torch.tensor(lower, dtype=self._raw_actions.dtype, device=self.device)
        self._hand_upper = torch.tensor(upper, dtype=self._raw_actions.dtype, device=self.device)
        self._hand_target_slew = _HandTargetSlewState(
            torch.zeros(
                (self.num_envs, len(self._hand_joint_names)),
                dtype=self._raw_actions.dtype,
                device=self.device,
            ),
            cfg.max_hand_target_step_rad,
        )

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process input without mutating the action-manager input during coupling."""
        super().process_actions(actions)
        if self._kitless_target_conditioning_enabled:
            assert self._hand_lower is not None and self._hand_upper is not None
            assert self._hand_target_slew is not None
            constrained = constrain_hand_joint_targets(
                self._target_hand_joint_positions,
                self._mimic_pairs,
                self._hand_lower,
                self._hand_upper,
            )
            self._target_hand_joint_positions = self._hand_target_slew.condition(constrained)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Clear commanded-target history when the owning session/environment resets."""
        super().reset(env_ids)
        if self._hand_target_slew is not None:
            self._hand_target_slew.reset(env_ids)

    def _compute_ik_solutions(self) -> torch.Tensor:
        targets = super()._compute_ik_solutions()
        if not self._kitless_target_conditioning_enabled:
            return targets
        current = self._asset.data.joint_pos.torch[:, self._isaaclab_controlled_joint_ids]
        return bound_ik_joint_targets(
            targets,
            current,
            self.cfg.max_ik_joint_delta_rad,
            self.cfg.max_ik_joint_magnitude_rad,
        )
