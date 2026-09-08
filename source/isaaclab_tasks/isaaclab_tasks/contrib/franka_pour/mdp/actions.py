# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Filtered arm and binary symmetric-gripper actions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.envs.mdp.actions import BinaryJointPositionAction, RelativeJointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .actions_cfg import (
        CurriculumGripperPositionActionCfg,
        EMARelativeJointPositionActionCfg,
    )

_GRIPPER_POSITION_TOLERANCE = 1.0e-6


class EMARelativeJointPositionAction(RelativeJointPositionAction):
    """Low-pass filter relative joint deltas without changing their policy-space meaning."""

    cfg: EMARelativeJointPositionActionCfg

    def __init__(self, cfg: EMARelativeJointPositionActionCfg, env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._alpha = float(cfg.alpha)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        self._previous_delta = torch.zeros_like(self._processed_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        """Affine-map the raw action, then smooth only the commanded joint delta."""
        super().process_actions(actions)
        self._processed_actions.lerp_(self._previous_delta, 1.0 - self._alpha)
        self._previous_delta.copy_(self._processed_actions)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        """Clear selected command history so a reset pose receives exactly zero delta."""
        selected = slice(None) if env_ids is None else env_ids
        super().reset(selected)
        self._processed_actions[selected] = 0.0
        self._previous_delta[selected] = 0.0


class CurriculumGripperPositionAction(BinaryJointPositionAction):
    """Filtered binary symmetric finger-position command with grasp-contact state."""

    cfg: CurriculumGripperPositionActionCfg

    def __init__(self, cfg: CurriculumGripperPositionActionCfg, env: ManagerBasedEnv) -> None:
        binary_cfg = cfg.copy()
        binary_cfg.open_command_expr = {".*": float(cfg.neutral_position)}
        binary_cfg.close_command_expr = {".*": float(cfg.close_position)}
        super().__init__(binary_cfg, env)
        self._joint_ids_torch = wp.to_torch(self._joint_ids)
        self._alpha = float(cfg.alpha)
        close_position = float(cfg.close_position)
        self._neutral_position = float(cfg.neutral_position)
        default_position = close_position if cfg.default_position is None else float(cfg.default_position)
        self._contact_min_deflection = float(cfg.contact_min_deflection)
        if not 0.0 < self._alpha <= 1.0:
            raise ValueError(f"Moving-average weight must lie in (0, 1], got {self._alpha}.")
        if (
            not math.isfinite(close_position)
            or not math.isfinite(self._neutral_position)
            or not close_position <= self._neutral_position
        ):
            raise ValueError("Curriculum gripper positions must be finite with close_position <= neutral_position.")
        if not math.isfinite(default_position) or not close_position <= default_position <= self._neutral_position:
            raise ValueError("default_position must lie in [close_position, neutral_position].")
        if not math.isfinite(self._contact_min_deflection) or self._contact_min_deflection <= 0.0:
            raise ValueError("contact_min_deflection must be finite and positive.")
        self._processed_actions.fill_(default_position)

    @property
    def commanded_position(self) -> torch.Tensor:
        """Current symmetric per-finger position target [m]."""
        return self._processed_actions[:, :1]

    @property
    def contact_deflection(self) -> torch.Tensor:
        """Per-finger position-drive deflection caused by contact [m]."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids_torch]
        finite = torch.isfinite(joint_position) & torch.isfinite(self._processed_actions)
        return torch.where(
            finite,
            torch.clamp(joint_position - self._processed_actions, min=0.0),
            torch.zeros_like(joint_position),
        )

    @property
    def bilateral_contact(self) -> torch.Tensor:
        """Whether both fingers remain deflected against the commanded cup contact."""
        joint_position = self._asset.data.joint_pos.torch[:, self._joint_ids_torch]
        deflection = self.contact_deflection
        finite = torch.isfinite(joint_position).all(dim=-1) & torch.isfinite(self._processed_actions).all(dim=-1)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return finite & command_valid & (deflection.amin(dim=-1) >= self._contact_min_deflection)

    @property
    def contact_quality(self) -> torch.Tensor:
        """Smooth bilateral drive-deflection quality in ``[0, 1]``.

        Finger velocity is deliberately excluded.  It is useful when deciding whether a grasp has
        settled, but it is not a measure of whether the cup is physically between the fingers.
        Including it in dense shaping creates a large artificial potential drop while a restored
        contact relaxes during its first simulation step.
        """
        deflection = self.contact_deflection
        deflection_quality = torch.clamp(deflection.amin(dim=-1) / self._contact_min_deflection, 0.0, 1.0)
        command_valid = self._processed_actions.amax(dim=-1) <= (self._neutral_position + _GRIPPER_POSITION_TOLERANCE)
        return deflection_quality * command_valid.float()

    def set_reset_position(
        self,
        position: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Align the filtered target with selected physical reset positions [m]."""
        selected = slice(None) if env_ids is None else env_ids
        expected_shape = self._raw_actions[selected].shape
        if position.shape != expected_shape:
            raise ValueError(f"Reset-position shape {tuple(position.shape)} does not match {tuple(expected_shape)}.")
        position = position.to(device=self.device, dtype=self._processed_actions.dtype)
        expanded = position.expand(-1, self._num_joints)
        self._processed_actions[selected] = expanded

    def process_actions(self, actions: torch.Tensor) -> None:
        previous_target = self._processed_actions
        super().process_actions(actions)
        previous_target.lerp_(self._processed_actions, self._alpha)
        self._processed_actions = previous_target
