# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal navigation reflection, following the native MicroDuck walker symmetry convention."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Joint observations preserve MICRODUCK_JOINT_NAMES: mirrored leg axes reverse sign;
# neck/head pitch retain sign, while head yaw/roll reverse it.
_JOINT_PERMUTATION = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_JOINT_SIGNS = (-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1)


@functools.cache
def _reflection(group: str, size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache the fixed feature mapping, including the configured ToF history length."""
    permutation = list(range(size))
    signs = [1] * size
    if group == "tof":
        # Observation history is term-major: H frames of ranges+validity, followed by H ages.
        history = size // 129
        zones = torch.arange(history * 128, device=device).reshape(history, 2, 8, 8)
        indices = torch.arange(size, device=device)
        indices[: history * 128] = zones.flip(-1).flatten()
        return indices, torch.ones(size, dtype=torch.long, device=device)
    if group == "actions":
        signs[:] = [1, -1, -1]
    else:
        if group == "policy":
            signs[:6] = [-1, 1, -1, 1, -1, 1]  # Angular velocity (axial), gravity (polar).
            joint_start, action_start, command_start, head_start, body_start = 6, 34, 37, 40, 44
            signs[50:53] = [1, -1, -1]  # Commands actually delivered to the frozen walker.
        else:
            signs[:9] = [1, -1, 1, -1, 1, -1, 1, -1, 1]  # Linear, angular, gravity.
            joint_start, action_start, command_start, head_start, body_start = 9, 37, 40, 55, 59
            for start in (43, 45, 47):  # Left/right foot height, air time, contact.
                permutation[start : start + 2] = [start + 1, start]
            permutation[49:55] = [52, 53, 54, 49, 50, 51]
            signs[49:55] = [1, -1, 1, 1, -1, 1]  # Contact forces are polar vectors.
            # Critic clearances [65:73] retain obstacle identity under whole-scene reflection.
        for start in (joint_start, joint_start + 14):
            permutation[start : start + 14] = [start + index for index in _JOINT_PERMUTATION]
            signs[start : start + 14] = _JOINT_SIGNS
        signs[action_start : action_start + 3] = [1, -1, -1]
        signs[command_start : command_start + 3] = [1, -1, -1]
        signs[head_start : head_start + 4] = [1, 1, -1, -1]
        signs[body_start : body_start + 6] = [1, -1, 1, -1, 1, -1]
    return torch.tensor(permutation, device=device), torch.tensor(signs, device=device)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append mirrored raw observations/actions for the native RSL-RL symmetry callback.

    Navigation groups have 53 policy features, 73 critic features, and 129 ToF features
    per history frame. Range and validity columns reverse within each frame; temporal
    order and ages stay fixed. Actions are forward braking, lateral, and yaw residuals.
    Inputs remain unchanged, and reflection precedes the actor's existing normalization.
    """
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        for group, values in obs.items():
            size = values.shape[-1]
            if group == "tof":
                if size == 0 or size % 129:
                    raise ValueError(f"Expected 129 ToF features per history frame, got {values.shape}.")
            elif group not in ("policy", "critic") or size != {"policy": 53, "critic": 73}[group]:
                raise ValueError(f"Unsupported MicroDuck navigation layout for {group!r}: {values.shape}.")
            permutation, signs = _reflection(group, size, str(values.device))
            obs_aug[group][obs.batch_size[0] :] = values[..., permutation] * signs
    actions_aug = None
    if actions is not None:
        if actions.shape[-1] != 3:
            raise ValueError(f"Expected 3 MicroDuck navigation actions, got {actions.shape}.")
        permutation, signs = _reflection("actions", 3, str(actions.device))
        actions_aug = torch.cat((actions, actions[..., permutation] * signs), dim=0)
    return obs_aug, actions_aug
