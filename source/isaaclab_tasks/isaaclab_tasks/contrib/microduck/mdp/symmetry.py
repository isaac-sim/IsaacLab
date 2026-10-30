# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection for MicroDuck's fixed servo and deployed observation order."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Both actions and joint observations explicitly use MICRODUCK_JOINT_NAMES with preserve_order=True.
# The USD's mirrored leg axes all reverse sign; neck/head pitch retain sign, yaw/roll reverse it.
_JOINT_PERMUTATION = (9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4)
_JOINT_SIGNS = (-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1)


@functools.cache
def _reflection(group: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache feature permutations and polar/axial vector signs on the input device."""
    size = {"policy": 61, "critic": 76, "actions": 14}[group]
    permutation = list(range(size))
    signs = [1] * size

    def joints(start):
        permutation[start : start + 14] = [start + index for index in _JOINT_PERMUTATION]
        signs[start : start + 14] = _JOINT_SIGNS

    if group == "actions":
        joints(0)
    else:
        if group == "policy":
            signs[:6] = [-1, 1, -1, 1, -1, 1]  # angular velocity (axial), gravity (polar)
            joint_start, command_start, head_start, body_start = 6, 48, 51, 55
        else:
            signs[:9] = [1, -1, 1, -1, 1, -1, 1, -1, 1]  # linear, angular, gravity
            joint_start, command_start, head_start, body_start = 9, 51, 66, 70
            for start in (54, 56, 58):  # left/right height, air time, contact
                permutation[start : start + 2] = [start + 1, start]
            permutation[60:66] = [63, 64, 65, 60, 61, 62]
            signs[60:66] = [1, -1, 1, 1, -1, 1]  # world contact forces are polar vectors
        for start in (joint_start, joint_start + 14, joint_start + 28):
            joints(start)
        signs[command_start : command_start + 3] = [1, -1, -1]
        signs[head_start : head_start + 4] = [1, 1, -1, -1]
        signs[body_start : body_start + 6] = [1, -1, 1, -1, 1, -1]  # translation, rotation
    return torch.tensor(permutation, device=device), torch.tensor(signs, device=device)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append left/right mirrored samples for RSL-RL augmentation and mirror-loss evaluation.

    Inputs remain unchanged. Observations must follow the policy's 61-value or critic's 76-value
    deploy contract. Actions contain 14 servo offsets [rad], in ``MICRODUCK_JOINT_NAMES`` order.
    The environment argument is accepted for the RSL-RL symmetry callback interface.
    """
    obs_aug = None
    if obs is not None:
        obs_aug = obs.repeat(2)
        for group, values in obs.items():
            if group not in ("policy", "critic") or values.shape[-1] != {"policy": 61, "critic": 76}[group]:
                raise ValueError(f"Unsupported MicroDuck observation layout for {group!r}: {values.shape}.")
            permutation, signs = _reflection(group, str(values.device))
            obs_aug[group][obs.batch_size[0] :] = values[..., permutation] * signs
    actions_aug = None
    if actions is not None:
        if actions.shape[-1] != 14:
            raise ValueError(f"Expected 14 MicroDuck servo actions, got {actions.shape}.")
        permutation, signs = _reflection("actions", str(actions.device))
        actions_aug = torch.cat((actions, actions[..., permutation] * signs), dim=0)
    return obs_aug, actions_aug
