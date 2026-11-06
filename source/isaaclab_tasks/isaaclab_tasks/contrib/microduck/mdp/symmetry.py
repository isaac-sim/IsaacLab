# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection for MicroDuck's observation terms and fixed servo order."""

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

# Signs within named terms: angular vectors are axial; gravity, translation and forces are polar.
_TERM_SIGNS = {
    "imu": (-1, 1, -1, 1, -1, 1),
    "base_lin_vel": (1, -1, 1),
    "base_ang_vel": (-1, 1, -1),
    "projected_gravity": (1, -1, 1),
    "velocity_commands": (1, -1, -1),
    "head_pose_commands": (1, 1, -1, -1),
    "body_pose_commands": (1, -1, 1, -1, 1, -1),
}


@functools.cache
def _reflection(layout: tuple[tuple[str, tuple[int, ...]], ...], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Cache permutations by manager term layout and device, including same-width reorderings."""
    permutation, signs = [], []
    for name, shape in layout:
        if name in ("joint_pos", "joint_vel", "actions"):
            term_permutation, term_signs = _JOINT_PERMUTATION, _JOINT_SIGNS
        elif name in ("foot_height", "foot_air_time", "foot_contact"):
            term_permutation, term_signs = (1, 0), (1, 1)
        elif name == "foot_contact_forces":
            term_permutation, term_signs = (3, 4, 5, 0, 1, 2), (1, -1, 1, 1, -1, 1)
        elif name in _TERM_SIGNS:
            term_signs = _TERM_SIGNS[name]
            term_permutation = range(len(term_signs))
        else:
            raise ValueError(f"Unsupported MicroDuck observation term {name!r}.")
        if shape != (len(term_signs),):
            raise ValueError(f"Unsupported MicroDuck observation shape for {name!r}: {shape}.")
        start = len(permutation)
        permutation.extend(start + index for index in term_permutation)
        signs.extend(term_signs)
    return torch.tensor(permutation, device=device, dtype=torch.long), torch.tensor(signs, device=device)


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv | None,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append left/right mirrored samples for RSL-RL augmentation and mirror-loss evaluation.

    Inputs remain unchanged. Observation columns are resolved from the environment's observation
    manager; unknown terms or unsupported term shapes are rejected. Actions contain 14 servo offsets
    [rad], in ``MICRODUCK_JOINT_NAMES`` order. The environment may be ``None`` for actions-only calls.
    """
    obs_aug = None
    if obs is not None:
        if env is None:
            raise ValueError("MicroDuck observation reflection requires an environment observation manager.")
        manager = env.unwrapped.observation_manager
        obs_aug = obs.repeat(2)
        for group, values in obs.items():
            if group not in ("policy", "critic"):
                raise ValueError(f"Unsupported MicroDuck observation group {group!r}.")
            layout = tuple(zip(manager.active_terms[group], manager.group_obs_term_dim[group], strict=True))
            permutation, signs = _reflection(layout, str(values.device))
            if values.shape[-1] != len(permutation):
                raise ValueError(f"Unsupported MicroDuck observation layout for {group!r}: {values.shape}.")
            obs_aug[group][obs.batch_size[0] :] = values[..., permutation] * signs
    actions_aug = None
    if actions is not None:
        if actions.shape[-1] != 14:
            raise ValueError(f"Expected 14 MicroDuck servo actions, got {actions.shape}.")
        permutation, signs = _reflection((("actions", (14,)),), str(actions.device))
        actions_aug = torch.cat((actions, actions[..., permutation] * signs), dim=0)
    return obs_aug, actions_aug
