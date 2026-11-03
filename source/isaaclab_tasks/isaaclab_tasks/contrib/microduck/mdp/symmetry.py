# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection for MicroDuck walking and Roulade.

Walking augmentation reflects named actor and critic terms. Roulade mirror loss reflects
its fixed 61-wide actor layout and repeats critic observations unchanged.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states", "compute_roulade_symmetric_states"]

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


MICRODUCK_ACTION_DIM = 14
"""Servos the policy drives, which is also the width of one observation joint block."""

MICRODUCK_OBSERVATION_DIM = 61
"""Width of the deployed actor observation the tables below describe."""

_ROULADE_JOINT_PERMUTATION = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
"""Left leg (0-4) swapped with right leg (9-13); the four midline neck and head servos stay."""

_ROULADE_JOINT_SIGN = [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
"""Sign applied after the swap. Only ``neck_pitch`` and ``head_pitch`` survive unchanged."""

_ROULADE_OBSERVATION_PERMUTATION = (
    [0, 1, 2]
    + [3, 4, 5]
    + [6 + joint for joint in _ROULADE_JOINT_PERMUTATION]
    + [20 + joint for joint in _ROULADE_JOINT_PERMUTATION]
    + [34 + joint for joint in _ROULADE_JOINT_PERMUTATION]
    + [48, 49, 50]
    + [51, 52, 53, 54]
    + [55, 56, 57, 58, 59, 60]
)
"""Column permutation of the 61-wide actor observation. Only the joint blocks move."""

_ROULADE_OBSERVATION_SIGN = (
    [-1.0, 1.0, -1.0]  # base angular velocity: negate roll and yaw
    + [1.0, -1.0, 1.0]  # projected gravity: negate y
    + _ROULADE_JOINT_SIGN  # joint positions
    + _ROULADE_JOINT_SIGN  # joint velocities
    + _ROULADE_JOINT_SIGN  # last action
    + [1.0, -1.0, -1.0]  # twist: negate lateral velocity and yaw rate
    + [1.0, 1.0, -1.0, -1.0]  # head command: negate yaw and roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body command: negate y, roll and yaw
)
"""Sign applied to each column after the permutation."""

_ROULADE_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _roulade_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four tables as tensors on a device, built once per device."""
    tables = _ROULADE_TABLE_CACHE.get(device)
    if tables is None:
        tables = (
            torch.tensor(_ROULADE_OBSERVATION_PERMUTATION, dtype=torch.long, device=device),
            torch.tensor(_ROULADE_OBSERVATION_SIGN, dtype=torch.float32, device=device),
            torch.tensor(_ROULADE_JOINT_PERMUTATION, dtype=torch.long, device=device),
            torch.tensor(_ROULADE_JOINT_SIGN, dtype=torch.float32, device=device),
        )
        _ROULADE_TABLE_CACHE[device] = tables
    return tables


@torch.no_grad()
def compute_roulade_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append the left-right mirror of a batch of observations and actions to itself.

    Use with actor mirror loss: the critic observations repeat unchanged.

    Args:
        env: The environment instance. Unused: the layout is a fixed deploy contract rather than
            something read off the scene, and pinning it here is what makes a drifted observation
            group fail loudly instead of training against a wrong mirror.
        obs: The observations, keyed by group. Defaults to None, which skips them.
        actions: The actions. Shape is (batch, 14). Defaults to None, which skips them.

    Returns:
        The augmented observations and actions, each twice as long along the batch axis with the
        originals first and their mirrors second, or None where the input was None.

    Raises:
        ValueError: If the policy group or the action vector is not the width the tables describe.
    """
    del env

    augmented_obs = None
    if obs is not None:
        policy = obs["policy"]
        if policy.shape[-1] != MICRODUCK_OBSERVATION_DIM:
            raise ValueError(
                "The MicroDuck symmetry tables describe the 61-wide deploy observation; the policy"
                f" group is {policy.shape[-1]} wide. Update the tables together with the layout."
            )
        obs_permutation, obs_sign, _, _ = _roulade_tables(policy.device)
        batch_size = policy.shape[0]

        augmented_obs = obs.repeat(2)
        augmented_obs["policy"][batch_size:] = policy[:, obs_permutation] * obs_sign

    augmented_actions = None
    if actions is not None:
        if actions.shape[-1] != MICRODUCK_ACTION_DIM:
            raise ValueError(
                "The MicroDuck symmetry tables describe the 14 servos; the action vector is"
                f" {actions.shape[-1]} wide. Update the tables together with the action term."
            )
        _, _, joint_permutation, joint_sign = _roulade_tables(actions.device)
        augmented_actions = torch.cat((actions, actions[:, joint_permutation] * joint_sign), dim=0)

    return augmented_obs, augmented_actions
