# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry of the shared MicroDuck observation and action spaces.

Ported from ``pollen-robotics/microduck_rl``'s ``tasks/symmetry.py`` (addendum section 4.11). The
tables describe the whole policy family's 61-wide deploy contract, not one task, because that vector
is identical everywhere; the forward-roll task is the only one that switches the augmentation on,
and it does so with the mirror loss rather than with data augmentation.

Actor layout, in the manager's term order:

===========  ===========================================================
slice        term
===========  ===========================================================
``[0:3]``    base angular velocity, body frame
``[3:6]``    projected gravity, body frame
``[6:20]``   joint positions relative to the stand pose, 14 servos
``[20:34]``  joint velocities, 14 servos
``[34:48]``  last action, 14 servos
``[48:51]``  twist command ``(v_x, v_y, w_z)``
``[51:55]``  head-pose command
``[55:61]``  body-pose command
===========  ===========================================================

The mirror is the reflection about the robot's sagittal plane:

* the left leg block swaps with the right leg block and the four midline neck and head servos stay
  put;
* every swapped joint negates. The yaw and roll axes reverse under a left-right reflection, and the
  pitch joints negate too because the stand pose itself is mirrored in sign -- ``left_hip_pitch`` is
  -0.458 rad where ``right_hip_pitch`` is +0.458 -- so their *relative* deviations mirror with a
  sign change as well. ``neck_pitch`` and ``head_pitch`` are sagittal and keep their sign;
* the base angular velocity negates its roll and yaw components, the projected gravity negates its
  ``y``, and each command block negates the axes that are themselves lateral.

.. note::
    Upstream mirrors only the actor observation and repeats the critic unchanged, which is sound for
    the mirror loss -- the only mode enabled -- because that loss is defined on the policy mean and
    never reads the critic group. Enabling data augmentation with these tables would train the value
    function on mirrored transitions labelled with unmirrored privileged observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensordict import TensorDict

    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

MICRODUCK_ACTION_DIM = 14
"""Servos the policy drives, which is also the width of one observation joint block."""

MICRODUCK_OBSERVATION_DIM = 61
"""Width of the deployed actor observation the tables below describe."""

_JOINT_PERMUTATION = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
"""Left leg (0-4) swapped with right leg (9-13); the four midline neck and head servos stay."""

_JOINT_SIGN = [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
"""Sign applied after the swap. Only ``neck_pitch`` and ``head_pitch`` survive unchanged."""

_OBSERVATION_PERMUTATION = (
    [0, 1, 2]
    + [3, 4, 5]
    + [6 + joint for joint in _JOINT_PERMUTATION]
    + [20 + joint for joint in _JOINT_PERMUTATION]
    + [34 + joint for joint in _JOINT_PERMUTATION]
    + [48, 49, 50]
    + [51, 52, 53, 54]
    + [55, 56, 57, 58, 59, 60]
)
"""Column permutation of the 61-wide actor observation. Only the joint blocks move."""

_OBSERVATION_SIGN = (
    [-1.0, 1.0, -1.0]  # base angular velocity: negate roll and yaw
    + [1.0, -1.0, 1.0]  # projected gravity: negate y
    + _JOINT_SIGN  # joint positions
    + _JOINT_SIGN  # joint velocities
    + _JOINT_SIGN  # last action
    + [1.0, -1.0, -1.0]  # twist: negate lateral velocity and yaw rate
    + [1.0, 1.0, -1.0, -1.0]  # head command: negate yaw and roll
    + [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]  # body command: negate y, roll and yaw
)
"""Sign applied to each column after the permutation."""

_TABLE_CACHE: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The four tables as tensors on a device, built once per device."""
    tables = _TABLE_CACHE.get(device)
    if tables is None:
        tables = (
            torch.tensor(_OBSERVATION_PERMUTATION, dtype=torch.long, device=device),
            torch.tensor(_OBSERVATION_SIGN, dtype=torch.float32, device=device),
            torch.tensor(_JOINT_PERMUTATION, dtype=torch.long, device=device),
            torch.tensor(_JOINT_SIGN, dtype=torch.float32, device=device),
        )
        _TABLE_CACHE[device] = tables
    return tables


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Append the left-right mirror of a batch of observations and actions to itself.

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
        obs_permutation, obs_sign, _, _ = _tables(policy.device)
        batch_size = policy.shape[0]
        # ``repeat`` copies every group, so the critic is carried through unmirrored, as upstream
        # does: the mirror loss reads the actor's action means only.
        augmented_obs = obs.repeat(2)
        augmented_obs["policy"][batch_size:] = policy[:, obs_permutation] * obs_sign

    augmented_actions = None
    if actions is not None:
        if actions.shape[-1] != MICRODUCK_ACTION_DIM:
            raise ValueError(
                "The MicroDuck symmetry tables describe the 14 servos; the action vector is"
                f" {actions.shape[-1]} wide. Update the tables together with the action term."
            )
        _, _, joint_permutation, joint_sign = _tables(actions.device)
        augmented_actions = torch.cat((actions, actions[:, joint_permutation] * joint_sign), dim=0)

    return augmented_obs, augmented_actions
