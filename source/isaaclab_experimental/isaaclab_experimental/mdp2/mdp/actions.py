# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms, the MDP2 counterparts of the action terms of :mod:`isaaclab.envs.mdp.actions`.

An action term gets its columns of the action as ``action`` and writes its ``outputs``. Actions that only scale and
offset the action into a field are :class:`~isaaclab_experimental.mdp2.DirectAction` s, which need no term.
"""

from __future__ import annotations

import torch
import warp as wp

from ..terms import Term


@wp.func
def _binary_joint_position(
    world: int,
    action: wp.array2d(dtype=float),
    joint_pos_target: wp.array2d(dtype=float),
    joint_ids: wp.array(dtype=wp.int32),
    open_position: float,
    close_position: float,
):
    target = open_position
    if action[world, 0] > 0.5:
        target = close_position
    for joint in range(joint_ids.shape[0]):
        joint_pos_target[world, joint_ids[joint]] = target


def _binary_joint_position_torch(
    action: torch.Tensor,
    joint_pos_target: torch.Tensor,
    joint_ids: torch.Tensor,
    open_position: float,
    close_position: float,
    mask: torch.Tensor | None,
) -> None:
    target = torch.where(action[:, :1] > 0.5, close_position, open_position).expand(-1, len(joint_ids))
    if mask is not None:
        target = torch.where(mask[:, None], target, joint_pos_target[:, joint_ids])
    joint_pos_target[:, joint_ids] = target


binary_joint_position = Term(torch=_binary_joint_position_torch, warp=_binary_joint_position, shape=("worlds", 1))
"""Binary gripper command, like MDP1's ``BinaryJointPositionAction`` with a 0 to 1 action: a value above 0.5 sets the
target of the joints of ``joint_ids`` to ``close_position``, and any other value to ``open_position``. Bind
``joint_pos_target`` as an output."""
