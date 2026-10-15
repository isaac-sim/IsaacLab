# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import torch

from isaaclab_contrib.mdp import mirror_joints, mirror_quat, mirror_vec3


def mirror_g1_joints(data: torch.Tensor, joint_names: Sequence[str]) -> torch.Tensor:
    """Reflect G1 joint values in the explicitly configured joint order.

    Roll and yaw axes change sign; pitch, knee, and elbow axes keep their signs.
    Both members of each left/right pair must be present. Values retain their units.
    """
    permutation, signs = _joint_mirror(tuple(joint_names))
    return mirror_joints(data, permutation, signs)


def mirror_velocity_heading(data: torch.Tensor) -> torch.Tensor:
    """Reflect ``[vx, vy, wz, qx, qy, qz, qw]`` commands, including the heading."""
    if data.shape[-1] != 7:
        raise ValueError("Expected three velocity commands followed by an XYZW heading.")
    velocity = data[..., :3] * data.new_tensor((1, -1, -1))
    return torch.cat((velocity, mirror_quat(data[..., 3:])), dim=-1)


def mirror_foot_scalars(data: torch.Tensor) -> torch.Tensor:
    """Swap the two foot scalar channels, preserving their units."""
    return mirror_joints(data, (1, 0))


def mirror_foot_forces(data: torch.Tensor) -> torch.Tensor:
    """Swap feet and reflect flattened force vectors (including signed-log forces)."""
    if data.shape[-1] != 6:
        raise ValueError("Expected two three-component foot forces.")
    vectors = data.reshape(*data.shape[:-1], 2, 3).flip(-2)
    return mirror_vec3(vectors).reshape_as(data)


@lru_cache(maxsize=32)
def _joint_mirror(joint_names: tuple[str, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    indices = {name: i for i, name in enumerate(joint_names)}
    if len(indices) != len(joint_names):
        raise ValueError("Joint names must be unique.")
    permutation, signs = [], []
    for name in joint_names:
        if name.startswith("left_"):
            partner = "right_" + name.removeprefix("left_")
        elif name.startswith("right_"):
            partner = "left_" + name.removeprefix("right_")
        else:
            partner = name
        if partner not in indices:
            raise ValueError(f"Missing mirror joint {partner!r} for {name!r}.")
        if name.endswith(("_roll_joint", "_yaw_joint")):
            sign = -1
        elif name.endswith(("_pitch_joint", "_knee_joint", "_elbow_joint")):
            sign = 1
        else:
            raise ValueError(f"No G1 mirror axis rule for joint {name!r}.")
        permutation.append(indices[partner])
        signs.append(sign)
    return tuple(permutation), tuple(signs)
