# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right symmetry augmentation for G1 body-joint policies.

Every supported observation term and the action are mirrored by a signed permutation of their columns.
Those permutations are built and validated once per environment over the joints the terms actually select,
so passive joints outside the policy (for example the Dex3 fingers) are never inspected. Each PPO
minibatch then costs one gather and one multiply per group.
"""

from __future__ import annotations

import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_SCAN_ROWS, _SCAN_COLS = 11, 17
"""Height-scan grid in (y, x) order; mirroring flips the 11 lateral rows."""

_VECTOR_SIGNS = {
    # A reflection through the robot's xz plane flips y and the rotations about x and z.
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    # The command is (v_x, v_y, omega_z).
    "velocity_commands": (1.0, -1.0, -1.0),
}

_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")

_PITCH_TOKENS = ("_pitch_", "knee", "elbow")
"""Name fragments of joints that turn about the pitch axis and so keep their sign under the mirror."""

_FLIP_TOKENS = ("_roll_", "_yaw_")
"""Name fragments of joints that turn about the roll or yaw axis and so flip sign under the mirror."""

_EXPLICIT_SIGNS = {"hand_thumb_0_joint": 1.0}
"""Joints whose axis the name does not reveal; thumb bases keep their sign under the mirror."""

_TOL = 1e-4

_MIRRORS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
"""Validated column mirrors per environment, keyed by observation group or ``None`` for the action."""


def _counterpart(name: str) -> str:
    """Return the joint that ``name`` maps to under a left-right reflection."""
    for side, other in (("left_", "right_"), ("right_", "left_")):
        if name.startswith(side):
            return other + name[len(side) :]
    return name


def _sign_from_name(name: str) -> float | None:
    """Return the mirror sign implied by the joint axis, or None if the name does not reveal it."""
    if any(token in name for token in _PITCH_TOKENS):
        return 1.0
    if any(token in name for token in _FLIP_TOKENS):
        return -1.0
    return next((value for suffix, value in _EXPLICIT_SIGNS.items() if name.endswith(suffix)), None)


def _sign_from_limits(limit: Sequence[float], other: Sequence[float]) -> float | None:
    """Return the mirror sign implied by a joint's limits and its counterpart's, or None if ambiguous."""
    same = abs(limit[0] - other[0]) < _TOL and abs(limit[1] - other[1]) < _TOL
    flipped = abs(limit[0] + other[1]) < _TOL and abs(limit[1] + other[0]) < _TOL
    if same != flipped:
        return 1.0 if same else -1.0
    return None


def _joint_mirror(env: ManagerBasedRLEnv, ids: list[int]) -> tuple[list[int], list[float]]:
    """Return the validated mirror of the selected joints, indexed in the selection's own order.

    Only the selected joints are checked: the selection must be closed under reflection, every sign must
    be determined without contradiction, and the mirror must leave the selected default pose unchanged.
    """
    robot = env.scene["robot"]
    all_names = list(robot.joint_names)
    names = [all_names[i] for i in ids]
    if not names or len(set(names)) != len(names) or any(_counterpart(n) not in names for n in names):
        raise ValueError(f"G1 joint selection must be nonempty, unique, and closed under reflection: {names}")

    limits = robot.data.joint_pos_limits.torch[0, ids].tolist()
    default = robot.data.default_joint_pos.torch[0, ids].tolist()
    perm = [names.index(_counterpart(n)) for n in names]
    sign = []
    for i, (name, j) in enumerate(zip(names, perm)):
        by_limits = _sign_from_limits(limits[i], limits[j])
        by_name = _sign_from_name(name)
        if by_limits is not None and by_name is not None and by_limits != by_name:
            raise RuntimeError(
                f"{name}: its limits say the mirror sign is {by_limits:+.0f} and its name says {by_name:+.0f};"
                " one of the two is wrong and guessing would corrupt training"
            )
        resolved = by_limits if by_limits is not None else by_name
        if resolved is None:
            raise RuntimeError(f"{name}: neither the limits nor the name determine the mirror sign; add it explicitly")
        sign.append(resolved)

    error = [abs(sign[i] * default[perm[i]] - default[i]) for i in range(len(names))]
    worst = max(range(len(names)), key=error.__getitem__)
    if error[worst] > 1e-5:
        raise RuntimeError(
            f"the mirror does not leave the default pose invariant; worst joint {names[worst]}"
            f" {default[worst]:+.4f} -> {sign[worst] * default[perm[worst]]:+.4f}"
        )
    return perm, sign


def _action_joint_ids(env: ManagerBasedRLEnv) -> list[int]:
    """Return the articulation indices of the policy's action joints, in articulation order.

    Joint observations must select the same joints (as the G1 velocity configs do); both then resolve to
    articulation order, and a different selection is caught by the size check in the caller.
    """
    if env.action_manager.active_terms != ["joint_pos"]:
        raise ValueError("G1 symmetry expects one joint_pos action term")
    cfg = env.action_manager.get_term("joint_pos").cfg
    if cfg.preserve_order:
        raise ValueError("G1 symmetry expects the joint_pos action in articulation order")
    ids, _ = env.scene["robot"].find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
    return list(ids)


def _to_device(env: ManagerBasedRLEnv, perm: list[int], sign: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Check that a column mirror is an involution and move it to the environment's device."""
    if any(perm[perm[i]] != i or sign[i] * sign[perm[i]] != 1.0 for i in range(len(perm))):
        raise RuntimeError("the mirror is not an involution")
    return torch.tensor(perm, dtype=torch.long, device=env.device), torch.tensor(sign, device=env.device)


def _build_group_mirror(env: ManagerBasedRLEnv, group: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose one signed column permutation for a concatenated observation group, including history."""
    manager = env.observation_manager
    perm: list[int] = []
    sign: list[float] = []
    for name, shape in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
        offset, size = len(perm), 1
        for s in shape:
            size *= int(s)
        if name in _VECTOR_SIGNS:
            if size % 3:
                raise ValueError(f"{group}.{name}: {size} values are not a sequence of 3-vectors")
            local_perm, local_sign = list(range(3)), list(_VECTOR_SIGNS[name])
        elif name in _JOINT_TERMS:
            local_perm, local_sign = _joint_mirror(env, _action_joint_ids(env))
            if size % len(local_perm):
                raise ValueError(f"{group}.{name}: observation size {size} does not match its joint selection")
        elif name == "height_scan":
            if size != _SCAN_ROWS * _SCAN_COLS:
                raise ValueError(f"height_scan is {size} values, not the {_SCAN_ROWS}x{_SCAN_COLS} grid")
            local_perm = [(_SCAN_ROWS - 1 - k // _SCAN_COLS) * _SCAN_COLS + k % _SCAN_COLS for k in range(size)]
            local_sign = [1.0] * size
        else:
            raise ValueError(f"no mirror defined for observation term {name!r}; refusing to guess")
        # History frames are stored frame-major, so the per-frame mirror repeats with a stride.
        width = len(local_perm)
        for frame in range(size // width):
            perm += [offset + frame * width + k for k in local_perm]
            sign += local_sign
    return _to_device(env, perm, sign)


def _mirror(env: ManagerBasedRLEnv, group: str | None, data: torch.Tensor) -> torch.Tensor:
    """Mirror an observation ``group``, or the action if None, building its column mirror on first use."""
    cache = _MIRRORS.get(env)
    if cache is None:
        cache = _MIRRORS[env] = {}
    if group not in cache:
        if group is None:
            cache[group] = _to_device(env, *_joint_mirror(env, _action_joint_ids(env)))
        else:
            cache[group] = _build_group_mirror(env, group)
    perm, sign = cache[group]
    if data.shape[-1] != perm.numel():
        raise ValueError(f"{group or 'actions'}: expected {perm.numel()} columns to mirror, got {data.shape[-1]}")
    return data[:, perm] * sign


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return each state alongside its left-right mirror image.

    The augmented batch contains the original samples followed by their reflections.

    Args:
        env: The environment instance.
        obs: Observation tensor dictionary, or None.
        actions: Action tensor, or None.

    Returns:
        The augmented observations and actions, either of which is None if its input was.
    """
    unwrapped = env.unwrapped

    obs_aug = None
    if obs is not None:
        batch = obs.batch_size[0]
        # repeat() already holds the originals in the first half.
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][batch:] = _mirror(unwrapped, group, obs[group])

    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat([actions, _mirror(unwrapped, None, actions)], dim=0)

    return obs_aug, actions_aug
