# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right mirror symmetry for the 29-DoF G1, for symmetry-augmented training.

Measured on flat ground under a pinned straight command -- where every part of this robot should be
mirror-symmetric -- both ``w100`` seeds that learned to walk are grossly asymmetric, and in opposite
directions:

===================  ==========  ==========  ==========
metric                 seed 42     seed 43     seed 44
===================  ==========  ==========  ==========
``success_rate``        0.880       0.822    0.000 (never trained)
pelvis roll            -4.51 deg   +3.35 deg   -1.02 deg
knee L / R            +32.7/-5.3  -5.1/+27.8    symmetric
airborne share L/R       0.647       1.654       0.893
===================  ==========  ==========  ==========

One knee is bent thirty degrees more than the other for the whole episode, the pelvis is rolled to
one side and stays there, and which side is a coin flip per seed. The only symmetric seed is the one
that never learned to walk. Nothing in the task says the two legs should behave alike, so each run
breaks the symmetry its own way and keeps it.

Three purpose-built asymmetry *rewards* were tried and all failed (``y1``/``y2``/``y3``: 0.973 /
0.000 / 0.966 with the single-stance share collapsing from 0.70 to 0.48). They penalized the
*outcome* -- the airborne-share difference -- and the cheapest way to equalise that is to stop
lifting the feet. This module constrains the *policy* instead: every transition is presented to PPO
alongside its mirror image, so the network is pushed toward equivariance rather than paid for a
statistic.

**How the mirror is derived, rather than assumed.** Under a left-right reflection a joint about the
pitch axis keeps its sign and a joint about the roll or yaw axis flips it. That is asserted against
the asset instead of trusted: for a pair whose limits are asymmetric, ``lim_R`` must equal the
mirrored ``lim_L``, which pins the sign with no ambiguity (it fixes ``hip_roll``, ``shoulder_roll``
and all four finger flexion joints). Where both limits are symmetric the test cannot discriminate and
the axis in the joint name decides. The whole map is then checked by requiring that it leaves the
default pose unchanged -- the shoulders' +-0.16 rad roll makes that a real test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_SCAN_ROWS, _SCAN_COLS = 11, 17
"""Height-scan grid shape, ``(y, x)``.

Verified against the sensor's own ray starts rather than the config: index ``i`` is
``y_index * 17 + x_index``, x varying fastest, 17 columns spanning +-0.8 m and 11 rows spanning
+-0.5 m. A left-right mirror therefore flips the row axis.
"""

_VECTOR_MIRRORS = {
    # A reflection through the robot's own xz plane: y flips, and so do the rotations about x and z.
    "base_lin_vel": (1.0, -1.0, 1.0),
    "base_ang_vel": (-1.0, 1.0, -1.0),
    "projected_gravity": (1.0, -1.0, 1.0),
    # The command is (v_x, v_y, omega_z).
    "velocity_commands": (1.0, -1.0, -1.0),
}

_JOINT_TERMS = ("joint_pos", "joint_vel", "actions")

_PITCH_TOKENS = ("_pitch_", "knee", "elbow")
"""Name fragments whose joint turns about the pitch axis and so keeps its sign under the mirror."""

_FLIP_TOKENS = ("_roll_", "_yaw_")
"""Name fragments whose joint turns about the roll or yaw axis and so flips sign under the mirror."""

_EXPLICIT_SIGNS = {"hand_thumb_0_joint": 1.0}
"""Signs that neither the joint limits nor the name determine, measured instead.

``hand_thumb_0`` has a range symmetric about zero and no axis in its name. Driving both thumbs to
the *same* angle and comparing the thumb links in the base frame gives a mirror error of 0.00018 m,
against 0.00279 m for opposite angles -- so it keeps its sign, which is the opposite of what its two
flexion siblings do. The rest of the asset is mirror-symmetric to 1e-5 m at the default pose, which
is what makes this test meaningful."""

_CACHE: dict[tuple[str, ...], tuple[torch.Tensor, torch.Tensor]] = {}


def _counterpart(name: str) -> str:
    """Return the joint that ``name`` maps to under a left-right reflection."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def _sign_from_name(name: str) -> float | None:
    """Mirror sign implied by the axis in the joint's name, or None if the name does not say.

    The finger joints carry no axis, which is why the limits are consulted first: theirs are
    asymmetric and settle the question without a guess.
    """
    if any(token in name for token in _PITCH_TOKENS):
        return 1.0
    if any(token in name for token in _FLIP_TOKENS):
        return -1.0
    for suffix, value in _EXPLICIT_SIGNS.items():
        if name.endswith(suffix):
            return value
    return None


def _build_map(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and validate the joint permutation and sign vectors for this robot.

    Args:
        env: Environment whose ``robot`` articulation defines the joint order.

    Returns:
        ``(perm, sign)``, both of length ``num_joints``: the mirrored joint's index, and the factor
        its value is multiplied by.

    Raises:
        RuntimeError: If the derived map contradicts the asset's joint limits or default pose.
    """
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    key = tuple(names)
    if key in _CACHE:
        return _CACHE[key]

    device = env.device
    limits = robot.data.joint_pos_limits
    limits = (limits.torch if hasattr(limits, "torch") else limits)[0]
    default = robot.data.default_joint_pos
    default = (default.torch if hasattr(default, "torch") else default)[0]

    perm = torch.zeros(len(names), dtype=torch.long, device=device)
    sign = torch.ones(len(names), device=device)
    for i, name in enumerate(names):
        j = names.index(_counterpart(name))
        perm[i] = j
        # The limits decide wherever they can: a pair whose ranges are mirror images of each other
        # can only be a sign flip, and a pair whose ranges are identical and not symmetric can only
        # be a sign keep. Where both hold -- a range symmetric about zero -- they carry no
        # information and the axis in the name is the only evidence.
        lo_i, hi_i = float(limits[i, 0]), float(limits[i, 1])
        lo_j, hi_j = float(limits[j, 0]), float(limits[j, 1])
        same = abs(lo_i - lo_j) < 1e-4 and abs(hi_i - hi_j) < 1e-4
        flipped = abs(lo_i + hi_j) < 1e-4 and abs(hi_i + lo_j) < 1e-4
        by_name = _sign_from_name(name)
        if same and not flipped:
            by_limits = 1.0
        elif flipped and not same:
            by_limits = -1.0
        else:
            by_limits = None
        if by_limits is not None and by_name is not None and by_limits != by_name:
            raise RuntimeError(
                f"{name}: its limits say the mirror sign is {by_limits:+.0f} and its name says"
                f" {by_name:+.0f}; one of the two is wrong and guessing would corrupt training"
            )
        resolved = by_limits if by_limits is not None else by_name
        if resolved is None:
            raise RuntimeError(
                f"{name}: neither the limits nor the name determine the mirror sign; add it explicitly"
            )
        sign[i] = resolved

    mirrored_default = sign * default[perm]
    if not torch.allclose(mirrored_default, default, atol=1e-5):
        worst = int((mirrored_default - default).abs().argmax())
        raise RuntimeError(
            f"the mirror does not leave the default pose invariant; worst joint {names[worst]} "
            f"{float(default[worst]):+.4f} -> {float(mirrored_default[worst]):+.4f}"
        )
    if not torch.equal(perm[perm], torch.arange(len(names), device=device)):
        raise RuntimeError("the joint permutation is not an involution")

    _CACHE[key] = (perm, sign)
    return perm, sign


def _mirror_joint_block(block: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mirror one ``(batch, num_joints)`` block of joint-indexed values."""
    return block[:, perm] * sign


def _mirror_group(env: ManagerBasedRLEnv, group: str, obs: torch.Tensor) -> torch.Tensor:
    """Mirror one observation group, term by term.

    The term layout is read from the observation manager rather than hard-coded, so a config that
    drops ``base_lin_vel`` or the height scan stays correct.

    Args:
        env: The environment, for the observation manager.
        group: Observation group name.
        obs: The group's flat observation, shape ``(batch, dim)``.

    Returns:
        The mirrored observation, same shape.

    Raises:
        ValueError: If the group contains a term whose mirror is unknown.
    """
    perm, sign = _build_map(env)
    out = obs.clone()
    offset = 0
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    for name, shape in zip(names, dims):
        size = 1
        for s in shape:
            size *= int(s)
        block = obs[:, offset : offset + size]
        if name in _VECTOR_MIRRORS:
            factor = torch.tensor(_VECTOR_MIRRORS[name], device=obs.device)
            out[:, offset : offset + size] = block * factor.repeat(size // 3)
        elif name in _JOINT_TERMS:
            frames = size // perm.numel()
            reshaped = block.reshape(block.shape[0] * frames, perm.numel())
            out[:, offset : offset + size] = _mirror_joint_block(reshaped, perm, sign).reshape(block.shape)
        elif name == "height_scan":
            if size != _SCAN_ROWS * _SCAN_COLS:
                raise ValueError(f"height_scan is {size} values, not the {_SCAN_ROWS}x{_SCAN_COLS} grid")
            out[:, offset : offset + size] = (
                block.view(-1, _SCAN_ROWS, _SCAN_COLS).flip(dims=[1]).reshape(block.shape)
            )
        else:
            raise ValueError(f"no mirror defined for observation term {name!r}; refusing to guess")
        offset += size
    return out


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Return each state alongside its left-right mirror image.

    A biped has one symmetry, not the quadruped's four, so the batch doubles rather than quadruples.

    Args:
        env: The environment instance.
        obs: Observation tensor dictionary, or None.
        actions: Action tensor, or None.

    Returns:
        The augmented observations and actions, either of which is None if its input was.
    """
    unwrapped = env.unwrapped

    if obs is not None:
        batch = obs.batch_size[0]
        obs_aug = obs.repeat(2)
        for group in obs.keys():
            obs_aug[group][:batch] = obs[group]
            obs_aug[group][batch:] = _mirror_group(unwrapped, group, obs[group])
    else:
        obs_aug = None

    if actions is not None:
        perm, sign = _build_map(unwrapped)
        actions_aug = torch.cat([actions, _mirror_joint_block(actions, perm, sign)], dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
