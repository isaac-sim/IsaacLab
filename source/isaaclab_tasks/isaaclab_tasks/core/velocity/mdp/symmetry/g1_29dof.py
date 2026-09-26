# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Left-right augmentation for G1 body-joint policies and height observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["compute_symmetric_states"]

_SCAN_ROWS, _SCAN_COLS = 11, 17
# Grid ordering is (y, x); mirroring flips the 11 lateral rows.

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
# Thumb base axes were verified geometrically; equal angles produce mirrored poses.


def _counterpart(name: str) -> str:
    """Return the joint that ``name`` maps to under a left-right reflection."""
    if name.startswith("left_"):
        return "right_" + name[len("left_") :]
    if name.startswith("right_"):
        return "left_" + name[len("right_") :]
    return name


def _sign_from_name(name: str) -> float | None:
    """Return the mirror sign implied by the joint axis, or None for unnamed axes."""
    if any(token in name for token in _PITCH_TOKENS):
        return 1.0
    if any(token in name for token in _FLIP_TOKENS):
        return -1.0
    for suffix, value in _EXPLICIT_SIGNS.items():
        if name.endswith(suffix):
            return value
    return None


def _build_map(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Build per-environment joint mirrors, validating limits, default pose, and involution."""
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    cached = getattr(env, "_g1_joint_mirror_map", None)
    if cached is not None:
        return cached

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
        # Asymmetric limits determine the sign; otherwise use the named joint axis.
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
            raise RuntimeError(f"{name}: neither the limits nor the name determine the mirror sign; add it explicitly")
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

    env._g1_joint_mirror_map = (perm, sign)
    return perm, sign


def _mirror_joint_block(block: torch.Tensor, perm: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
    """Mirror one ``(batch, num_joints)`` block of joint-indexed values."""
    return block[:, perm] * sign


def _selected_joint_map(env: ManagerBasedRLEnv, joint_ids: list[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
    """Restrict the validated robot mirror to a term's joint selection and order."""
    perm, sign = _build_map(env)
    all_ids = tuple(range(perm.numel()))
    selected = all_ids[joint_ids] if isinstance(joint_ids, slice) else tuple(joint_ids)
    if selected == all_ids:
        return perm, sign
    # Selections are fixed when managers resolve their terms; cache per environment.
    cache = getattr(env, "_g1_joint_mirror_subsets", None)
    if cache is None:
        cache = env._g1_joint_mirror_subsets = {}
    if selected not in cache:
        names = env.scene["robot"].joint_names
        selected_names = [names[i] for i in selected]
        counterparts = [_counterpart(name) for name in selected_names]
        if not selected or len(set(selected)) != len(selected) or any(n not in selected_names for n in counterparts):
            raise ValueError("G1 joint selection must be nonempty, unique, and closed under reflection")
        local_perm = torch.tensor([selected_names.index(n) for n in counterparts], device=perm.device)
        cache[selected] = local_perm, sign[list(selected)]
    return cache[selected]


def _action_joint_map(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve action order independently of the articulation's full joint order."""
    cached = getattr(env, "_g1_action_mirror_map", None)
    if cached is None:
        if env.action_manager.active_terms != ["joint_pos"]:
            raise ValueError("G1 symmetry expects one joint_pos action term")
        cfg = env.action_manager.get_term("joint_pos").cfg
        ids, _ = env.scene["robot"].find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        cached = env._g1_action_mirror_map = _selected_joint_map(env, ids)
    return cached


def _observation_joint_map(env: ManagerBasedRLEnv, group: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read the observation manager's resolved selection, which can differ from actions."""
    cfg = env.observation_manager.cfg
    group_cfg = cfg[group] if isinstance(cfg, dict) else getattr(cfg, group)
    term_cfg = group_cfg[name] if isinstance(group_cfg, dict) else getattr(group_cfg, name)
    asset_cfg = term_cfg.params.get("asset_cfg")
    if asset_cfg is None:
        return _build_map(env)
    if asset_cfg.name != "robot":
        raise ValueError("G1 joint observations must refer to robot")
    return _selected_joint_map(env, asset_cfg.joint_ids)


def _mirror_group(env: ManagerBasedRLEnv, group: str, obs: torch.Tensor) -> torch.Tensor:
    """Mirror each observation term in its resolved joint order, including history frames."""
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
            perm, sign = _action_joint_map(env) if name == "actions" else _observation_joint_map(env, group, name)
            if size % perm.numel():
                raise ValueError(f"{group}.{name}: observation size {size} does not match its joint selection")
            frames = size // perm.numel()
            reshaped = block.reshape(block.shape[0] * frames, perm.numel())
            out[:, offset : offset + size] = _mirror_joint_block(reshaped, perm, sign).reshape(block.shape)
        elif name == "height_scan":
            if size != _SCAN_ROWS * _SCAN_COLS:
                raise ValueError(f"height_scan is {size} values, not the {_SCAN_ROWS}x{_SCAN_COLS} grid")
            out[:, offset : offset + size] = block.view(-1, _SCAN_ROWS, _SCAN_COLS).flip(dims=[1]).reshape(block.shape)
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
        perm, sign = _action_joint_map(unwrapped)
        actions_aug = torch.cat([actions, _mirror_joint_block(actions, perm, sign)], dim=0)
    else:
        actions_aug = None

    return obs_aug, actions_aug
