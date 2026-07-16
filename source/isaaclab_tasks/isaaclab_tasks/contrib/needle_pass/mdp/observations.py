# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stable, recorder-friendly observations for dVRK needle pass."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .terminations import (
    HandoffPhase,
    HandoffPhaseCfg,
    jaw_needle_contact_measurements,
    update_handoff_phase,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _articulation(env: ManagerBasedRLEnv, cfg: SceneEntityCfg) -> Articulation:
    return env.scene[cfg.name]


def joint_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return all live articulation joint positions in native USD order."""

    return _articulation(env, asset_cfg).data.joint_pos.torch


def joint_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return all live articulation joint velocities in native USD order."""

    return _articulation(env, asset_cfg).data.joint_vel.torch


def end_effector_pose_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    body_name: str = "psm_tool_tip_link",
) -> torch.Tensor:
    """Return live tool-tip pose ``[xyz, qx, qy, qz, qw]`` in world frame."""

    asset = _articulation(env, asset_cfg)
    body_ids, body_names = asset.find_bodies(body_name)
    if len(body_ids) != 1:
        raise RuntimeError(f"expected one {body_name!r} body, found {body_names}")
    body_id = body_ids[0]
    return torch.cat((asset.data.body_pos_w.torch[:, body_id], asset.data.body_quat_w.torch[:, body_id]), dim=-1)


def needle_pose_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("needle"),
) -> torch.Tensor:
    """Return simulated needle pose ``[xyz, qx, qy, qz, qw]`` in world frame."""

    needle: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((needle.data.root_pos_w.torch, needle.data.root_quat_w.torch), dim=-1)


def needle_velocity_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("needle"),
) -> torch.Tensor:
    """Return simulated needle linear then angular world velocity."""

    needle: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((needle.data.root_lin_vel_w.torch, needle.data.root_ang_vel_w.torch), dim=-1)


def jaw_needle_contact_force(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return four projected normal loads in left-jaw-1 through right-jaw-2 order."""

    loads, _, _ = jaw_needle_contact_measurements(env)
    return loads


def handoff_phase(env: ManagerBasedRLEnv, phase_cfg: HandoffPhaseCfg) -> torch.Tensor:
    """Return one physical phase column; INITIAL is reset-held pending fresh contact."""

    return update_handoff_phase(env, phase_cfg).phase.unsqueeze(-1)


def phase_at_least(
    env: ManagerBasedRLEnv,
    phase_cfg: HandoffPhaseCfg,
    phase: HandoffPhase,
) -> torch.Tensor:
    """Return a recorder subtask flag derived solely from measured phase state."""

    current = update_handoff_phase(env, phase_cfg).phase
    return (current >= int(phase)).to(dtype=torch.float32).unsqueeze(-1)


__all__ = [
    "end_effector_pose_w",
    "handoff_phase",
    "jaw_needle_contact_force",
    "joint_position",
    "joint_velocity",
    "needle_pose_w",
    "needle_velocity_w",
    "phase_at_least",
]
