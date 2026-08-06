# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Domain-agnostic observation terms shared across terrain and factory tasks.

Groups:

- **Vision obs** — ``vision_obs`` converts camera-style sensors and grid
  raycasters to CNN-shaped image tensors.
- **Multi-task command obs** — ``command_progress``, ``command_reach``,
  ``command_track``, ``command_active``. Wrap properties of any
  :class:`~.commands.MultiTaskCommand` so the policy can read its
  current task state. Domain-agnostic because the underlying command term
  is.
- **Frame-relative obs** — ``target_asset_pose_in_root_asset_frame``,
  ``asset_link_velocity_in_root_asset_frame``. Read pose/velocity of one
  scene asset relative to another. Pure rigid-body math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_tasks.contrib.nist.utils import get_reset_state

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .util.pose_offset import Offset


# ---------------------------------------------------------------------------
# Vision observation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Episode-progress observation.
# ---------------------------------------------------------------------------


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Fraction of the episode remaining ∈ ``[0, 1]``, shape ``[num_envs, 1]``.

    Bounded and episode-length-invariant — preferred over absolute-seconds
    forms because the policy's obs distribution stays stable across
    curriculum changes to ``episode_length_s``.
    """
    time_left_frac = 1 - env.episode_length_buf / env.max_episode_length
    return time_left_frac.view(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Multi-task command observation accessors.
# ---------------------------------------------------------------------------


def command_progress(env, command_name: str = "goal_point"):
    """Scalar per-env task progress ∈ [0, 1], shape ``[num_envs, 1]``.

    Mean of the env's active-subtask activations — a task-normalized "how close am I?"
    signal with no reward-kernel parameters baked in.
    """
    return env.command_manager.get_term(command_name).progress.unsqueeze(-1)


def command_reach(env, command_name: str = "goal_point"):
    """Canonical state delta for instant ("reach") subtasks.

    Shape ``[num_envs, reach_canonical_width]``. Populated only by instant subtasks;
    tracking subtasks write to :func:`command_track`. Keeps the two semantic
    categories in separate obs tensors so the policy reads them positionally.
    """
    return env.command_manager.get_term(command_name).command_reach


def command_track(env, command_name: str = "goal_point"):
    """Canonical state delta for tracking subtasks.

    Shape ``[num_envs, track_canonical_width]``. Populated only by tracking
    subtasks. Same positional encoding as :func:`command_reach` but disjoint
    channels, so same-kernel reach + track subtasks coexist without aliasing.
    """
    return env.command_manager.get_term(command_name).command_track


def command_active(env, command_name: str = "goal_point"):
    """Per-channel active mask paired with :func:`command_reach` + :func:`command_track`.

    Shape ``[num_envs, reach_canonical_width + track_canonical_width]``. The
    layout mirrors ``cat([command_reach, command_track], dim=-1)`` slot-for-
    slot: column ``i`` of this mask gates column ``i`` of the concatenated
    delta. ``1.0`` iff the channel is populated by a live subtask of the
    env's current task; ``0.0`` otherwise (inactive channel, or joint-kernel
    subtask with no canonical projection).
    """
    return env.command_manager.get_term(command_name).command_active


# ---------------------------------------------------------------------------
# Frame-relative pose / velocity observations.
# ---------------------------------------------------------------------------


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset: Offset | None = None,
    root_asset_offset: Offset | None = None,
):
    """Pose of ``target_asset`` expressed in the root frame of ``root_asset``.

    Optional ``Offset`` cfgs let callers compose static frame offsets onto
    either side (e.g. observe an end-effector grasp point relative to a
    fixed-asset tip).

    Returns a ``[num_envs, 7]`` tensor — translation (3) + quaternion xyzw (4).
    """
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w.torch[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w.torch[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w.torch[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w.torch[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
    return torch.cat([target_pos_b, target_quat_b], dim=1)


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Linear + angular velocity of ``target_asset``'s link, expressed in the root frame.

    Returns a ``[num_envs, 6]`` tensor — linear (3) + angular (3) in body frame.
    """
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    root_quat = root_asset.data.root_quat_w.torch
    lin_vel_w = target_asset.data.body_lin_vel_w.torch[:, target_body_idx].view(-1, 3)
    ang_vel_w = target_asset.data.body_ang_vel_w.torch[:, target_body_idx].view(-1, 3)

    lin_vel_b = math_utils.quat_apply_inverse(root_quat, lin_vel_w)
    ang_vel_b = math_utils.quat_apply_inverse(root_quat, ang_vel_w)

    return torch.cat([lin_vel_b, ang_vel_b], dim=1)


def get_state(env: ManagerBasedRLEnv, reset_assets: list[str]):
    return get_reset_state(env, slice(None), reset_assets, is_relative=True)
