# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import warp as wp

from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        # no-op during start, since the difficulty fraction near 0 is wasting of resource.
        return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume initial value element and final value element have the same structure as data
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val.item()


class DifficultyScheduler(ManagerTermBase):
    """Adaptive difficulty scheduler for curriculum learning.

    Tracks per-environment difficulty levels and adjusts them based on task performance. Difficulty increases when
    position/orientation errors fall below given tolerances, and decreases otherwise (unless `promotion_only` is set).
    The normalized average difficulty across environments is exposed as `difficulty_frac` for use in curriculum
    interpolation.

    Args:
        cfg: Configuration object specifying scheduler parameters.
        env: The manager-based RL environment.

    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0

    def get_state(self):
        return self.current_adr_difficulties

    def set_state(self, state: torch.Tensor):
        self.current_adr_difficulties = state.clone().to(self._env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        pos_tol: float = 0.1,
        rot_tol: float | None = None,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        promotion_only: bool = False,
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        object = env.scene[object_cfg.name]
        command = env.command_manager.get_command("object_pose")
        # Convert warp arrays to torch tensors
        asset_root_pos_w = wp.to_torch(asset.data.root_pos_w)
        asset_root_quat_w = wp.to_torch(asset.data.root_quat_w)
        object_root_pos_w = wp.to_torch(object.data.root_pos_w)
        object_root_quat_w = wp.to_torch(object.data.root_quat_w)
        des_pos_w, des_quat_w = combine_frame_transforms(
            asset_root_pos_w[env_ids], asset_root_quat_w[env_ids], command[env_ids, :3], command[env_ids, 3:7]
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w, des_quat_w, object_root_pos_w[env_ids], object_root_quat_w[env_ids]
        )
        pos_dist = torch.norm(pos_err, dim=1)
        rot_dist = torch.norm(rot_err, dim=1)
        move_up = (pos_dist < pos_tol) & (rot_dist < rot_tol) if rot_tol else pos_dist < pos_tol
        demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up,
            self.current_adr_difficulties[env_ids] + 1,
            demot,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac
