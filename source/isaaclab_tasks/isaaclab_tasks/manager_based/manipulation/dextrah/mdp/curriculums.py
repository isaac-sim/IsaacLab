# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import re
import torch
from typing import Any
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, sample_uniform
from isaaclab.envs import mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, iv, fv, difficulty_term_str):
    """
    Interpolate between initial value iv and final value fv, for any arbitrarily
    nested structure of lists/tuples in 'data'. Scalars (int/float) are handled
    at the leaves.
    """
    # get the fraction scalar on the device
    difficulty_term: DifficultyScheduler = env.curriculum_manager.get_term_cfg(difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 1.0:
        return mdp.modify_env_param.NO_CHANGE

    # convert iv/fv to tensors, but we'll peel them apart in recursion
    iv_t = torch.tensor(iv, device=env.device)
    fv_t = torch.tensor(fv, device=env.device)

    return recurse(iv_t.tolist(), fv_t.tolist(), data, frac)

def recurse(iv_elem, fv_elem, data_elem, frac):
    # If it's a sequence, rebuild the same type with each element recursed
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        # Note: we assume iv_elem and fv_elem have the same structure as data_elem
        return type(data_elem)(
            recurse(iv_e, fv_e, d_e, frac)
            for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem)
        )
    # Otherwise it's a leaf scalar: do the interpolation
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val.item())
    else:
        # cast floats or any numeric
        return new_val.item()


def resample_bucket_range(
    env: ManagerBasedRLEnv,
    env_id,
    data,
    static_fric_range: tuple[tuple[float, float], tuple[float, float]],
    dynamic_fric_range: tuple[tuple[float, float], tuple[float, float]],
    restitution_range: tuple[tuple[float, float], tuple[float, float]],
    difficulty_term_str: str
):
    # cpu only
    iv_s, fv_s = torch.tensor(static_fric_range[0]), torch.tensor(static_fric_range[1])
    iv_d, fv_d = torch.tensor(dynamic_fric_range[0]), torch.tensor(dynamic_fric_range[1])
    iv_r, fv_r = torch.tensor(restitution_range[0]), torch.tensor(restitution_range[1])
    difficulty_term: DifficultyScheduler = env.curriculum_manager.get_term_cfg(difficulty_term_str).func
    difficulty_frac = difficulty_term.difficulty_frac.item()
    new_static_fric_range = difficulty_frac * (fv_s - iv_s) + iv_s
    new_dynamic_fric_range = difficulty_frac * (fv_d - iv_d) + iv_d
    new_restitution_range = difficulty_frac * (fv_r - iv_r) + iv_r
    ranges = torch.stack([new_static_fric_range, new_dynamic_fric_range, new_restitution_range], dim=0)
    new_buckets = sample_uniform(ranges[:, 0], ranges[:, 1], (len(data), 3), device="cpu")
    return new_buckets


def value_override(env: ManagerBasedRLEnv, env_id, data, new_val, num_steps):
    if env.common_step_counter > num_steps:
        return new_val

class DifficultyScheduler(ManagerTermBase):
    
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
        promotion_only: bool = False
    ):
        asset: Articulation = env.scene[asset_cfg.name]
        object: RigidObject = env.scene[object_cfg.name]
        command = env.command_manager.get_command("object_pose")
        des_pos_w, des_quat_w = combine_frame_transforms(
            asset.data.root_pos_w[env_ids], asset.data.root_quat_w[env_ids], command[env_ids, :3], command[env_ids, 3:7]
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w, des_quat_w, object.data.root_pos_w[env_ids], object.data.root_quat_w[env_ids]
        )
        pos_dist = torch.norm(pos_err, dim=1)
        rot_dist = torch.norm(rot_err, dim=1)
        move_up = (pos_dist < pos_tol) & (rot_dist < rot_tol) if rot_tol else pos_dist < pos_tol
        demot = self.current_adr_difficulties[env_ids] if promotion_only else self.current_adr_difficulties[env_ids] - 1
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up, self.current_adr_difficulties[env_ids] + 1, demot,
        ).clamp(min=min_difficulty, max=max_difficulty)
        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac

def cfg_get(root: Any, path: str) -> Any:
    """
    Retrieve a deeply nested attribute/key/index from `root` using a string path.
    Examples:
      get_by_path(obj, "a.b.c")
      get_by_path(obj, "a.list_field[2].x")
    """
    # Matches “name[3]” → group(1)=name, group(2)=3
    _INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")
    current = root
    for part in path.split("."):
        m = _INDEX_RE.match(part)
        if m:
            name, idx = m.group(1), int(m.group(2))
            # first attribute/key, then index
            current = getattr(current, name) if not isinstance(current, dict) else current[name]
            current = current[idx]
        else:
            # plain attr or dict lookup
            current = current[part] if isinstance(current, dict) else getattr(current, part)
    return current

def cfg_set(root: Any, path: str, value: Any) -> None:
    """
    Assign `value` to the leaf specified by `path` on `root`.
    Examples:
      set_by_path(obj, "a.b.c", 123)
      set_by_path(obj, "a.list_field[2].x", "foo")
    """
    # Matches “name[3]” → group(1)=name, group(2)=3
    _INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")
    parts = path.split(".")
    target = root
    # walk to the parent of the leaf
    for part in parts[:-1]:
        m = _INDEX_RE.match(part)
        if m:
            name, idx = m.group(1), int(m.group(2))
            # attr/dict lookup then index
            node = getattr(target, name) if not isinstance(target, dict) else target[name]
            target = node[idx]
        else:
            target = target[part] if isinstance(target, dict) else getattr(target, part)

    # now 'target' is the container whose child we want to set:
    last = parts[-1]
    m = _INDEX_RE.match(last)
    if m:
        name, idx = m.group(1), int(m.group(2))
        container = getattr(target, name) if not isinstance(target, dict) else target[name]
        container[idx] = value
    else:
        if isinstance(target, dict):
            target[last] = value
        else:
            setattr(target, last, value)
