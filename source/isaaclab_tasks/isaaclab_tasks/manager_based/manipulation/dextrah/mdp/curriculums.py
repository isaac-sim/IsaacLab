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
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, iv, fv, difficulty_term_str):
    iv_, fv_ = torch.tensor(iv, device=env.device), torch.tensor(fv, device=env.device)
    difficulty_term: DifficultyScheduler = env.curriculum_manager.get_term_cfg(difficulty_term_str).func
    new_val = difficulty_term.difficulty_frac * (fv_ - iv_) + iv_
    if isinstance(data, float):
        return new_val.item()
    elif isinstance(data, int):
        return int(new_val.item())
    elif isinstance(data, (tuple, list)):
        raw = new_val.tolist()
        # assume iv is sequence of all ints or all floats:
        is_int = isinstance(iv[0], int)
        casted = [int(x) if is_int else float(x) for x in raw]
        return tuple(casted) if isinstance(data, tuple) else casted
    else:
        raise TypeError(f"Does not support the type {type(data)}")


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
