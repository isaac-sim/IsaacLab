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
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Matches “name[3]” → group(1)=name, group(2)=3
_INDEX_RE = re.compile(r"^(\w+)\[(\d+)\]$")

@configclass
class ADRTermCfg:
    init_v: float = MISSING
    final_v: float = MISSING


class ADRTerm:
    def __init__(self, env: ManagerBasedRLEnv, address:str, cfg: ADRTermCfg):
        self.env = env
        self.address = address
        self.cfg = cfg

    def update(self, factor: float) -> None:
        val = factor * (self.cfg.final_v - self.cfg.init_v) + self.cfg.init_v
        set(self.env, self.address, val)


class DifficultyScheduler(ManagerTermBase):
    
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params.get("asset_cfg").name]
        self.object: RigidObject = env.scene[cfg.params.get("object_cfg").name]
        adr_terms_cfg: dict[str, ADRTermCfg] = cfg.params.get('adr_terms')
        self.adr_terms: list[ADRTerm] = [ADRTerm(env, name, term_cfg) for name, term_cfg in adr_terms_cfg.items()]
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * self.cfg.params.get("init_difficulty", 0)
    
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
        dist_tol: float = 0.1,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 50,
        adr_terms: dict[str, ADRTermCfg] = {}
    ):
        command = env.command_manager.get_command("object_pose")
        des_pos_b = command[env_ids, :3]
        des_pos_w, _ = combine_frame_transforms(
            self.asset.data.root_state_w[env_ids, :3], self.asset.data.root_state_w[env_ids, 3:7], des_pos_b
        )

        distance = torch.norm(des_pos_w - self.object.data.root_pos_w[env_ids, :3], dim=1)
        move_up = distance < dist_tol
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up, self.current_adr_difficulties[env_ids] + 1, self.current_adr_difficulties[env_ids] - 1,
        ).clamp(min=min_difficulty, max=max_difficulty)
        for term in self.adr_terms:
            term.update((torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)).item())

        return torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)

def get(root: Any, path: str) -> Any:
    """
    Retrieve a deeply nested attribute/key/index from `root` using a string path.
    Examples:
      get_by_path(obj, "a.b.c")
      get_by_path(obj, "a.list_field[2].x")
    """
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

def set(root: Any, path: str, value: Any) -> None:
    """
    Assign `value` to the leaf specified by `path` on `root`.
    Examples:
      set_by_path(obj, "a.b.c", 123)
      set_by_path(obj, "a.list_field[2].x", "foo")
    """
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
