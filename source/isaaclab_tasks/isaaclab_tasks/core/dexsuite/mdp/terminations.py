# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the dexsuite task.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


class out_of_bound(ManagerTermBase):
    """Termination condition for when the object falls out of bound.

    The world-space bounds are cached and rebuilt per axis only when the corresponding
    ``in_bound_range`` entry changes. This keeps the hot path free of host-to-device
    transfers while still honoring runtime updates (e.g. from a curriculum term).
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self._object: RigidObject = env.scene[asset_cfg.name]

        # Pre-apply env_origins so we can compare directly against world-space positions.
        self._origins = env.scene.env_origins  # (N, 3)
        self._lower = self._origins.clone()  # (N, 3)
        self._upper = self._origins.clone()  # (N, 3)
        self._cached_axis: list[tuple[float, ...] | None] = [None, None, None]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        in_bound_range: dict[str, tuple[float, float]] = {},
    ) -> torch.Tensor:
        # rebuild only the axes whose bounds changed (curriculum typically only moves one)
        for i, key in enumerate(["x", "y", "z"]):
            bounds = tuple(in_bound_range.get(key, (0.0, 0.0)))
            if bounds != self._cached_axis[i]:
                lo, hi = bounds
                self._lower[:, i] = self._origins[:, i] + lo
                self._upper[:, i] = self._origins[:, i] + hi
                self._cached_axis[i] = bounds

        pos_w = self._object.data.root_pos_w.torch
        return ((pos_w < self._lower) | (pos_w > self._upper)).any(dim=1)


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminating environment when violation of velocity limits detects, this usually indicates unstable physics caused
    by very bad, or aggressive action"""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel.torch
    joint_vel_limits = robot.data.joint_vel_limits.torch
    return (joint_vel.abs() > (joint_vel_limits * 2)).any(dim=1)
