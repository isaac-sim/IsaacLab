# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success of the selected category- and size-compatible tabletop task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class task_success(ManagerTermBase):
    """Require a released object, contained footprint and supporting contact with its selected target."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._tabletop = env.event_manager.get_term_cfg("place_objects").func
        self._sensor = env.scene["contact"]
        bodies = [*self._tabletop.names, self._tabletop.gripper_body]
        self._sensor_rows = torch.tensor([self._sensor.sensor_names.index(name) for name in bodies], device=env.device)
        self._sensor_columns = torch.tensor(
            [self._sensor.filter_object_names.index(name) for name in bodies], device=env.device
        )
        self._envs = torch.arange(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        state = self._tabletop
        poses = torch.stack([asset.data.root_link_pose_w.torch for asset in state.assets], dim=1)
        source, target = state.object_ids, state.target_ids
        source_pose, target_pose = poses[self._envs, source], poses[self._envs, target]
        corners = quat_apply(source_pose[:, None, 3:].expand(-1, 8, 4), state.local_corners[self._envs, source])
        corners += source_pose[:, None, :3]
        relative = quat_apply_inverse(target_pose[:, None, 3:].expand(-1, 8, 4), corners - target_pose[:, None, :3])
        bounds = state.local_bounds[self._envs, target]
        fits = (relative[..., :2] >= bounds[:, None, 0, :2] + 0.002).all(dim=(1, 2))
        fits &= (relative[..., :2] <= bounds[:, None, 1, :2] - 0.002).all(dim=(1, 2))
        bottom = relative[..., 2].amin(1)
        height = bounds[:, 1, 2]
        supported = (bottom >= height - 0.01) & (bottom <= height + 0.01)
        inside = (bottom < height - 0.002) & (bottom >= bounds[:, 0, 2] - 0.002)
        forces = self._sensor.data.normal_force_matrix_w.torch
        touching = forces[self._envs, self._sensor_rows[source], self._sensor_columns[target]].norm(dim=-1) >= 0.1
        released = forces[self._envs, self._sensor_rows[source], self._sensor_columns[-1]].norm(dim=-1) < 0.1
        up = quat_apply(target_pose[:, 3:], target_pose.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, 3))
        return (
            (state.task_ids >= 0)
            & poses.isfinite().all(dim=(1, 2))
            & fits
            & touching
            & released
            & (up[:, 2] > 0.98)
            & torch.where(state.in_container, inside, supported)
        )
