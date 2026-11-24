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
    """Require a released object that rests in or on its selected target.

    The center of the object's collision bounds must lie inside the target's bounding rectangle, and the target
    has to carry the object: the object touches the target and not the table, the gripper has let go, the object
    is at rest and the target stands upright. Containment also requires the object's bottom below the rim and its
    center above the container's bottom. Corners may overhang, and a rim may rise above the resting surface. An
    object that passes over its target, bounces in it or leans on it from the table does not count.
    """

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._tabletop = env.event_manager.get_term_cfg("place_objects").func
        self._sensor = env.scene["contact"]
        bodies = [*self._tabletop.names, self._tabletop.gripper_body]
        self._sensor_rows = torch.tensor([self._sensor.sensor_names.index(name) for name in bodies], device=env.device)
        self._sensor_columns = torch.tensor(
            [self._sensor.filter_object_names.index(name) for name in bodies], device=env.device
        )
        table = self._tabletop.table.cfg.prim_path.rsplit("/", 1)[-1]
        self._table_column = self._sensor.filter_object_names.index(table)
        self._envs = torch.arange(env.num_envs, device=env.device)

    def __call__(self, env: ManagerBasedRLEnv, rest_speed: float = 0.05) -> torch.Tensor:
        """Whether each environment's object rests in or on its target.

        Args:
            env: The environment.
            rest_speed: Speed below which the object counts as at rest [m/s].
        """
        state = self._tabletop
        poses = state.objects.data.body_link_pose_w.torch
        source, target = state.object_ids, state.target_ids
        source_pose, target_pose = poses[self._envs, source], poses[self._envs, target]
        corners = quat_apply(source_pose[:, None, 3:].expand(-1, 8, 4), state.local_corners[self._envs, source])
        corners += source_pose[:, None, :3]
        relative = quat_apply_inverse(target_pose[:, None, 3:].expand(-1, 8, 4), corners - target_pose[:, None, :3])
        bounds = state.local_bounds[self._envs, target]
        # The mean of the corners is the center of the object's bounds.
        center = relative.mean(dim=1)
        over = ((center[:, :2] >= bounds[:, 0, :2]) & (center[:, :2] <= bounds[:, 1, :2])).all(dim=1)
        # A corner of rotated bounds lies below a tilted object; the center of the bounds does not.
        inside = (relative[..., 2].amin(1) < bounds[:, 1, 2] - 0.002) & (center[:, 2] >= bounds[:, 0, 2])
        forces = self._sensor.data.normal_force_matrix_w.torch[self._envs, self._sensor_rows[source]]
        touching = forces[self._envs, self._sensor_columns[target]].norm(dim=-1) >= 0.1
        grounded = forces[:, self._table_column].norm(dim=-1) >= 0.1
        released = forces[:, self._sensor_columns[-1]].norm(dim=-1) < 0.1
        speed = state.objects.data.body_com_vel_w.torch[self._envs, source, :3].norm(dim=-1)
        up = quat_apply(target_pose[:, 3:], target_pose.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, 3))
        return (
            (state.task_ids >= 0)
            & poses.isfinite().all(dim=(1, 2))
            & over
            & touching
            & ~grounded
            & released
            & (speed < rest_speed)
            & (up[:, 2] > 0.98)
            & (inside | ~state.in_container)
        )
