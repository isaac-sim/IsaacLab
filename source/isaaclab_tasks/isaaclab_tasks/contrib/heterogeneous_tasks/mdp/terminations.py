# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Success of the selected category- and size-compatible tabletop task."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import TerminationTermCfg


class task_success(ManagerTermBase):
    """Require a released object that has rested in or on its selected target for a while.

    The object is placed while the center of its collision bounds lies inside the target's footprint, it touches
    the target, the gripper has let go and the target stands upright; containment also requires the object's bottom
    below the rim. The target has to carry the object, not enclose it: corners may overhang, and a rim may rise
    above the resting surface. Success is reported once the object has stayed placed for ``hold_time``, so an object
    that passes over its target, bounces out of it or tips it over does not count.
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
        self._envs = torch.arange(env.num_envs, device=env.device)
        self._placed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._placed_steps[slice(None) if env_ids is None else env_ids] = 0

    def __call__(self, env: ManagerBasedRLEnv, hold_time: float = 0.5) -> torch.Tensor:
        """Whether each environment's object has stayed placed for the hold time.

        Args:
            env: The environment.
            hold_time: Time for which the object has to stay placed [s].
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
        center = relative[..., :2].mean(dim=1)
        over = ((center >= bounds[:, 0, :2]) & (center <= bounds[:, 1, :2])).all(dim=1)
        bottom = relative[..., 2].amin(1)
        inside = (bottom < bounds[:, 1, 2] - 0.002) & (bottom >= bounds[:, 0, 2] - 0.002)
        forces = self._sensor.data.normal_force_matrix_w.torch
        touching = forces[self._envs, self._sensor_rows[source], self._sensor_columns[target]].norm(dim=-1) >= 0.1
        released = forces[self._envs, self._sensor_rows[source], self._sensor_columns[-1]].norm(dim=-1) < 0.1
        up = quat_apply(target_pose[:, 3:], target_pose.new_tensor([0.0, 0.0, 1.0]).expand(env.num_envs, 3))
        placed = (
            (state.task_ids >= 0)
            & poses.isfinite().all(dim=(1, 2))
            & over
            & touching
            & released
            & (up[:, 2] > 0.98)
            & (inside | ~state.in_container)
        )
        self._placed_steps = torch.where(placed, self._placed_steps + 1, 0)
        return self._placed_steps >= max(1, math.ceil(hold_time / env.step_dt - 1e-6))
