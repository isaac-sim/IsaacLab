# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared goal state and task parameters for the handover workflows."""

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import index_fill_
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.core.reorient.utils import SuccessTracker

__all__ = [
    "GOAL_MARKER_CFG",
    "GOAL_POSITION_OFFSET",
    "HandoverGoal",
    "OBJECT_RADIUS",
]


OBJECT_RADIUS: float = 0.0335
"""Hand-over object sphere radius [m], also used for the goal marker."""

GOAL_POSITION_OFFSET: tuple[float, float, float] = (0.36, 0.0, 0.04)
"""Goal-position offset in each hand's local root frame [m]."""


class HandoverGoal:
    """Track alternating goal positions and cumulative dwell for both workflows."""

    def __init__(self, hand_root_poses: torch.Tensor, position_offset: tuple[float, float, float]):
        """Initialize from right/left hand poses of shape (2, num_envs, 7), in environment frames."""
        num_envs = hand_root_poses.shape[1]
        device = hand_root_poses.device
        offset = torch.tensor(position_offset, device=device).expand(2, num_envs, 3)
        self._positions = hand_root_poses[..., :3] + quat_apply(hand_root_poses[..., 3:7], offset)
        self.position = self._positions[1].clone()
        self.dwell = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = SuccessTracker(num_envs, device)
        self._side = torch.ones(num_envs, dtype=torch.long, device=device)

    def update(self, succeeded: torch.Tensor, dwell_steps: int) -> torch.Tensor:
        """Accumulate earned dwell, switch completed goals, and return their environment indices."""
        self.dwell += succeeded.long()
        env_ids = (self.dwell >= dwell_steps).nonzero(as_tuple=False).flatten()
        self.success.record_goal_reached(env_ids)
        self._side[env_ids] = 1 - self._side[env_ids]
        self.dwell[env_ids] = 0
        self.position[env_ids] = self._positions[self._side[env_ids], env_ids]
        return env_ids

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Start a new episode with the left-hand goal and no earned dwell or transfers."""
        index_fill_(self._side, env_ids, 1)
        index_fill_(self.dwell, env_ids, 0)
        self.position[env_ids] = self._positions[1, env_ids]
        self.success.clear(env_ids, skip_next_update=torch.zeros_like(self.dwell[env_ids], dtype=torch.bool))


GOAL_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.SphereCfg(
            radius=OBJECT_RADIUS,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
        ),
    },
)
"""Goal-marker template shared by the Direct environment and the manager command term.

Consumers relying on a different prim path use ``replace``
on this template; configclass deep-copies defaults, so sharing the instance is safe.
"""
