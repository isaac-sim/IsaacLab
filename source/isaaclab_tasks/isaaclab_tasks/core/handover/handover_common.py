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
from isaaclab.visualizers import VisualizerCfg

from isaaclab_tasks.core.reorient.utils import SuccessTracker

__all__ = [
    "GOAL_MARKER_CFG",
    "GOAL_POSITION_OFFSET",
    "HandoverGoal",
    "OBJECT_RADIUS",
    "VISUALIZER_CFG",
]


OBJECT_RADIUS: float = 0.0335
"""Hand-over object sphere radius [m], also used for the goal marker."""

GOAL_POSITION_OFFSET: tuple[float, float, float] = (0.36, 0.0, 0.04)
"""Goal-position offset in each hand's local root frame [m]."""

# The camera is world-framed, so this aims at the environment nearest the grid center for the
# default 2048 environments (x = 0.75 m); other counts shift the grid.
VISUALIZER_CFG = VisualizerCfg(eye=(1.9, -1.65, 1.15), lookat=(0.75, -0.5, 0.55), focal_length=35.0)
"""Recording view angled down at one hand pair, with neighboring environments behind it."""


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


class HandoverGoal:
    """Track alternating goal positions and cumulative success steps for both workflows."""

    def __init__(self, hand_root_poses: torch.Tensor, position_offset: tuple[float, float, float]):
        """Initialize from right/left hand poses of shape (2, num_envs, 7), in environment frames."""
        num_envs = hand_root_poses.shape[1]
        device = hand_root_poses.device
        offset = torch.tensor(position_offset, device=device).expand(2, num_envs, 3)
        self._positions = hand_root_poses[..., :3] + quat_apply(hand_root_poses[..., 3:7], offset)
        self.position = self._positions[1].clone()
        self.success_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.success = SuccessTracker(num_envs, device)
        self._left = torch.ones(num_envs, dtype=torch.bool, device=device)

    def update(self, succeeded: torch.Tensor, steps_required: int) -> torch.Tensor:
        """Accumulate success steps, switch completed goals, and return the mask of switched environments.

        Raises:
            ValueError: If ``steps_required`` is not positive, which would switch goals every step.
        """
        if steps_required < 1:
            raise ValueError(f"success_steps_required must be positive, got {steps_required}.")
        self.success_steps += succeeded
        switched = self.success_steps >= steps_required
        self.success.record_goal_reached(env_mask=switched)
        self._left ^= switched
        self.success_steps.masked_fill_(switched, 0)
        torch.where(self._left.unsqueeze(-1), self._positions[1], self._positions[0], out=self.position)
        return switched

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice) -> None:
        """Start a new episode with the left-hand goal and no success steps or transfers."""
        index_fill_(self._left, env_ids, True)
        index_fill_(self.success_steps, env_ids, 0)
        self.position[env_ids] = self._positions[1, env_ids]
        self.success.clear(env_ids, skip_next_update=torch.zeros_like(self.success_steps[env_ids], dtype=torch.bool))
