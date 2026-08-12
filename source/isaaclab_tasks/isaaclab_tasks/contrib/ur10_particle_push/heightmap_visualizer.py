# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton GL visualization of the particle-push policy heightmap."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
import warp as wp
from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg

if TYPE_CHECKING:
    from .ur10_particle_push_env import UR10ParticlePushEnv


_HEIGHTMAP_PATH = "/task/policy_heightmap"
_HEIGHTMAP_SHAPE = (8, 16)


class _ParticlePushHeightmapVisualizer(NewtonGLVisualizer):
    """Add an opt-in policy-heightmap overlay to Newton GL."""

    def __init__(self, cfg: NewtonGLVisualizerCfg):
        super().__init__(cfg)
        self._source: tuple[weakref.ReferenceType[UR10ParticlePushEnv], int] | None = None
        self._buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self._warp_buffers: tuple[wp.array, wp.array, wp.array] | None = None
        self._radius = 0.0
        self._visible = False
        self._logged = False

    def initialize(self, scene_data_provider) -> None:
        """Initialize the viewer and register the heightmap controls."""
        super().initialize(scene_data_provider)
        if self._viewer is not None:
            self._viewer.register_ui_callback(self._heightmap_panel, position="panel")

    def bind_heightmap_source(self, env: UR10ParticlePushEnv) -> None:
        """Bind the first environment shown by this visualizer."""
        visible_env_ids = self._resolved_visible_env_ids
        if visible_env_ids == []:
            return
        env_id = 0 if visible_env_ids is None else visible_env_ids[0]
        rows, columns = _HEIGHTMAP_SHAPE
        x_lo, x_hi = env.cfg.heightmap_x_bounds
        y_lo, y_hi = env.cfg.heightmap_y_bounds
        x = torch.linspace(
            x_lo + 0.5 * (x_hi - x_lo) / columns,
            x_hi - 0.5 * (x_hi - x_lo) / columns,
            columns,
            device=env.device,
        )
        y = torch.linspace(
            y_lo + 0.5 * (y_hi - y_lo) / rows,
            y_hi - 0.5 * (y_hi - y_lo) / rows,
            rows,
            device=env.device,
        )
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack((grid_x, grid_y, torch.zeros_like(grid_x)), dim=-1).reshape(-1, 3)
        positions.add_(env.scene.env_origins[env_id])
        radii = torch.zeros(rows * columns, device=env.device)
        colors = torch.zeros((rows * columns, 3), device=env.device)

        self._source = (weakref.ref(env), env_id)
        self._buffers = (positions, radii, colors)
        self._warp_buffers = (
            wp.from_torch(positions, dtype=wp.vec3f),
            wp.from_torch(radii, dtype=wp.float32),
            wp.from_torch(colors, dtype=wp.vec3f),
        )
        self._radius = 0.2 * min((x_hi - x_lo) / columns, (y_hi - y_lo) / rows)

    def _pre_step(self) -> None:
        """Update the overlay from the cached policy observation."""
        super()._pre_step()
        if not self._visible:
            if self._logged:
                self._viewer.log_points(_HEIGHTMAP_PATH, None)
                self._logged = False
            return
        if self._source is None or self._buffers is None or self._warp_buffers is None:
            return
        env = self._source[0]()
        heightmap = None if env is None else env.obs_buf.get("heightmap")
        if not isinstance(heightmap, torch.Tensor):
            return

        env_id = self._source[1]
        values = F.adaptive_max_pool2d(heightmap[env_id : env_id + 1, :1], _HEIGHTMAP_SHAPE).flatten()
        values.nan_to_num_(nan=0.0, posinf=1.0, neginf=0.0).clamp_(0.0, 1.0)
        positions, radii, colors = self._buffers
        radii.copy_((values > 0.0) * self._radius)
        positions[:, 2].copy_(
            env.scene.env_origins[env_id, 2]
            + env.cfg.heightmap_z_min
            + values * env.cfg.heightmap_z_range
            + self._radius
        )
        colors[:, 0].copy_(values)
        colors[:, 1].copy_(0.25 + 0.75 * values)
        colors[:, 2].copy_(1.0 - values)
        self._viewer.log_points(_HEIGHTMAP_PATH, *self._warp_buffers)
        self._logged = True

    def _heightmap_panel(self, imgui) -> None:
        """Draw the task-local heightmap checkbox."""
        imgui.set_next_item_open(False, imgui.Cond_.appearing)
        if imgui.collapsing_header("Task Visualization"):
            _, self._visible = imgui.checkbox("Show policy heightmap", self._visible)
            imgui.text("8 x 16 max-pooled spheres")
