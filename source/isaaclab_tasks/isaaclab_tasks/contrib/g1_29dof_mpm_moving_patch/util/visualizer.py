# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Standard particle rendering with task-local visibility filtering."""

import numpy as np
import warp as wp
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg
from isaaclab_visualizers.newton.newton_visualizer import (
    NewtonGLVisualizer,
    NewtonRTXVisualizer,
    NewtonViewerGL,
    NewtonViewerRTX,
)

from isaaclab.utils import configclass

from .kernel import gather_visible_particles, mark_visible_particles
from .particles import MovingPatchParticles


class MovingPatchParticleRendering:
    """Filter the standard particle batch and optionally follow a robot body."""

    def log_state(self, state):
        """Render particles and track the configured body's world position."""
        super().log_state(state)
        visualizer = self.moving_patch_visualizer
        body_path = visualizer.cfg.follow_body_path
        if body_path is None or self.model is None:
            return

        follow_key = (id(self.model), body_path)
        if getattr(self, "_follow_body_key", None) != follow_key:
            self._follow_body_index = self.model.body_label.index(body_path)
            self._follow_body_key = follow_key

        body_position = state.body_q[self._follow_body_index : self._follow_body_index + 1].numpy()[0, :3]
        eye = tuple(body_position + np.asarray(visualizer.cfg.eye))
        target = tuple(body_position + np.asarray(visualizer.cfg.lookat))
        eye_offset, target_offset = visualizer.cfg.eye, visualizer.cfg.lookat
        try:
            visualizer.set_camera_view(eye, target)
        finally:
            visualizer.cfg.eye, visualizer.cfg.lookat = eye_offset, target_offset

    def _log_particles(self, state):
        patch = getattr(self.moving_patch_visualizer, "moving_patch_particle", None)
        if patch is None:
            super()._log_particles(state)
            return
        model = patch.model
        if getattr(self, "_particle_filter_model", None) is not model:
            self._particle_filter_model = model
            count = model.particle_count
            device = state.particle_q.device
            self._particle_mask = wp.empty(count, dtype=int, device=device)
            self._particle_offsets = wp.empty(count, dtype=int, device=device)
            self._visible_positions = wp.empty(count, dtype=wp.vec3, device=device)
            self._visible_radii = wp.empty(count, dtype=float, device=device)
            self._particle_colors = wp.full(count, wp.vec3(0.7, 0.6, 0.4), dtype=wp.vec3, device=device)
            self._particle_filter_world_mask = wp.zeros(model.world_count, dtype=int, device=device)
            self._particle_filter_world_ids = None
            self._empty_positions = wp.empty(0, dtype=wp.vec3, device=device)

        if not self.show_particles or model.particle_count == 0:
            self.log_points("/model/particles", points=self._empty_positions, hidden=True)
            return

        visible = self.moving_patch_visualizer.get_visualized_env_ids()
        if visible is None and patch.terrain.show_boundary_particles:
            self.log_points(
                "/model/particles",
                points=state.particle_q,
                radii=model.particle_radius,
                colors=self._particle_colors,
            )
            return

        if visible is not None:
            world_ids = tuple(visible)
            if world_ids != self._particle_filter_world_ids:
                selected = set(world_ids)
                self._particle_filter_world_mask.assign([int(i in selected) for i in range(model.world_count)])
                self._particle_filter_world_ids = world_ids

        wp.launch(
            mark_visible_particles,
            dim=model.particle_count,
            inputs=[
                model.particle_world,  # particle_env_ids
                patch.particle_is_dynamic,  # particle_is_dynamic
                self._particle_filter_world_mask,  # env_visibility_mask
                visible is not None,  # filter_envs
                patch.terrain.show_boundary_particles,  # show_boundary_particles
                self._particle_mask,  # particle_visibility_mask
            ],
            device=state.particle_q.device,
        )
        wp.utils.array_scan(self._particle_mask, self._particle_offsets, inclusive=True)
        # Standard log_points needs an array length; only the count returns to the CPU.
        count = int(self._particle_offsets[-1:].numpy()[0])
        if count == 0:
            self.log_points("/model/particles", points=self._empty_positions, hidden=True)
            return
        wp.launch(
            gather_visible_particles,
            dim=model.particle_count,
            inputs=[
                self._particle_mask,  # particle_visibility_mask
                self._particle_offsets,  # particle_visible_offsets
                state.particle_q,  # particle_q
                model.particle_radius,  # particle_radius
                self._visible_positions,  # visible_particle_q
                self._visible_radii,  # visible_particle_radius
            ],
            device=state.particle_q.device,
        )
        self.log_points(
            "/model/particles",
            points=self._visible_positions[:count],
            radii=self._visible_radii[:count],
            colors=self._particle_colors[:count],
        )


class MovingPatchViewerGL(MovingPatchParticleRendering, NewtonViewerGL):
    """Standard GL viewer with moving-terrain particle filtering."""


class MovingPatchViewerRTX(MovingPatchParticleRendering, NewtonViewerRTX):
    """Standard RTX viewer with moving-terrain particle filtering."""


class MovingPatchGLVisualizer(NewtonGLVisualizer):
    """OpenGL rendering for the experimental moving terrain."""

    moving_patch_particle: MovingPatchParticles | None = None

    _viewer_class = MovingPatchViewerGL

    def initialize(self, sim, *, cameras) -> None:
        """Bind the native viewer to the moving particle patch after initialization."""
        super().initialize(sim, cameras=cameras)
        self._viewer.moving_patch_visualizer = self


class MovingPatchRTXVisualizer(NewtonRTXVisualizer):
    """RTX rendering for the experimental moving terrain."""

    moving_patch_particle: MovingPatchParticles | None = None

    _viewer_class = MovingPatchViewerRTX

    def initialize(self, sim, *, cameras) -> None:
        """Bind the native viewer to the moving particle patch after initialization."""
        super().initialize(sim, cameras=cameras)
        self._viewer.moving_patch_visualizer = self


@configclass
class MovingPatchGLVisualizerCfg(NewtonGLVisualizerCfg):
    """Use the task-local GL subclass while keeping IsaacLab's visualizer configuration."""

    class_type: type = MovingPatchGLVisualizer

    follow_body_path: str | None = None
    """Body prim path to follow, or ``None`` for a fixed camera.

    When set, ``eye`` and ``lookat`` are world-axis offsets from the body's position.
    The camera follows translation without rotating with the body.
    """
    show_particles: bool = True


@configclass
class MovingPatchRTXVisualizerCfg(NewtonRTXVisualizerCfg):
    """Use the task-local RTX subclass while keeping IsaacLab's visualizer configuration."""

    class_type: type = MovingPatchRTXVisualizer

    follow_body_path: str | None = None
    """Body prim path to follow, or ``None`` for a fixed camera.

    When set, ``eye`` and ``lookat`` are world-axis offsets from the body's position.
    The camera follows translation without rotating with the body.
    """
    show_particles: bool = True
