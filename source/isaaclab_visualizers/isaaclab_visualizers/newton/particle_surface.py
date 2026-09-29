# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish a Newton particle surface through the GL and RTX visualizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import warp as wp
from isaaclab_newton.physics import NewtonManager

from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

if TYPE_CHECKING:
    from newton.geometry import ParticleSurface

    from isaaclab.visualizers.base_visualizer import BaseVisualizer


class ParticleSurfaceRenderer:
    """Extract one particle surface and stage its latest mesh for Newton viewers.

    Construct this after the Newton simulation has reset. Surface configuration
    remains with the caller; this class owns only extraction and viewer updates.
    """

    def __init__(
        self,
        visualizers: Iterable[BaseVisualizer],
        surface: ParticleSurface,
        *,
        path: str,
        color: tuple[float, float, float],
        opacity: float,
        roughness: float = 0.1,
        use_cuda_graph: bool = True,
    ) -> None:
        """Allocate reusable extraction storage and optionally capture its CUDA graph.

        Args:
            visualizers: Active visualizers; non-Newton viewers are ignored.
            surface: Configured Newton surface extractor.
            path: Viewer mesh path.
            color: Surface RGB color in [0, 1].
            opacity: Surface opacity in [0, 1].
            roughness: Surface roughness in [0, 1].
            use_cuda_graph: Capture extraction on CUDA when enabled.

        Raises:
            RuntimeError: If no Newton GL or RTX visualizer is active.
        """
        self._visualizers = tuple(
            visualizer
            for visualizer in visualizers
            if isinstance(visualizer, (NewtonGLVisualizer, NewtonRTXVisualizer))
        )
        if not self._visualizers:
            raise RuntimeError("Particle surface rendering requires a Newton GL or RTX visualizer.")

        self._model = NewtonManager.get_model()
        self._state = NewtonManager.get_state_0()
        self._surface = surface
        self._path = path
        self._color = color
        self._opacity = opacity
        self._roughness = roughness
        self._empty_points = wp.empty(0, dtype=wp.vec3, device=self._model.device)
        self._empty_indices = wp.empty(0, dtype=wp.int32, device=self._model.device)
        self._empty_normals = wp.empty(0, dtype=wp.vec3, device=self._model.device)
        self._surface_mesh = None
        self._surface_graph = None
        if use_cuda_graph and self._model.device.is_cuda:
            self._surface_mesh = self._extract_surface()
            with wp.ScopedCapture(device=self._model.device) as capture:
                self._surface_mesh = self._extract_surface()
            self._surface_graph = capture.graph

    def update(self) -> int:
        """Stage the latest surface mesh and return its triangle count."""
        if self._surface_graph is None:
            self._surface_mesh = self._extract_surface()
        else:
            wp.capture_launch(self._surface_graph)

        vertices, indices, normals = self._surface_mesh.to_arrays()
        hidden = vertices is None
        triangle_count = 0 if hidden else indices.shape[0] // 3
        if hidden:
            vertices = self._empty_points
            indices = self._empty_indices
            normals = self._empty_normals

        for visualizer in self._visualizers:
            visualizer.log_mesh(
                self._path,
                vertices,
                indices,
                normals=normals,
                hidden=hidden,
                backface_culling=False,
                color=self._color,
                roughness=self._roughness,
                metallic=0.0,
                dynamic=True,
                opacity=self._opacity,
            )
        return triangle_count

    def _extract_surface(self):
        """Extract the surface from the current Newton particle state."""
        return self._surface.extract(
            self._state.particle_q,
            self._model.particle_radius,
            particle_flags=self._model.particle_flags,
            particle_world=self._model.particle_world if self._surface.world_count > 1 else None,
        )
