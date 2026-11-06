# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Publish a Newton particle surface through the Kit, GL and RTX visualizers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import warp as wp

from .newton_visualizer import NewtonGLVisualizer, NewtonRTXVisualizer

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager
    from newton.geometry import ParticleSurface

    from isaaclab.visualizers.base_visualizer import BaseVisualizer


class ParticleSurfaceRenderer:
    """Extract one particle surface and stage its latest mesh for active viewers.

    Configure a Newton ``ParticleSurface`` and construct this renderer after
    resetting the simulation. Call :meth:`update` before rendering each frame
    to reconstruct the surface from the current particle state.
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
        backend_manager: NewtonManager | None = None,
    ) -> None:
        """Allocate reusable extraction storage and optionally capture its CUDA graph.

        Args:
            visualizers: Active visualizers; Kit and Newton viewers are supported.
            surface: Configured Newton surface extractor.
            path: Viewer mesh path.
            color: Surface RGB color in [0, 1].
            opacity: Surface opacity in [0, 1].
            roughness: Surface roughness in [0, 1].
            use_cuda_graph: Capture extraction on CUDA when enabled.
            backend_manager: Newton manager whose particle state is reconstructed. Defaults to the active
                simulation's manager (deprecated; pass it explicitly).

        Raises:
            RuntimeError: If no Kit, Newton GL or RTX visualizer is active.
        """
        from isaaclab_visualizers.kit import KitVisualizer

        visualizers = tuple(visualizers)
        self._visualizers = tuple(
            visualizer
            for visualizer in visualizers
            if isinstance(visualizer, (NewtonGLVisualizer, NewtonRTXVisualizer))
        )
        self._kit_mesh = None
        if any(isinstance(visualizer, KitVisualizer) for visualizer in visualizers):
            from pxr import UsdGeom

            from isaaclab.sim import PreviewSurfaceCfg, bind_visual_material, get_current_stage, spawn_preview_surface

            self._kit_mesh = UsdGeom.Mesh.Define(get_current_stage(), path)
            self._kit_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            self._kit_mesh.CreateDoubleSidedAttr(True)
            self._kit_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            material_path = f"{path}/material"
            spawn_preview_surface(
                material_path,
                PreviewSurfaceCfg(diffuse_color=color, opacity=opacity, roughness=roughness),
            )
            bind_visual_material(path, material_path)
        if not self._visualizers and self._kit_mesh is None:
            raise RuntimeError("Particle surface rendering requires a Kit, Newton GL or RTX visualizer.")

        from isaaclab_newton.physics import NewtonManager

        if backend_manager is None:
            backend_manager = NewtonManager._deprecated_active("ParticleSurfaceRenderer()")
        self._model = backend_manager.backend.model
        self._state = backend_manager.backend.state_0
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
        hidden = vertices is None or vertices.shape[0] == 0
        triangle_count = 0 if hidden else indices.shape[0] // 3
        if hidden:
            vertices = self._empty_points
            indices = self._empty_indices
            normals = self._empty_normals

        if self._kit_mesh is not None:
            from pxr import UsdGeom, Vt

            points = Vt.Vec3fArray.FromNumpy(vertices.numpy())
            self._kit_mesh.GetPointsAttr().Set(points)
            self._kit_mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices.numpy()))
            self._kit_mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(triangle_count, 3))
            self._kit_mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.numpy()))
            if not hidden:
                self._kit_mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
            self._kit_mesh.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible if hidden else UsdGeom.Tokens.inherited)

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
