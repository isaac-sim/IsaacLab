# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private runtime-only coupled solvers for the Rizon–Sharpa teapot example."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import warp as wp
from isaaclab_contrib.coupling import CouplerSolver
from newton import GeoType
from newton.solvers.experimental.coupled import SolverCoupledProxy
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from newton import ModelBuilder


class _TeapotProxySolver(SolverCoupledProxy):
    """Use Newton's proxy algorithm with convex rigid contacts and the exact MPM shell.

    Proxy bodies normally expose every attached shape in the destination. The
    hollow pot needs a separate shell for MPM so its convex grasp colliders do not
    fill the water cavity. Only geometry visibility differs from the standard solver.
    """

    def _apply_entry_shape_visibility(self, view, cfg, proxy_body_keep):
        visible = self._entry_visible_shapes(cfg, set(cfg.bodies) | proxy_body_keep)
        if cfg.name == "robot":
            # MJWarp CCD requires zero margins; the fluid shell retains its margins.
            view.shape_margin = wp.zeros_like(view.shape_margin)
        super()._apply_entry_shape_visibility(view, replace(cfg, shapes=sorted(visible)), proxy_body_keep)

    def _entry_visible_shapes(self, cfg, visible_bodies):
        visible = super()._entry_visible_shapes(cfg, visible_bodies)
        if cfg.name == "robot":
            visible = {i for i in visible if "/PourContainer/geometry/" not in self.model.shape_label[i]}
            visible.update(
                i
                for i, label in enumerate(self.model.shape_label)
                if "/TabletopCollider/" in label or label.startswith("/World/Ground/")
            )
        else:
            visible.update(i for i, label in enumerate(self.model.shape_label) if "/PourContainer/geometry/" in label)
            visible = {
                i
                for i in visible
                if "/PourContainer/" not in self.model.shape_label[i]
                or "/PourContainer/geometry/" in self.model.shape_label[i]
            }
        return visible


def repair_teapot_hulls(builder: ModelBuilder) -> None:
    """Keep generated rigid hulls above the exact shell's body-local base plane."""
    for body, label in enumerate(builder.body_label):
        if not label.endswith("/PourContainer"):
            continue
        shell = next(i for i, name in enumerate(builder.shape_label) if name.startswith(label + "/geometry/"))
        transform = builder.shape_transform[shell]
        points = Rotation.from_quat(transform.q).apply(
            builder.shape_source[shell].vertices * np.asarray(builder.shape_scale[shell])
        ) + np.asarray(transform.p)
        base_z = float(points[:, 2].min())
        for shape, name in enumerate(builder.shape_label):
            if (
                builder.shape_body[shape] != body
                or not name.startswith(label + "/RigidCollider/")
                or builder.shape_type[shape] != GeoType.CONVEX_MESH
            ):
                continue
            mesh = builder.shape_source[shape]
            transform = builder.shape_transform[shape]
            rotation = Rotation.from_quat(transform.q)
            scale = np.asarray(builder.shape_scale[shape])
            points = rotation.apply(mesh.vertices * scale) + np.asarray(transform.p)
            below = points[:, 2] < base_z
            if not below.any():
                continue
            # CoACD's voxel preprocessing can extend hulls below a flat base,
            # producing a first-step impulse and an uneven support footprint.
            points[below, 2] = base_z
            vertices = mesh.vertices.copy()
            vertices[below] = rotation.inv().apply(points[below] - np.asarray(transform.p)) / scale
            hull = ConvexHull(vertices)
            faces = hull.simplices.copy()
            normals = np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]]
            )
            reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0.0
            faces[reverse] = faces[reverse][:, (0, 2, 1)]
            # Copying geometry rebuilds mesh caches while preserving authored
            # mass properties; the exact shell and handle vertices are untouched.
            builder.shape_source[shape] = mesh.copy(vertices=vertices, indices=faces)


class TeapotSolver(CouplerSolver):
    """Couple the dynamic grasped pot to its exact MPM collision shell."""

    @classmethod
    def prepare_builder(cls, cfg, builder: ModelBuilder) -> None:
        # MPM preparation converts convex parts to triangle meshes; repair their generated support geometry first.
        repair_teapot_hulls(builder)
        super().prepare_builder(cfg, builder)

    @classmethod
    def build_coupled(cls, cfg, model, entries, proxies, mode):
        coupling = SolverCoupledProxy.Config(
            proxies=[SolverCoupledProxy.Proxy(**vars(proxy_cfg)) for proxy_cfg in proxies],
            iterations=cfg.iterations,
        )
        return _TeapotProxySolver(model=model, entries=entries, coupling=coupling)
