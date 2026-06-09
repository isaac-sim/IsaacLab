# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Counting particles inside closed meshes using Warp point-mesh queries.

This module provides :class:`ParticleMeshCounter`, a fast, solver-agnostic utility for counting
how many particles fall inside one or more closed (watertight) *region* meshes. It is intended for
training-time, privileged measurements such as "how many MPM media particles are inside the scoop
bowl / the source container / the target container" without relying on hand-tuned analytic regions.

The counter is built on Warp's BVH-accelerated point-mesh query
(:func:`warp.mesh_query_point_sign_winding_number`): each particle is transformed into a region's
local frame and tested for containment via the mesh winding number. The winding-number sign method
is robust for poorly conditioned, non-watertight meshes, which makes it a good default for region
geometry that is generated procedurally or extracted from USD assets.

Region meshes are static in their own local frame; only their per-environment world (or environment)
transform changes from step to step, so the BVH is built once and reused. The cost is therefore
``O(num_envs * num_particles * num_regions)`` queries, each ``O(log(num_faces))`` on the GPU.

The :func:`make_box_region_mesh` and :func:`make_frustum_region_mesh` helpers build watertight,
outward-oriented region meshes for the two most common regions of interest (axis-aligned boxes and
capped circular frusta / cup cavities).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import warp as wp

# Disable Warp's init banner and initialize the module so the kernel below is registered at import
# time (matches the pattern used in :mod:`isaaclab.utils.warp.ops`).
wp.config.quiet = True
wp.init()


@wp.func
def _is_finite_vec3(v: wp.vec3) -> bool:
    """Return ``True`` when all components of ``v`` are finite (no NaN / Inf)."""
    return not (
        wp.isnan(v[0]) or wp.isnan(v[1]) or wp.isnan(v[2]) or wp.isinf(v[0]) or wp.isinf(v[1]) or wp.isinf(v[2])
    )


@wp.kernel
def count_particles_in_meshes_kernel(
    particle_pos: wp.array2d(dtype=wp.vec3),
    region_mesh_ids: wp.array(dtype=wp.uint64),
    region_pos: wp.array2d(dtype=wp.vec3),
    region_quat: wp.array2d(dtype=wp.quat),
    max_query_dist: wp.float32,
    inside: wp.array3d(dtype=wp.float32),
):
    """Mark, per environment/particle/region, whether the particle is inside the region mesh.

    The thread grid is ``(num_envs, num_particles, num_regions)``. Each particle position is
    transformed into the region's local frame using the region's rigid transform and tested for
    containment with the mesh winding number. Non-finite particle positions are treated as outside.

    Args:
        particle_pos: Particle positions in a common frame, shape ``(num_envs, num_particles)``.
        region_mesh_ids: Warp mesh ids of the region meshes, shape ``(num_regions,)``.
        region_pos: Region origins in the same frame as ``particle_pos``, shape
            ``(num_regions, num_envs)``.
        region_quat: Region orientations as ``(x, y, z, w)`` quaternions, shape
            ``(num_regions, num_envs)``.
        max_query_dist: Maximum distance for the closest-point search [m].
        inside: Output containment flags (``1.0`` inside, ``0.0`` outside), shape
            ``(num_envs, num_particles, num_regions)``.
    """
    env_id, particle_id, region_id = wp.tid()
    point = particle_pos[env_id, particle_id]
    flag = wp.float32(0.0)
    if _is_finite_vec3(point):
        region_tf = wp.transform(region_pos[region_id, env_id], region_quat[region_id, env_id])
        point_local = wp.transform_point(wp.transform_inverse(region_tf), point)
        query = wp.mesh_query_point_sign_winding_number(region_mesh_ids[region_id], point_local, max_query_dist)
        # Warp convention: a negative winding-number sign means the point is inside the mesh.
        if query.result and query.sign < 0.0:
            flag = wp.float32(1.0)
    inside[env_id, particle_id, region_id] = flag


class ParticleMeshCounter:
    """Counts particles inside closed region meshes using Warp winding-number point queries.

    The counter owns one Warp mesh per region (built once with winding-number support) and, on every
    :meth:`count` call, transforms each environment's particles into each region's local frame to
    test containment. Regions may move and rotate between calls (e.g. a scoop bowl welded to a
    gripper); only their transforms are passed in, the geometry is fixed in its local frame.

    Positions and region transforms must be expressed in a *common* frame (typically the per-env
    frame or the world frame). The counter does not assume any particular frame.

    Note on input layouts: region transforms are region-major (``(num_regions, num_envs, ...)``)
    while particle positions are env-major (``(num_envs, num_particles, 3)``). Keep this
    transposition in mind when assembling inputs.

    Example:
        .. code-block:: python

            verts, faces = make_frustum_region_mesh(0.02, 0.04, -0.02, 0.03)
            counter = ParticleMeshCounter([(verts, faces)], num_envs=128, device="cuda:0")
            counts = counter.count(particle_pos_e, region_pos, region_quat)  # (num_envs, num_regions)
            in_bowl = counts[:, 0]

    Args:
        region_meshes: One entry per region, each either a built :class:`warp.Mesh` or a
            ``(vertices, indices)`` pair. ``vertices`` is shape ``(num_vertices, 3)`` [m]; ``indices``
            is the flattened or ``(num_faces, 3)`` triangle index array. A pre-built mesh is rebuilt
            on :paramref:`device` with winding-number support, so it need not have been created with
            ``support_winding_number=True``.
        num_envs: Number of environments.
        device: Torch device string the counter operates on (e.g. ``"cuda:0"`` or ``"cpu"``).
        max_query_dist: Maximum distance for the closest-point search [m]. Defaults to a large value
            so the winding-number sign is always resolved regardless of how deep a point sits inside.
    """

    def __init__(
        self,
        region_meshes: Sequence[wp.Mesh | tuple[np.ndarray, np.ndarray]],
        num_envs: int,
        device: str,
        *,
        max_query_dist: float = 1.0e6,
    ) -> None:
        if len(region_meshes) == 0:
            raise ValueError("`region_meshes` must contain at least one region mesh.")
        self._device = str(device)
        self._wp_device = wp.device_from_torch(torch.device(self._device))
        self._num_envs = int(num_envs)
        self._max_query_dist = float(max_query_dist)
        self._meshes: list[wp.Mesh] = [self._as_winding_mesh(mesh) for mesh in region_meshes]
        self._num_regions = len(self._meshes)
        self._mesh_ids = wp.array([mesh.id for mesh in self._meshes], dtype=wp.uint64, device=self._wp_device)
        # buffers sized on first use (depend on the per-call particle count)
        self._num_particles = 0
        self._inside_torch: torch.Tensor | None = None
        self._inside_wp: wp.array | None = None

    """Public properties."""

    @property
    def num_regions(self) -> int:
        """Number of region meshes."""
        return self._num_regions

    @property
    def num_envs(self) -> int:
        """Number of environments."""
        return self._num_envs

    @property
    def device(self) -> str:
        """Torch device string the counter operates on."""
        return self._device

    @property
    def meshes(self) -> list[wp.Mesh]:
        """The region meshes, one per region."""
        return self._meshes

    @property
    def mesh_ids(self) -> wp.array:
        """Warp mesh ids of the region meshes, shape ``(num_regions,)``."""
        return self._mesh_ids

    """Operations."""

    def count(
        self,
        particle_positions: torch.Tensor,
        region_positions: torch.Tensor,
        region_orientations: torch.Tensor | None = None,
        *,
        return_mask: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Count particles inside each region, per environment.

        Args:
            particle_positions: Particle positions in a common frame, shape
                ``(num_envs, num_particles, 3)`` [m].
            region_positions: Region origins in the same frame, shape ``(num_regions, num_envs, 3)``
                [m]. A ``(num_regions, 3)`` tensor is broadcast across environments (useful for
                regions that are static in the common frame).
            region_orientations: Region orientations as ``(x, y, z, w)`` quaternions, shape
                ``(num_regions, num_envs, 4)`` or ``(num_regions, 4)`` (broadcast). Defaults to
                identity for every region when ``None``.
            return_mask: When ``True``, also return the per-particle containment mask.

        Returns:
            The per-environment, per-region counts, shape ``(num_envs, num_regions)``, float. When
            :paramref:`return_mask` is ``True``, a tuple of the counts and the boolean containment
            mask of shape ``(num_envs, num_particles, num_regions)``.
        """
        inside = self._compute_inside_mask(particle_positions, region_positions, region_orientations)
        counts = inside.sum(dim=1)
        if return_mask:
            return counts, inside > 0.5
        return counts

    def _compute_inside_mask(
        self,
        particle_positions: torch.Tensor,
        region_positions: torch.Tensor,
        region_orientations: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the per-particle containment mask for every region (internal helper).

        Returns the reused internal buffer; the public :meth:`count` (with
        ``return_mask=True``) is the supported way to obtain a standalone mask.

        Args:
            particle_positions: Particle positions in a common frame, shape
                ``(num_envs, num_particles, 3)`` [m].
            region_positions: Region origins in the same frame, shape ``(num_regions, num_envs, 3)``
                or ``(num_regions, 3)`` (broadcast) [m].
            region_orientations: Region orientations as ``(x, y, z, w)`` quaternions, shape
                ``(num_regions, num_envs, 4)`` or ``(num_regions, 4)`` (broadcast), or ``None`` for
                identity.

        Returns:
            The containment mask (``1.0`` inside, ``0.0`` outside), shape
            ``(num_envs, num_particles, num_regions)``, float. The returned tensor is an internal,
            reused buffer; clone it if you need to retain it across calls.
        """
        points = particle_positions.to(device=self._device, dtype=torch.float32)
        if points.dim() != 3 or points.shape[0] != self._num_envs or points.shape[2] != 3:
            raise ValueError(
                f"`particle_positions` must have shape (num_envs={self._num_envs}, num_particles, 3),"
                f" got {tuple(particle_positions.shape)}."
            )
        points = points.contiguous()
        num_particles = points.shape[1]
        region_pos, region_quat = self._prepare_region_transforms(region_positions, region_orientations)
        self._ensure_buffers(num_particles)
        wp.launch(
            count_particles_in_meshes_kernel,
            dim=(self._num_envs, num_particles, self._num_regions),
            inputs=[
                wp.from_torch(points, dtype=wp.vec3),
                self._mesh_ids,
                wp.from_torch(region_pos, dtype=wp.vec3),
                wp.from_torch(region_quat, dtype=wp.quat),
                self._max_query_dist,
                self._inside_wp,
            ],
            device=self._wp_device,
        )
        return self._inside_torch

    """Helper functions."""

    def _as_winding_mesh(self, mesh: wp.Mesh | tuple[np.ndarray, np.ndarray]) -> wp.Mesh:
        """Build a winding-number-capable :class:`warp.Mesh` on the counter's device.

        ``mesh_query_point_sign_winding_number`` silently returns wrong signs for a
        mesh built without ``support_winding_number=True``, and Warp does not expose
        that flag on an existing :class:`warp.Mesh`. A pre-built mesh is therefore
        rebuilt from its points and indices so the winding-number BVH is guaranteed
        to exist; region meshes are built once, so this cost is paid only at setup.
        """
        if isinstance(mesh, wp.Mesh):
            vertices, indices = mesh.points.numpy(), mesh.indices.numpy()
        else:
            vertices, indices = mesh
        vertices = np.asarray(vertices, dtype=np.float32).reshape(-1, 3)
        indices = np.asarray(indices, dtype=np.int32).reshape(-1)
        return wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=self._wp_device),
            indices=wp.array(indices, dtype=wp.int32, device=self._wp_device),
            support_winding_number=True,
        )

    def _prepare_region_transforms(
        self, region_positions: torch.Tensor, region_orientations: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate and broadcast region transforms to ``(num_regions, num_envs, {3,4})``."""
        region_pos = region_positions.to(device=self._device, dtype=torch.float32)
        # Only broadcast the (num_regions, 3) form; an ill-shaped 2-D input falls through to the
        # shape check below so it raises a clear ValueError instead of a raw expand() error.
        if region_pos.dim() == 2 and tuple(region_pos.shape) == (self._num_regions, 3):
            region_pos = region_pos.unsqueeze(1).expand(self._num_regions, self._num_envs, 3)
        if tuple(region_pos.shape) != (self._num_regions, self._num_envs, 3):
            raise ValueError(
                f"`region_positions` must broadcast to (num_regions={self._num_regions},"
                f" num_envs={self._num_envs}, 3), got {tuple(region_positions.shape)}."
            )

        if region_orientations is None:
            region_quat = torch.zeros((self._num_regions, self._num_envs, 4), device=self._device, dtype=torch.float32)
            region_quat[..., 3] = 1.0
        else:
            region_quat = region_orientations.to(device=self._device, dtype=torch.float32)
            if region_quat.dim() == 2 and tuple(region_quat.shape) == (self._num_regions, 4):
                region_quat = region_quat.unsqueeze(1).expand(self._num_regions, self._num_envs, 4)
            if tuple(region_quat.shape) != (self._num_regions, self._num_envs, 4):
                raise ValueError(
                    f"`region_orientations` must broadcast to (num_regions={self._num_regions},"
                    f" num_envs={self._num_envs}, 4), got {tuple(region_orientations.shape)}."
                )
        return region_pos.contiguous(), region_quat.contiguous()

    def _ensure_buffers(self, num_particles: int) -> None:
        """(Re)allocate the containment buffer when the particle count changes."""
        if self._inside_torch is not None and self._num_particles == num_particles:
            return
        self._num_particles = num_particles
        self._inside_torch = torch.zeros(
            (self._num_envs, num_particles, self._num_regions), device=self._device, dtype=torch.float32
        )
        self._inside_wp = wp.from_torch(self._inside_torch, dtype=wp.float32)


def make_box_region_mesh(
    half_extents: Sequence[float], center: Sequence[float] = (0.0, 0.0, 0.0)
) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed, axis-aligned box region mesh with outward-facing triangles.

    Args:
        half_extents: Box half-extents ``(hx, hy, hz)`` [m].
        center: Box center in the mesh-local frame [m].

    Returns:
        A tuple of the vertices, shape ``(8, 3)`` float32 [m], and the triangle indices, shape
        ``(12, 3)`` int32.
    """
    hx, hy, hz = (float(half_extents[0]), float(half_extents[1]), float(half_extents[2]))
    if hx <= 0.0 or hy <= 0.0 or hz <= 0.0:
        raise ValueError(f"`half_extents` must be positive, got {(hx, hy, hz)}.")
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    vertices = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    ) + np.array([cx, cy, cz], dtype=np.float32)
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # -z
            [4, 5, 6],
            [4, 6, 7],  # +z
            [0, 1, 5],
            [0, 5, 4],  # -y
            [1, 2, 6],
            [1, 6, 5],  # +x
            [2, 3, 7],
            [2, 7, 6],  # +y
            [3, 0, 4],
            [3, 4, 7],  # -x
        ],
        dtype=np.int32,
    )
    return vertices, faces


def make_frustum_region_mesh(
    radius_bottom: float,
    radius_top: float,
    z_bottom: float,
    z_top: float,
    num_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a closed (capped) circular frustum region mesh aligned with the local +Z axis.

    This is the natural "cup cavity" region: a frustum that interpolates linearly in radius from
    :paramref:`radius_bottom` at :paramref:`z_bottom` to :paramref:`radius_top` at :paramref:`z_top`,
    capped at both ends so the mesh is watertight. Triangles face outward.

    Args:
        radius_bottom: Radius at the bottom ring [m].
        radius_top: Radius at the top ring [m].
        z_bottom: Local Z of the bottom ring [m].
        z_top: Local Z of the top ring [m].
        num_segments: Number of angular segments around the axis.

    Returns:
        A tuple of the vertices, shape ``(2 * num_segments + 2, 3)`` float32 [m], and the triangle
        indices, shape ``(4 * num_segments, 3)`` int32.
    """
    n = int(num_segments)
    if n < 3:
        raise ValueError(f"`num_segments` must be >= 3, got {num_segments}.")
    if radius_bottom <= 0.0 or radius_top <= 0.0:
        raise ValueError(f"Radii must be positive, got bottom={radius_bottom}, top={radius_top}.")
    if z_bottom >= z_top:
        raise ValueError(f"`z_bottom` must be < `z_top`, got {z_bottom} >= {z_top}.")
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    bottom = np.stack([radius_bottom * cos_a, radius_bottom * sin_a, np.full(n, z_bottom)], axis=1)
    top = np.stack([radius_top * cos_a, radius_top * sin_a, np.full(n, z_top)], axis=1)
    center_bottom = np.array([[0.0, 0.0, z_bottom]])
    center_top = np.array([[0.0, 0.0, z_top]])
    vertices = np.concatenate([bottom, top, center_bottom, center_top], axis=0).astype(np.float32)

    idx_center_bottom, idx_center_top = 2 * n, 2 * n + 1
    faces = []
    for i in range(n):
        j = (i + 1) % n
        b_i, b_j, t_i, t_j = i, j, n + i, n + j
        # side wall (outward)
        faces.append([b_i, b_j, t_j])
        faces.append([b_i, t_j, t_i])
        # bottom cap (outward = -Z)
        faces.append([idx_center_bottom, b_j, b_i])
        # top cap (outward = +Z)
        faces.append([idx_center_top, t_i, t_j])
    return vertices, np.array(faces, dtype=np.int32)
