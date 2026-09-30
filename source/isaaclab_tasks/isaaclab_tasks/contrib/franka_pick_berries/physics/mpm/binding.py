# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""One-way, rest-space binding and correct affine Gaussian covariance transport."""

import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@wp.kernel
def transport(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    offsets: wp.array2d[wp.vec3],
    base: wp.array[wp.mat33],
    positions: wp.array[wp.vec3],
    frames: wp.array[wp.mat33],
    xyz: wp.array[wp.vec3],
    scales: wp.array[wp.vec3],
    rotations: wp.array[wp.quat],
):
    i = wp.tid()
    x = wp.vec3(0.0)
    f = wp.mat33(0.0)
    for k in range(4):
        j = ids[i, k]
        w = weights[i, k]
        x += w * (positions[j] + frames[j] @ offsets[i, k])
        f += w * frames[j]
    u, sigma, v = wp.svd3(f @ base[i])
    if wp.determinant(u) < 0.0:
        for row in range(3):
            u[row, 2] = -u[row, 2]
    xyz[i] = x
    scales[i] = wp.vec3(
        wp.max(wp.abs(sigma[0]), 1.0e-9), wp.max(wp.abs(sigma[1]), 1.0e-9), wp.max(wp.abs(sigma[2]), 1.0e-9)
    )
    rotations[i] = wp.quat_from_matrix(u)


@wp.kernel
def transport_factors(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    offsets: wp.array2d[wp.vec3],
    base: wp.array[wp.mat33],
    positions: wp.array[wp.vec3],
    frames: wp.array[wp.mat33],
    xyz: wp.array[wp.vec3],
    affine: wp.array[wp.mat33],
):
    i = wp.tid()
    x = wp.vec3(0.0)
    f = wp.mat33(0.0)
    for k in range(4):
        j = ids[i, k]
        w = weights[i, k]
        x += w * (positions[j] + frames[j] @ offsets[i, k])
        f += w * frames[j]
    xyz[i] = x
    affine[i] = f @ base[i]


class Binding:
    def __init__(self, splats, rest, regions):
        self.xyz = splats["xyz"]
        nearest = cKDTree(rest).query(self.xyz, workers=-1)[1]
        groups = splats.get("regions", regions[nearest])
        if len(groups) != len(self.xyz) or not np.isin(groups, np.unique(regions)).all():
            raise ValueError("Every visual sample must bind to an existing physical region")
        self.ids = np.empty((len(self.xyz), 4), np.int32)
        self.weights = np.empty((len(self.xyz), 4), np.float32)
        for group in np.unique(groups):
            selected = np.flatnonzero(groups == group)
            members = np.flatnonzero(regions == group)
            k = min(4, len(members))
            dist, local = cKDTree(rest[members]).query(self.xyz[selected], k=k, workers=-1)
            if k == 1:
                dist, local = dist[:, None], local[:, None]
            if k < 4:
                dist = np.pad(dist, ((0, 0), (0, 4 - k)), constant_values=np.inf)
                local = np.pad(local, ((0, 0), (0, 4 - k)), mode="edge")
            weight = 1 / np.maximum(dist, 0.0002) ** 2
            self.ids[selected] = members[local]
            self.weights[selected] = weight / weight.sum(axis=1, keepdims=True)
        self.offset = self.xyz[:, None] - rest[self.ids]
        self.base = (Rotation.from_quat(splats["rotations"]).as_matrix() * splats["scales"][:, None]).astype(np.float32)
        self.gpu = None

    def deform_gpu(self, positions, frames, host=True):
        if self.gpu is None:
            wp.init()
            wp.set_device("cuda:0")
            self.gpu = [
                wp.array(self.ids, dtype=int),
                wp.array(self.weights, dtype=float),
                wp.array(self.offset, dtype=wp.vec3),
                wp.array(self.base, dtype=wp.mat33),
                wp.empty(len(self.xyz), dtype=wp.vec3),
                wp.empty(len(self.xyz), dtype=wp.vec3),
                wp.empty(len(self.xyz), dtype=wp.quat),
            ]
        p = wp.array(positions, dtype=wp.vec3) if isinstance(positions, np.ndarray) else positions
        f = wp.array(frames, dtype=wp.mat33) if isinstance(frames, np.ndarray) else frames
        wp.launch(transport, dim=len(self.xyz), inputs=[*self.gpu[:4], p, f, *self.gpu[4:]])
        return tuple(a.numpy() for a in self.gpu[4:]) if host else tuple(self.gpu[4:])

    def factors_gpu(self, positions, frames):
        if self.gpu is None:
            self.deform_gpu(positions, frames, host=False)
        if not hasattr(self, "affine_gpu"):
            self.affine_gpu = wp.empty(len(self.xyz), dtype=wp.mat33)
        wp.launch(
            transport_factors,
            dim=len(self.xyz),
            inputs=[*self.gpu[:4], positions, frames, self.gpu[4], self.affine_gpu],
        )
        return self.gpu[4].numpy(), self.affine_gpu.numpy()

    def deform(self, positions, frames):
        local_f = frames[self.ids]
        xyz = np.sum(
            self.weights[:, :, None] * (positions[self.ids] + np.einsum("nkij,nkj->nki", local_f, self.offset)), axis=1
        )
        gradient = np.sum(self.weights[:, :, None, None] * local_f, axis=1)
        affine = gradient @ self.base
        u, scale, _ = np.linalg.svd(affine)
        negative = np.linalg.det(u) < 0
        u[negative, :, -1] *= -1
        rotations = Rotation.from_matrix(u).as_quat().astype(np.float32)
        return xyz.astype(np.float32), np.maximum(scale, 1.0e-9).astype(np.float32), rotations


@wp.kernel
def transport_mls(
    ids: wp.array2d[int],
    weights: wp.array2d[float],
    gradients: wp.array2d[wp.vec3],
    base: wp.array[wp.mat33],
    offsets: wp.array[wp.vec3],
    coherent: int,
    positions: wp.array[wp.vec3],
    xyz: wp.array[wp.vec3],
    affine: wp.array[wp.mat33],
):
    i = wp.tid()
    x = wp.vec3(0.0)
    f = wp.mat33(0.0)
    for k in range(ids.shape[1]):
        p = positions[ids[i, k]]
        x += weights[i, k] * p
        f += wp.outer(p, gradients[i, k])
    raw_f = f
    u, s, v = wp.svd3(f)
    for axis in range(3):
        s[axis] = wp.clamp(s[axis], 0.08, 3.0)
    f = u @ wp.diag(s) @ wp.transpose(v)
    if coherent != 0:
        x += (f - raw_f) @ offsets[i]
    xyz[i] = x
    affine[i] = f @ base[i]


@wp.kernel
def factor_components(factors: wp.array[wp.mat33], scales: wp.array[wp.vec3], rotations: wp.array[wp.quat]):
    i = wp.tid()
    u, s, v = wp.svd3(factors[i])
    if wp.determinant(u) < 0.0:
        for row in range(3):
            u[row, 2] = -u[row, 2]
    scales[i] = wp.vec3(wp.abs(s[0]), wp.abs(s[1]), wp.abs(s[2]))
    rotations[i] = wp.quat_from_matrix(u)


class MLSBinding:
    """Affine moving least squares skin from 32 same-region physical samples.

    Rendering only. Physical positions determine the fit; integrated display
    frames and all appearance data remain outside the simulation.
    """

    def __init__(self, splats, rest, regions, neighbors=32, coherent=False):
        self.xyz = splats["xyz"]
        self.coherent = coherent
        self.offsets = np.empty_like(self.xyz)
        nearest = cKDTree(rest).query(self.xyz, workers=-1)[1]
        groups = splats.get("regions", regions[nearest])
        if not np.isin(groups, np.unique(regions)).all():
            raise ValueError("Unknown physical region in visual binding")
        self.ids = np.empty((len(self.xyz), neighbors), np.int32)
        self.weights = np.zeros((len(self.xyz), neighbors), np.float32)
        self.gradients = np.zeros((len(self.xyz), neighbors, 3), np.float32)
        for region in np.unique(groups):
            selected = np.flatnonzero(groups == region)
            members = np.flatnonzero(regions == region)
            count = min(neighbors, len(members))
            distance, local = cKDTree(rest[members]).query(self.xyz[selected], k=count, workers=-1)
            ids = members[local]
            w = 1.0 / np.maximum(distance, 0.0008) ** 2
            w /= w.sum(axis=1, keepdims=True)
            center = np.einsum("nk,nki->ni", w, rest[ids])
            self.offsets[selected] = self.xyz[selected] - center
            delta = rest[ids].astype(float) - center[:, None]
            moment = np.einsum("nk,nki,nkj->nij", w, delta, delta)
            if np.any(np.linalg.cond(moment) > 1.0e8):
                raise ValueError("Rank-deficient visual neighborhood")
            gradient = w[:, :, None] * np.einsum("nij,nkj->nki", np.linalg.inv(moment), delta)
            weights = w + np.einsum("ni,nki->nk", self.xyz[selected] - center, gradient)
            self.ids[selected, :count] = ids
            self.ids[selected, count:] = members[0]
            self.weights[selected, :count] = weights
            self.gradients[selected, :count] = gradient
        self.base = (Rotation.from_quat(splats["rotations"]).as_matrix() * splats["scales"][:, None]).astype(np.float32)
        self.gpu = None

    def _update(self, positions):
        if self.gpu is None:
            wp.init()
            wp.set_device("cuda:0")
            self.gpu = [
                wp.array(self.ids, dtype=int),
                wp.array(self.weights, dtype=float),
                wp.array(self.gradients, dtype=wp.vec3),
                wp.array(self.base, dtype=wp.mat33),
                wp.array(self.offsets, dtype=wp.vec3),
            ]
            self.positions = wp.empty(len(self.xyz), dtype=wp.vec3)
            self.factors = wp.empty(len(self.xyz), dtype=wp.mat33)
            self.scales = wp.empty(len(self.xyz), dtype=wp.vec3)
            self.rotations = wp.empty(len(self.xyz), dtype=wp.quat)
        p = wp.array(positions, dtype=wp.vec3) if isinstance(positions, np.ndarray) else positions
        wp.launch(
            transport_mls, dim=len(self.xyz), inputs=[*self.gpu, int(self.coherent), p, self.positions, self.factors]
        )

    def factors_gpu(self, positions, frames=None):
        self._update(positions)
        return self.positions.numpy(), self.factors.numpy()

    def deform_gpu(self, positions, frames=None, host=True):
        self._update(positions)
        wp.launch(factor_components, dim=len(self.xyz), inputs=[self.factors, self.scales, self.rotations])
        arrays = (self.positions, self.scales, self.rotations)
        return tuple(a.numpy() for a in arrays) if host else arrays


@wp.kernel
def transport_fracture_mls(
    ids: wp.array2d[int],
    counts: wp.array[int],
    rest: wp.array[wp.vec3],
    source: wp.array[wp.vec3],
    base: wp.array[wp.mat33],
    positions: wp.array[wp.vec3],
    xyz: wp.array[wp.vec3],
    affine: wp.array[wp.mat33],
    spacing: float,
):
    i = wp.tid()
    anchor = ids[i, 0]
    rest_anchor = rest[anchor]
    current_anchor = positions[anchor]
    weight_sum = float(0.0)
    rest_center = wp.vec3(0.0)
    current_center = wp.vec3(0.0)
    for k in range(counts[i]):
        j = ids[i, k]
        distance = wp.length(rest[j] - source[i])
        weight = 1.0 / wp.max(distance * distance, 0.0008 * 0.0008)
        original = wp.length(rest[j] - rest_anchor)
        current = wp.length(positions[j] - current_anchor)
        t = wp.clamp((current - 1.5 * original - 0.25 * spacing) / (0.5 * original + 0.25 * spacing), 0.0, 1.0)
        weight *= 1.0 - t * t * (3.0 - 2.0 * t)
        weight_sum += weight
        rest_center += weight * rest[j]
        current_center += weight * positions[j]
    rest_center /= weight_sum
    current_center /= weight_sum
    moment = wp.mat33(0.0)
    cross = wp.mat33(0.0)
    for k in range(counts[i]):
        j = ids[i, k]
        distance = wp.length(rest[j] - source[i])
        weight = 1.0 / wp.max(distance * distance, 0.0008 * 0.0008) / weight_sum
        original = wp.length(rest[j] - rest_anchor)
        current = wp.length(positions[j] - current_anchor)
        t = wp.clamp((current - 1.5 * original - 0.25 * spacing) / (0.5 * original + 0.25 * spacing), 0.0, 1.0)
        weight *= 1.0 - t * t * (3.0 - 2.0 * t)
        r = rest[j] - rest_center
        moment += weight * wp.outer(r, r)
        cross += weight * wp.outer(positions[j] - current_center, r)
    # Continue with the best-fit rotation when support becomes rank deficient.
    # This regularization is a display approximation, never a physical force.
    rotation_u, rotation_s, rotation_v = wp.svd3(cross)
    rotation = rotation_u @ wp.transpose(rotation_v)
    if wp.determinant(rotation) < 0.0:
        for row in range(3):
            rotation_u[row, 2] = -rotation_u[row, 2]
        rotation = rotation_u @ wp.transpose(rotation_v)
    epsilon = spacing * spacing * 1.0e-6
    identity = wp.identity(3, dtype=float)
    f = (cross + epsilon * rotation) @ wp.inverse(moment + epsilon * identity)
    u, s, v = wp.svd3(f)
    for axis in range(3):
        s[axis] = wp.clamp(s[axis], 0.08, 3.0)
    f = u @ wp.diag(s) @ wp.transpose(v)
    xyz[i] = current_center + f @ (source[i] - rest_center)
    affine[i] = f @ base[i]


class FractureMLSBinding(MLSBinding):
    """One-way skin fit that excludes excessively separated material neighbors."""

    def __init__(self, splats, rest, regions):
        super().__init__(splats, rest, regions, coherent=True)
        self.rest = np.asarray(rest, np.float32)
        self.spacing = float(np.median(cKDTree(rest).query(rest, k=2)[0][:, 1]))
        self.counts = np.minimum(self.ids.shape[1], np.bincount(regions)[regions[self.ids[:, 0]]]).astype(np.int32)

    def _update(self, positions):
        if self.gpu is None:
            super()._update(positions)
            self.rest_gpu = wp.array(self.rest, dtype=wp.vec3)
            self.source_gpu = wp.array(self.xyz, dtype=wp.vec3)
            self.counts_gpu = wp.array(self.counts, dtype=int)
        p = wp.array(positions, dtype=wp.vec3) if isinstance(positions, np.ndarray) else positions
        wp.launch(
            transport_fracture_mls,
            dim=len(self.xyz),
            inputs=[
                self.gpu[0],
                self.counts_gpu,
                self.rest_gpu,
                self.source_gpu,
                self.gpu[3],
                p,
                self.positions,
                self.factors,
                self.spacing,
            ],
        )


def make_binding(splats, rest, regions, mode="affine4"):
    if mode == "affine4":
        return Binding(splats, rest, regions)
    if mode == "mls32":
        return MLSBinding(splats, rest, regions)
    if mode == "mls32-coherent":
        return MLSBinding(splats, rest, regions, coherent=True)
    if mode == "mls32-fracture":
        return FractureMLSBinding(splats, rest, regions)
    raise ValueError(f"Unknown visual binding: {mode}")
