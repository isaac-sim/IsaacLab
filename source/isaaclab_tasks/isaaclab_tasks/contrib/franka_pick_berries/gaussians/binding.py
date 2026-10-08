# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How the berry's Gaussians follow its MPM particles: the bridge between physics and rendering.

The berry's appearance is a few hundred thousand 3D Gaussians, far more than the few thousand MPM particles that carry
its physics. Each Gaussian is bound once, at rest, to its 32 nearest particles of the same tissue region. On every
frame :meth:`GaussianBinding.deform` fits, for each Gaussian, the affine motion of those particles by moving least
squares: the Gaussian's center moves with them and its covariance (scales and orientation) stretches and turns with
the fitted deformation gradient. The binding is one-way: the Gaussians never act on the physics.

Two details keep the berry looking like a berry when it is handled roughly:

* **Tearing.** A neighbor that has separated from the Gaussian's anchor particle well beyond its rest distance is
  faded out of the fit, so that Gaussians on either side of a tear follow their own side instead of smearing across
  the gap. When the nearest particle itself was torn away (a lone particle left behind), the Gaussian re-anchors on
  whichever of its next nearest particles keeps most of its neighborhood.
* **Shading.** A Gaussian's color is degree-three spherical harmonics, which describe how it looks from each
  direction in the berry's rest frame. The rotation part of the fitted deformation is sent to the renderer as a
  quaternion, so that the shader rotates the viewing direction into the rest frame: highlights turn with the tissue.
  The quaternion's length darkens the color with the tissue's damage, which shows bruises.
"""

import numpy as np
import warp as wp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

NEIGHBORS = 32
"""Particles each Gaussian follows."""

BRUISE_DARKENING = 0.45
"""Fraction by which fully damaged tissue darkens its Gaussians."""


@wp.func
def fracture_weight(
    distance: float, rest: wp.vec3, current: wp.vec3, rest_anchor: wp.vec3, current_anchor: wp.vec3, spacing: float
):
    """Inverse-square weight of a neighbor, faded out as it separates from the anchor beyond its rest distance."""
    weight = 1.0 / wp.max(distance * distance, 0.0008 * 0.0008)
    original = wp.length(rest - rest_anchor)
    separation = wp.length(current - current_anchor)
    t = wp.clamp((separation - 1.5 * original - 0.25 * spacing) / (0.5 * original + 0.25 * spacing), 0.0, 1.0)
    return weight * (1.0 - t * t * (3.0 - 2.0 * t))


@wp.kernel
def deform_gaussians(
    ids: wp.array2d[int],
    counts: wp.array[int],
    rest: wp.array[wp.vec3],
    source: wp.array[wp.vec3],
    base: wp.array[wp.mat33],
    positions: wp.array[wp.vec3],
    spacing: float,
    xyz: wp.array[wp.vec3],
    gradient: wp.array[wp.mat33],
    covariance: wp.array[wp.mat33],
):
    """Fit each Gaussian's deformation gradient to its particles and move its center and covariance factor with it."""
    i = wp.tid()
    # A lone particle left behind keeps almost none of its neighbors. Its Gaussians then anchor to whichever of their
    # next nearest particles keeps the most, instead of following it. Across a tear, the nearest particle keeps its
    # side's neighbors and stays the anchor, so that each Gaussian stays on its own side.
    anchor = ids[i, 0]
    best = float(-1.0)
    for c in range(wp.min(counts[i], 4)):
        candidate = ids[i, c]
        kept = float(0.0)
        total = float(0.0)
        for k in range(counts[i]):
            j = ids[i, k]
            if j != candidate:
                distance = wp.length(rest[j] - source[i])
                kept += fracture_weight(distance, rest[j], positions[j], rest[candidate], positions[candidate], spacing)
                total += 1.0 / wp.max(distance * distance, 0.0008 * 0.0008)
        if c == 0 and kept >= 0.25 * total:
            break
        if kept > best:
            best = kept
            anchor = candidate
    rest_anchor = rest[anchor]
    current_anchor = positions[anchor]
    # Weighted centers of the neighborhood, at rest and now.
    weight_sum = float(0.0)
    rest_center = wp.vec3(0.0)
    current_center = wp.vec3(0.0)
    for k in range(counts[i]):
        j = ids[i, k]
        distance = wp.length(rest[j] - source[i])
        weight = fracture_weight(distance, rest[j], positions[j], rest_anchor, current_anchor, spacing)
        weight_sum += weight
        rest_center += weight * rest[j]
        current_center += weight * positions[j]
    rest_center /= weight_sum
    current_center /= weight_sum
    # Least-squares affine fit: current - current_center = F (rest - rest_center).
    moment = wp.mat33(0.0)
    cross = wp.mat33(0.0)
    for k in range(counts[i]):
        j = ids[i, k]
        distance = wp.length(rest[j] - source[i])
        weight = fracture_weight(distance, rest[j], positions[j], rest_anchor, current_anchor, spacing) / weight_sum
        r = rest[j] - rest_center
        moment += weight * wp.outer(r, r)
        cross += weight * wp.outer(positions[j] - current_center, r)
    # Continue with the best-fit rotation when the neighborhood becomes rank deficient; a display approximation only.
    rotation_u, rotation_s, rotation_v = wp.svd3(cross)
    rotation = rotation_u @ wp.transpose(rotation_v)
    if wp.determinant(rotation) < 0.0:
        for row in range(3):
            rotation_u[row, 2] = -rotation_u[row, 2]
        rotation = rotation_u @ wp.transpose(rotation_v)
    epsilon = spacing * spacing * 1.0e-6
    f = (cross + epsilon * rotation) @ wp.inverse(moment + epsilon * wp.identity(3, dtype=float))
    # Bound the stretch, so that a Gaussian never collapses or balloons.
    u, s, v = wp.svd3(f)
    for axis in range(3):
        s[axis] = wp.clamp(s[axis], 0.08, 3.0)
    f = u @ wp.diag(s) @ wp.transpose(v)
    xyz[i] = current_center + f @ (source[i] - rest_center)
    gradient[i] = f
    covariance[i] = f @ base[i]


@wp.kernel
def gaussian_shape(
    covariance: wp.array[wp.mat33],
    gradient: wp.array[wp.mat33],
    nearest: wp.array[int],
    damage: wp.array[float],
    darkening: float,
    scales: wp.array[wp.vec3],
    orientations: wp.array[wp.quat],
    shading: wp.array[wp.vec4],
):
    """Split each covariance factor into scales and an orientation, and encode the SH rotation and bruise tint."""
    i = wp.tid()
    u, s, v = wp.svd3(covariance[i])
    if wp.determinant(u) < 0.0:
        for row in range(3):
            u[row, 2] = -u[row, 2]
    scales[i] = wp.vec3(wp.abs(s[0]), wp.abs(s[1]), wp.abs(s[2]))
    orientations[i] = wp.quat_from_matrix(u)
    # Rotation part of the deformation gradient, by polar decomposition.
    u, s, v = wp.svd3(gradient[i])
    r = u @ wp.transpose(v)
    if wp.determinant(r) < 0.0:
        for j in range(3):
            u[j, 2] = -u[j, 2]
        r = u @ wp.transpose(v)
    q = wp.normalize(wp.quat_from_matrix(r))
    shading[i] = wp.vec4(q[0], q[1], q[2], q[3]) * (1.0 - darkening * damage[nearest[i]])


class GaussianBinding:
    """Bind a berry's Gaussians to its MPM particles at rest, then deform and shade them from the particles.

    Args:
        gaussians: The Gaussians at rest: ``xyz`` centers [m], ``scales`` [m], ``rotations`` (xyzw quaternions) and
            ``regions``, the tissue region of each.
        particles: Particle rest positions [m].
        regions: Tissue region of each particle; Gaussians only follow particles of their own region.
        device: Warp device of the particles.
    """

    def __init__(self, gaussians: dict, particles: np.ndarray, regions: np.ndarray, device):
        xyz = np.asarray(gaussians["xyz"], np.float32)
        count = len(xyz)
        ids = np.empty((count, NEIGHBORS), np.int32)
        for region in np.unique(gaussians["regions"]):
            selected = np.flatnonzero(gaussians["regions"] == region)
            members = np.flatnonzero(regions == region)
            if len(members) == 0:
                raise ValueError(f"No particles in tissue region {region} of the Gaussians")
            k = min(NEIGHBORS, len(members))
            local = cKDTree(particles[members]).query(xyz[selected], k=k, workers=-1)[1].reshape(len(selected), k)
            ids[selected, :k] = members[local]
            ids[selected, k:] = members[0]
        counts = np.minimum(NEIGHBORS, np.bincount(regions)[regions[ids[:, 0]]])
        # Covariance factor at rest: rotation times scales.
        base = Rotation.from_quat(gaussians["rotations"]).as_matrix() * gaussians["scales"][:, None]
        self.spacing = float(np.median(cKDTree(particles).query(particles, k=2)[0][:, 1]))
        with wp.ScopedDevice(device):
            self._ids = wp.array(ids, dtype=int)
            # Bruises show the damage of each Gaussian's nearest particle.
            self._nearest = wp.array(ids[:, 0].copy(), dtype=int)
            self._counts = wp.array(counts.astype(np.int32), dtype=int)
            self._rest = wp.array(np.asarray(particles, np.float32), dtype=wp.vec3)
            self._source = wp.array(xyz, dtype=wp.vec3)
            self._base = wp.array(base.astype(np.float32), dtype=wp.mat33)
            self.positions = wp.empty(count, dtype=wp.vec3)
            self.scales = wp.empty(count, dtype=wp.vec3)
            self.orientations = wp.empty(count, dtype=wp.quat)
            self.shading = wp.empty(count, dtype=wp.vec4)
            self._gradient = wp.empty(count, dtype=wp.mat33)
            self._covariance = wp.empty(count, dtype=wp.mat33)

    def deform(self, particles: wp.array, damage: wp.array) -> None:
        """Update :attr:`positions` [m], :attr:`scales` [m], :attr:`orientations` and :attr:`shading` on the device.

        Args:
            particles: Current particle positions [m], in the same frame as at rest.
            damage: Tissue damage of each particle, from 0 (intact) to 1.
        """
        count = len(self.positions)
        wp.launch(
            deform_gaussians,
            dim=count,
            inputs=[self._ids, self._counts, self._rest, self._source, self._base, particles, self.spacing],
            outputs=[self.positions, self._gradient, self._covariance],
        )
        wp.launch(
            gaussian_shape,
            dim=count,
            inputs=[self._covariance, self._gradient, self._nearest, damage, BRUISE_DARKENING],
            outputs=[self.scales, self.orientations, self.shading],
        )
