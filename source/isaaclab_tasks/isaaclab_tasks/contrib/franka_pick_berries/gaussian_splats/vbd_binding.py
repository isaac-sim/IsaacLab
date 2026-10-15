# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""How the berry's Gaussians follow its tetrahedral mesh with the VBD tissue solver: linear skinning to the mesh.

The counterpart of :mod:`.mpm_binding` for a mesh. The berry's asset binds each Gaussian to a tetrahedron, by the
barycentric coordinates of its anchor: the Gaussian's center when the mesh contains it, else the nearest point of the
tetrahedron. On every frame :meth:`VBDBinding.deform` computes each
tetrahedron's deformation gradient from its nodes, averages it at the nodes by volume, and interpolates it at each
Gaussian with the same coordinates, which keeps the deformation continuous across tetrahedra. The Gaussian's anchor
moves with the nodes, its offset from the anchor and its covariance with the interpolated gradient; its scales,
orientation and shading follow as with MPM particles (:func:`.mpm_binding.gaussian_shape`).
"""

import numpy as np
import warp as wp
from scipy.spatial.transform import Rotation

from .mpm_binding import BRUISE_DARKENING, gaussian_shape


@wp.kernel
def tet_gradients(
    tets: wp.array[wp.vec4i],
    rest_inverse: wp.array[wp.mat33],
    volume: wp.array[float],
    nodes: wp.array[wp.vec3],
    node_gradient: wp.array[wp.mat33],
):
    """Add each tetrahedron's deformation gradient, weighted by its rest volume, to its four nodes."""
    t = wp.tid()
    tet = tets[t]
    x0 = nodes[tet[0]]
    edges = wp.matrix_from_cols(nodes[tet[1]] - x0, nodes[tet[2]] - x0, nodes[tet[3]] - x0)
    weighted = volume[t] * (edges @ rest_inverse[t])
    for k in range(4):
        wp.atomic_add(node_gradient, tet[k], weighted)


@wp.kernel
def skin_gaussians(
    tets: wp.array[wp.vec4i],
    tet: wp.array[int],
    barycentric: wp.array[wp.vec4],
    offset: wp.array[wp.vec3],
    base: wp.array[wp.mat33],
    nodes: wp.array[wp.vec3],
    node_gradient: wp.array[wp.mat33],
    node_volume: wp.array[float],
    xyz: wp.array[wp.vec3],
    gradient: wp.array[wp.mat33],
    covariance: wp.array[wp.mat33],
):
    """Move each Gaussian's anchor with its tetrahedron's nodes, and its offset and covariance factor with the
    deformation gradient interpolated at it."""
    i = wp.tid()
    corners = tets[tet[i]]
    anchor = wp.vec3(0.0)
    f = wp.mat33(0.0)
    for k in range(4):
        node = corners[k]
        anchor += barycentric[i][k] * nodes[node]
        f += (barycentric[i][k] / node_volume[node]) * node_gradient[node]
    # Bound the stretch, so that a Gaussian never collapses or balloons.
    u, s, v = wp.svd3(f)
    for axis in range(3):
        s[axis] = wp.clamp(s[axis], 0.08, 3.0)
    f = u @ wp.diag(s) @ wp.transpose(v)
    xyz[i] = anchor + f @ offset[i]
    gradient[i] = f
    covariance[i] = f @ base[i]


class VBDBinding:
    """Bind a berry's Gaussians to its tetrahedral mesh at rest, then deform and shade them from the mesh's nodes.

    Has the interface of :class:`.mpm_binding.MPMBinding`.

    Args:
        gaussians: The Gaussians at rest: ``xyz`` centers [m], ``scales`` [m], ``rotations`` (xyzw quaternions), and
            their ``tet`` and ``barycentric`` binding from the asset.
        nodes: Node rest positions [m].
        tets: The four node indices of each tetrahedron.
        device: Warp device of the nodes.
    """

    def __init__(self, gaussians: dict, nodes: np.ndarray, tets: np.ndarray, device):
        tet = np.asarray(gaussians["tet"], np.int32)
        barycentric = np.asarray(gaussians["barycentric"], np.float32).reshape(-1, 4)
        corners = nodes[tets]
        edges = np.transpose(corners[:, 1:] - corners[:, :1], (0, 2, 1))
        volume = np.linalg.det(edges) / 6
        if np.any(volume <= 0):
            raise ValueError("The berry's tetrahedral mesh has inverted or flat tetrahedra")
        node_volume = np.bincount(tets.reshape(-1), np.repeat(volume, 4), minlength=len(nodes))
        offset = gaussians["xyz"] - np.einsum("nk,nkj->nj", barycentric, corners[tet])
        base = Rotation.from_quat(gaussians["rotations"]).as_matrix() * gaussians["scales"][:, None]
        count = len(tet)
        with wp.ScopedDevice(device):
            self._tets = wp.array(tets, dtype=wp.vec4i)
            self._rest_inverse = wp.array(np.linalg.inv(edges).astype(np.float32), dtype=wp.mat33)
            self._volume = wp.array(volume.astype(np.float32), dtype=float)
            self._node_volume = wp.array(node_volume.astype(np.float32), dtype=float)
            self._node_gradient = wp.empty(len(nodes), dtype=wp.mat33)
            self._tet = wp.array(tet, dtype=int)
            self._barycentric = wp.array(barycentric, dtype=wp.vec4)
            self._offset = wp.array(offset.astype(np.float32), dtype=wp.vec3)
            self._base = wp.array(base.astype(np.float32), dtype=wp.mat33)
            # Bruises show the damage of each Gaussian's most weighted node.
            self._nearest = wp.array(tets[tet, np.argmax(barycentric, axis=1)], dtype=int)
            self.positions = wp.empty(count, dtype=wp.vec3)
            self.scales = wp.empty(count, dtype=wp.vec3)
            self.orientations = wp.empty(count, dtype=wp.quat)
            self.shading = wp.empty(count, dtype=wp.vec4)
            self._gradient = wp.empty(count, dtype=wp.mat33)
            self._covariance = wp.empty(count, dtype=wp.mat33)

    def deform(self, nodes: wp.array, damage: wp.array) -> None:
        """Update :attr:`positions` [m], :attr:`scales` [m], :attr:`orientations` and :attr:`shading` on the device.

        Args:
            nodes: Current node positions [m], in the same frame as at rest.
            damage: Tissue damage of each node, from 0 (intact) to 1.
        """
        count = len(self.positions)
        self._node_gradient.zero_()
        wp.launch(
            tet_gradients,
            dim=len(self._tets),
            inputs=[self._tets, self._rest_inverse, self._volume, nodes],
            outputs=[self._node_gradient],
        )
        wp.launch(
            skin_gaussians,
            dim=count,
            inputs=[
                self._tets,
                self._tet,
                self._barycentric,
                self._offset,
                self._base,
                nodes,
                self._node_gradient,
                self._node_volume,
            ],
            outputs=[self.positions, self._gradient, self._covariance],
        )
        wp.launch(
            gaussian_shape,
            dim=count,
            inputs=[self._covariance, self._gradient, self._nearest, damage, BRUISE_DARKENING],
            outputs=[self.scales, self.orientations, self.shading],
        )
