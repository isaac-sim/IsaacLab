# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Send a material rotation and bruise tint while retaining all static SH3 terms."""

import warp as wp

NAMES = tuple(f"primvars:squishyShRow{i}" for i in range(3))
QUATERNION_NAME = "primvars:squishyShQuaternion"


@wp.kernel
def material_frame(
    factors: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
    nearest: wp.array[int],
    damage: wp.array[float],
    row0: wp.array[wp.vec3],
    row1: wp.array[wp.vec3],
    row2: wp.array[wp.vec3],
):
    i = wp.tid()
    u, s, v = wp.svd3(factors[i] @ inverse[i])
    r = u @ wp.transpose(v)
    if wp.determinant(r) < 0.0:
        for j in range(3):
            u[j, 2] = -u[j, 2]
        r = u @ wp.transpose(v)
    tint = 1.0 - 0.45 * damage[nearest[i]]
    row0[i] = wp.vec3(r[0, 0], r[1, 0], r[2, 0]) * tint
    row1[i] = wp.vec3(r[0, 1], r[1, 1], r[2, 1]) * tint
    row2[i] = wp.vec3(r[0, 2], r[1, 2], r[2, 2]) * tint


class SHMaterialFrame:
    def __init__(self, sh, quaternion=False):
        self.sh = sh
        self.quaternion = quaternion
        self.names = (QUATERNION_NAME,) if quaternion else NAMES
        self.rows = [
            wp.empty(len(sh.inverse), dtype=wp.vec4 if quaternion else wp.vec3, device=sh.inverse.device)
            for _ in self.names
        ]

    def evaluate(self, damage, rotate=True):
        if rotate:
            kernel = material_quaternion if self.quaternion else material_frame
            inputs = [self.sh.binding.factors, self.sh.inverse, self.sh.nearest, damage, *self.rows]
        else:
            kernel = tint_quaternion if self.quaternion else tint_frame
            inputs = [self.sh.nearest, damage, *self.rows]
        wp.launch(kernel, dim=len(self.sh.inverse), inputs=inputs, device=self.sh.inverse.device)
        return dict(zip(self.names, self.rows))


@wp.kernel
def material_quaternion(
    factors: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
    nearest: wp.array[int],
    damage: wp.array[float],
    output: wp.array[wp.vec4],
):
    i = wp.tid()
    u, s, v = wp.svd3(factors[i] @ inverse[i])
    r = u @ wp.transpose(v)
    if wp.determinant(r) < 0.0:
        for j in range(3):
            u[j, 2] = -u[j, 2]
        r = u @ wp.transpose(v)
    q = wp.normalize(wp.quat_from_matrix(r))
    output[i] = wp.vec4(q[0], q[1], q[2], q[3]) * (1.0 - 0.45 * damage[nearest[i]])


@wp.kernel
def tint_quaternion(nearest: wp.array[int], damage: wp.array[float], output: wp.array[wp.vec4]):
    i = wp.tid()
    # Identity bypasses the MDL direction rotation, retaining the encoded bruise tint.
    output[i] = wp.vec4(0.0, 0.0, 0.0, 1.0 - 0.45 * damage[nearest[i]])


@wp.kernel
def tint_frame(
    nearest: wp.array[int],
    damage: wp.array[float],
    row0: wp.array[wp.vec3],
    row1: wp.array[wp.vec3],
    row2: wp.array[wp.vec3],
):
    i = wp.tid()
    tint = 1.0 - 0.45 * damage[nearest[i]]
    row0[i] = wp.vec3(tint, 0.0, 0.0)
    row1[i] = wp.vec3(0.0, tint, 0.0)
    row2[i] = wp.vec3(0.0, 0.0, tint)
