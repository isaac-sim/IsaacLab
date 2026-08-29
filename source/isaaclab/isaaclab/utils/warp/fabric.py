# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: ignore
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).  # noqa: E501
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp kernels for GPU-accelerated Fabric operations."""

from typing import TYPE_CHECKING, Any

import warp as wp

if TYPE_CHECKING:
    FabricArrayUInt32 = Any
    FabricArrayMat44d = Any
    IndexedFabricArrayMat44d = Any
    ArrayUInt32 = Any
    ArrayUInt32_1d = Any
    ArrayInt32_1d = Any
    ArrayFloat32_2d = Any
else:
    FabricArrayUInt32 = wp.fabricarray(dtype=wp.uint32)
    FabricArrayMat44d = wp.fabricarray(dtype=wp.mat44d)
    IndexedFabricArrayMat44d = wp.indexedfabricarray(dtype=wp.mat44d)
    ArrayUInt32 = wp.array(ndim=1, dtype=wp.uint32)
    ArrayUInt32_1d = wp.array(dtype=wp.uint32)
    ArrayInt32_1d = wp.array(dtype=wp.int32)
    ArrayFloat32_2d = wp.array(ndim=2, dtype=wp.float32)


@wp.kernel(enable_backward=False)
def set_view_to_fabric_array(fabric_to_view: FabricArrayUInt32, view_to_fabric: ArrayUInt32):
    """Create bidirectional mapping from view indices to fabric indices."""
    fabric_idx = int(wp.tid())
    view_idx = int(fabric_to_view[fabric_idx])
    view_to_fabric[view_idx] = wp.uint32(fabric_idx)


@wp.kernel(enable_backward=False)
def arange_k(a: ArrayUInt32_1d):
    """Fill array with sequential indices."""
    tid = int(wp.tid())
    a[tid] = wp.uint32(tid)


@wp.kernel(enable_backward=False)
def map_view_indices_to_fabric_slots(view_indices: FabricArrayUInt32, fabric_slots: ArrayInt32_1d):
    """Invert a selection's per-prim view-index attribute into a slot lookup table.

    Inverts a permutation: ``view_indices`` holds each selected prim's view-side
    index, and after the launch ``fabric_slots[view_index]`` is that prim's
    fabric-side slot, ready to use as :class:`wp.indexedfabricarray` indices.

    The launch dimension must equal the selection's prim count, and the stored
    view indices must cover ``0..dim-1`` exactly for the table to be complete.
    """
    fabric_slot = int(wp.tid())
    view_index = int(view_indices[fabric_slot])
    fabric_slots[view_index] = fabric_slot


@wp.kernel(enable_backward=False)
def gather_fabric_slots(slots: ArrayInt32_1d, gather_map: ArrayUInt32_1d, out_slots: ArrayInt32_1d):
    """Gather ``slots`` entries through ``gather_map``: ``out_slots[i] = slots[gather_map[i]]``."""
    i = int(wp.tid())
    out_slots[i] = slots[int(gather_map[i])]


@wp.kernel(enable_backward=False)
def decompose_fabric_transformation_matrix_to_warp_arrays(
    fabric_matrices: FabricArrayMat44d,
    array_positions: ArrayFloat32_2d,
    array_orientations: ArrayFloat32_2d,
    array_scales: ArrayFloat32_2d,
    indices: ArrayUInt32,
    mapping: ArrayUInt32,
):
    """Decompose Fabric transformation matrices into position, orientation, and scale arrays.

    This kernel extracts transform components from Fabric's omni:fabric:worldMatrix attribute
    and stores them in separate arrays. It handles the quaternion convention conversion

    Args:
        fabric_matrices: Fabric array containing 4x4 transformation matrices
        array_positions: Output array for positions (N, 3)
        array_orientations: Output array for quaternions in xyzw format (N, 4)
        array_scales: Output array for scales (N, 3)
        indices: View indices to process
        mapping: Mapping from view indices to fabric indices
    """
    # Thread index is the output array index (0, 1, 2, ... for N elements)
    output_index = wp.tid()
    # View index is which prim in the view we're reading from (e.g., 0, 2, 4 from indices=[0,2,4])
    view_index = indices[output_index]
    # Fabric index is where that prim is stored in Fabric
    fabric_index = mapping[view_index]

    # decompose transform matrix
    position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[fabric_index]))
    # extract position - write to sequential output array (check if array has elements)
    if array_positions.shape[0] > 0:
        array_positions[output_index, 0] = position[0]
        array_positions[output_index, 1] = position[1]
        array_positions[output_index, 2] = position[2]
    # extract orientation
    if array_orientations.shape[0] > 0:
        array_orientations[output_index, 0] = rotation[0]  # x
        array_orientations[output_index, 1] = rotation[1]  # y
        array_orientations[output_index, 2] = rotation[2]  # z
        array_orientations[output_index, 3] = rotation[3]  # w
    # extract scale
    if array_scales.shape[0] > 0:
        array_scales[output_index, 0] = scale[0]
        array_scales[output_index, 1] = scale[1]
        array_scales[output_index, 2] = scale[2]


@wp.kernel(enable_backward=False)
def compose_fabric_transformation_matrix_from_warp_arrays(
    fabric_matrices: FabricArrayMat44d,
    array_positions: ArrayFloat32_2d,
    array_orientations: ArrayFloat32_2d,
    array_scales: ArrayFloat32_2d,
    broadcast_positions: bool,
    broadcast_orientations: bool,
    broadcast_scales: bool,
    indices: ArrayUInt32,
    mapping: ArrayUInt32,
):
    """Compose Fabric transformation matrices from position, orientation, and scale arrays.

    This kernel updates Fabric's omni:fabric:worldMatrix attribute from separate component arrays.

    After calling this kernel, IFabricHierarchy.updateWorldXforms() should be called to
    propagate changes through the hierarchy.

    Args:
        fabric_matrices: Fabric array containing 4x4 transformation matrices to update
        array_positions: Input array for positions (N, 3) or None
        array_orientations: Input array for quaternions in xyzw format (N, 4) or None
        array_scales: Input array for scales (N, 3) or None
        broadcast_positions: If True, use first position for all prims
        broadcast_orientations: If True, use first orientation for all prims
        broadcast_scales: If True, use first scale for all prims
        indices: View indices to process
        mapping: Mapping from view indices to fabric indices
    """
    i = wp.tid()
    # resolve fabric index
    fabric_index = mapping[indices[i]]
    # decompose current transform matrix to get existing values
    position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[fabric_index]))
    # update position (check if array has elements, not just if it exists)
    if array_positions.shape[0] > 0:
        if broadcast_positions:
            index = 0
        else:
            index = i
        position[0] = array_positions[index, 0]
        position[1] = array_positions[index, 1]
        position[2] = array_positions[index, 2]
    # update orientation (convert from wxyz to xyzw for Warp)
    if array_orientations.shape[0] > 0:
        if broadcast_orientations:
            index = 0
        else:
            index = i
        rotation[0] = array_orientations[index, 0]  # x
        rotation[1] = array_orientations[index, 1]  # y
        rotation[2] = array_orientations[index, 2]  # z
        rotation[3] = array_orientations[index, 3]  # w
    # update scale
    if array_scales.shape[0] > 0:
        if broadcast_scales:
            index = 0
        else:
            index = i
        scale[0] = array_scales[index, 0]
        scale[1] = array_scales[index, 1]
        scale[2] = array_scales[index, 2]
    # set transform matrix (need transpose for column-major ordering)
    # Using transform_compose as wp.matrix() is deprecated
    fabric_matrices[fabric_index] = wp.mat44d(  # type: ignore[arg-type]
        wp.transpose(wp.transform_compose(position, rotation, scale))  # type: ignore[arg-type]
    )


@wp.kernel(enable_backward=False)
def decompose_indexed_fabric_transforms(
    fabric_matrices: IndexedFabricArrayMat44d,
    array_positions: ArrayFloat32_2d,
    array_orientations: ArrayFloat32_2d,
    array_scales: ArrayFloat32_2d,
    indices: ArrayUInt32,
):
    """Decompose indexed Fabric transformation matrices into position, orientation, and scale.

    Like :func:`decompose_fabric_transformation_matrix_to_warp_arrays` but operates on a
    :class:`wp.indexedfabricarray` that already encodes the view-to-fabric mapping, removing
    the need for a separate ``mapping`` array.

    Args:
        fabric_matrices: Indexed fabric array containing 4x4 transformation matrices.
        array_positions: Output array for positions [m], shape (N, 3).
        array_orientations: Output array for quaternions in xyzw format, shape (N, 4).
        array_scales: Output array for scales, shape (N, 3).
        indices: View indices to process (subset selection).
    """
    output_index = wp.tid()
    view_index = indices[output_index]

    position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[view_index]))

    if array_positions.shape[0] > 0:
        array_positions[output_index, 0] = position[0]
        array_positions[output_index, 1] = position[1]
        array_positions[output_index, 2] = position[2]
    if array_orientations.shape[0] > 0:
        array_orientations[output_index, 0] = rotation[0]
        array_orientations[output_index, 1] = rotation[1]
        array_orientations[output_index, 2] = rotation[2]
        array_orientations[output_index, 3] = rotation[3]
    if array_scales.shape[0] > 0:
        array_scales[output_index, 0] = scale[0]
        array_scales[output_index, 1] = scale[1]
        array_scales[output_index, 2] = scale[2]


@wp.kernel(enable_backward=False)
def compose_indexed_fabric_transforms(
    fabric_matrices: IndexedFabricArrayMat44d,
    array_positions: ArrayFloat32_2d,
    array_orientations: ArrayFloat32_2d,
    array_scales: ArrayFloat32_2d,
    broadcast_positions: bool,
    broadcast_orientations: bool,
    broadcast_scales: bool,
    indices: ArrayUInt32,
):
    """Compose indexed Fabric transformation matrices from position, orientation, and scale.

    Like :func:`compose_fabric_transformation_matrix_from_warp_arrays` but operates on a
    :class:`wp.indexedfabricarray` that already encodes the view-to-fabric mapping, removing
    the need for a separate ``mapping`` array.

    Args:
        fabric_matrices: Indexed fabric array containing 4x4 transformation matrices to update.
        array_positions: Input array for positions [m], shape (N, 3).
        array_orientations: Input array for quaternions in xyzw format, shape (N, 4).
        array_scales: Input array for scales, shape (N, 3).
        broadcast_positions: If True, use first position for all prims.
        broadcast_orientations: If True, use first orientation for all prims.
        broadcast_scales: If True, use first scale for all prims.
        indices: View indices to process (subset selection).
    """
    i = wp.tid()
    view_index = indices[i]
    position, rotation, scale = _decompose_transformation_matrix(wp.mat44f(fabric_matrices[view_index]))

    if array_positions.shape[0] > 0:
        if broadcast_positions:
            index = 0
        else:
            index = i
        position[0] = array_positions[index, 0]
        position[1] = array_positions[index, 1]
        position[2] = array_positions[index, 2]
    if array_orientations.shape[0] > 0:
        if broadcast_orientations:
            index = 0
        else:
            index = i
        rotation[0] = array_orientations[index, 0]
        rotation[1] = array_orientations[index, 1]
        rotation[2] = array_orientations[index, 2]
        rotation[3] = array_orientations[index, 3]
    if array_scales.shape[0] > 0:
        if broadcast_scales:
            index = 0
        else:
            index = i
        scale[0] = array_scales[index, 0]
        scale[1] = array_scales[index, 1]
        scale[2] = array_scales[index, 2]

    fabric_matrices[view_index] = wp.mat44d(  # type: ignore[arg-type]
        wp.transpose(wp.transform_compose(position, rotation, scale))  # type: ignore[arg-type]
    )


@wp.kernel(enable_backward=False)
def update_indexed_local_matrix_from_world(
    child_world_matrices: IndexedFabricArrayMat44d,
    parent_world_matrices: IndexedFabricArrayMat44d,
    child_local_matrices: IndexedFabricArrayMat44d,
    indices: ArrayUInt32,
):
    """Recompute child localMatrix from (parent worldMatrix, child worldMatrix).

    Computes ``child_local = inv(parent_world) * child_world`` per prim and writes the
    result back to the child's :data:`omni:fabric:localMatrix` so that subsequent
    ``get_local_poses`` calls see consistent values after a world-pose write.

    All three indexed arrays are expected to be indexed by the same per-view indices
    (i.e. ``view_to_child_fabric``, ``view_to_parent_fabric``, ``view_to_child_fabric``)
    so the kernel only needs the view-side indices.

    Storage convention: Fabric matrices are stored as the transpose of the standard
    column-major math convention.  Math is ``local = inv(parent) * world``; under
    the transpose identity ``(A * B)^T = B^T * A^T`` (and ``inv(A^T) = inv(A)^T``)
    that is equivalent to storage-side ``local^T = world^T * inv(parent^T)``, so we
    can compute it directly on the stored matrices without explicit transposes.

    Args:
        child_world_matrices: Indexed fabric array of child world matrices (read).
        parent_world_matrices: Indexed fabric array of parent world matrices (read).
        child_local_matrices: Indexed fabric array of child local matrices (written).
        indices: View indices to process.
    """
    i = wp.tid()
    view_index = indices[i]
    child_world = wp.mat44f(child_world_matrices[view_index])
    parent_world = wp.mat44f(parent_world_matrices[view_index])
    child_local_matrices[view_index] = wp.mat44d(  # type: ignore[arg-type]
        child_world * wp.inverse(parent_world)
    )


@wp.kernel(enable_backward=False)
def update_indexed_world_matrix_from_local(
    child_local_matrices: IndexedFabricArrayMat44d,
    parent_world_matrices: IndexedFabricArrayMat44d,
    child_world_matrices: IndexedFabricArrayMat44d,
    indices: ArrayUInt32,
):
    """Recompute child worldMatrix from (parent worldMatrix, child localMatrix).

    Computes ``child_world = parent_world * child_local`` per prim and writes the
    result back to the child's :data:`omni:fabric:worldMatrix`.  Used after a
    ``set_local_poses`` write so that subsequent ``get_world_poses`` calls see
    consistent values.  Mirror of :func:`update_indexed_local_matrix_from_world`.

    Args:
        child_local_matrices: Indexed fabric array of child local matrices (read).
        parent_world_matrices: Indexed fabric array of parent world matrices (read).
        child_world_matrices: Indexed fabric array of child world matrices (written).
        indices: View indices to process.

    Storage convention: same as :func:`update_indexed_local_matrix_from_world`.
    Math is ``world = parent * local``; under the transpose identity that becomes
    storage-side ``world^T = local^T * parent^T``, no explicit transposes needed.
    """
    i = wp.tid()
    view_index = indices[i]
    child_local = wp.mat44f(child_local_matrices[view_index])
    parent_world = wp.mat44f(parent_world_matrices[view_index])
    child_world_matrices[view_index] = wp.mat44d(  # type: ignore[arg-type]
        child_local * parent_world
    )


@wp.func
def _decompose_transformation_matrix(m: Any):  # -> tuple[wp.vec3f, wp.quatf, wp.vec3f]
    """Decompose a 4x4 transformation matrix into position, orientation, and scale.

    Args:
        m: 4x4 transformation matrix

    Returns:
        Tuple of (position, rotation_quaternion, scale)
    """
    # extract position from translation column
    position = wp.vec3f(m[3, 0], m[3, 1], m[3, 2])
    # extract rotation matrix components
    r00, r01, r02 = m[0, 0], m[0, 1], m[0, 2]
    r10, r11, r12 = m[1, 0], m[1, 1], m[1, 2]
    r20, r21, r22 = m[2, 0], m[2, 1], m[2, 2]
    # get scale magnitudes from column vectors
    sx = wp.sqrt(r00 * r00 + r01 * r01 + r02 * r02)
    sy = wp.sqrt(r10 * r10 + r11 * r11 + r12 * r12)
    sz = wp.sqrt(r20 * r20 + r21 * r21 + r22 * r22)
    # normalize rotation matrix components by scale
    if sx != 0.0:
        r00 /= sx
        r01 /= sx
        r02 /= sx
    if sy != 0.0:
        r10 /= sy
        r11 /= sy
        r12 /= sy
    if sz != 0.0:
        r20 /= sz
        r21 /= sz
        r22 /= sz
    # extract rotation quaternion from normalized rotation matrix
    rotation = wp.quat_from_matrix(  # type: ignore[arg-type]
        wp.transpose(wp.mat33f(r00, r01, r02, r10, r11, r12, r20, r21, r22))  # type: ignore[arg-type]
    )
    # extract scale
    scale = wp.vec3f(sx, sy, sz)
    return position, rotation, scale
