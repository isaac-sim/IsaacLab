# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Watertight SN2201 contact geometry measured from the pinned SimReady CAD."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh

from isaaclab.utils.math import quat_apply

from .assets import GB300_SIMREADY_EXTERNAL_USD_SHA256

if TYPE_CHECKING:
    import newton


SOCKET_CONTACT_BOUNDS_M = ((-0.009, -0.01275, -0.0075), (0.009, 0.001, 0.0065))
"""Detailed collision-proxy extent relative to the registered socket origin [m]."""
SOCKET_APERTURE_PROFILE_M = (
    (-0.006135, -0.004990),
    (0.005775, -0.004990),
    (0.005775, 0.002735),
    (0.002970, 0.002735),
    (0.002970, 0.004980),
    (-0.003330, 0.004980),
    (-0.003330, 0.002735),
    (-0.006135, 0.002735),
)
"""Measured cavity outline in the socket's horizontal/vertical plane [m]."""
SOCKET_KEEPER_X_INTERVALS_M = ((-0.003330, -0.002210), (0.001850, 0.002970))
"""Horizontal extents of the two native latch-retaining shoulders [m]."""
SOCKET_KEEPER_DEPTH_INTERVAL_M = (-0.012250, -0.010950)
"""Shoulder front and rear coordinates along the socket insertion axis [m]."""
SOCKET_KEEPER_UNDERSIDE_M = 0.003730
"""Measured shoulder underside height in the socket frame [m]."""
SOCKET_GUIDE_ROOF_M = (0.002010, 0.002060)
"""Outer and inner underside heights of the two sloped cavity guides [m]."""
SOCKET_GUIDE_REAR_M = -0.004200
"""Rear extent of the measured front guides along the insertion axis [m]."""
SOCKET_REAR_GUIDE_ROOF_M = 0.000960
"""Measured underside of the rear guides beside the central keyhole [m]."""
SOCKET_ENTRY_LIP_DEPTH_INTERVAL_M = (-0.011350, -0.010650)
"""Front and rear coordinates of the measured lower entry support [m]."""
SOCKET_ENTRY_LIP_TOP_M = -0.004790
"""Measured top of the lower entry support, above the ordinary cavity floor [m]."""
LATCH_BARB_REAR_FACES_M = (
    (
        (-0.003045795, 0.005522263, -0.026458728),
        (-0.001639293, 0.005522263, -0.026458728),
        (-0.001639293, 0.006820000, -0.026458728),
        (-0.003045795, 0.006820000, -0.026458728),
    ),
    (
        (0.001639293, 0.005522263, -0.026458728),
        (0.003045795, 0.005522263, -0.026458728),
        (0.003045795, 0.006820000, -0.026458728),
        (0.001639293, 0.006820000, -0.026458728),
    ),
)
"""Fitted CAD barb rear-face corners in the latch's rest head-center frame [m]."""
LATCH_CONTACT_BOUNDS_M = (
    (-0.003045800, 0.004677866, -0.031008737),
    (0.003045800, 0.009662120, -0.017108725),
)
"""Native clip collision bounds in its rest head-center frame [m]."""


def latch_keeper_exit_margin(latch_pose_in_socket: torch.Tensor) -> torch.Tensor:
    """Return conservative axial clip overlap with the keeper front plane [m].

    Args:
        latch_pose_in_socket: Measured latch-body pose [m, XYZW], shape (..., 7).

    Returns:
        Maximum insertion-axis overlap [m], shape (...). A negative value proves
        the entire rotated clip bounding box is outside both keepers. This remains
        conservative under connector tilt and clip deflection.
    """
    corners = torch.cartesian_prod(*(latch_pose_in_socket.new_tensor(axis) for axis in zip(*LATCH_CONTACT_BOUNDS_M)))
    corners = quat_apply(latch_pose_in_socket[..., None, 3:], corners) + latch_pose_in_socket[..., None, :3]
    return corners[..., 1].amax(dim=-1) - SOCKET_KEEPER_DEPTH_INTERVAL_M[0]


def latch_geometry_margins(latch_pose_in_socket: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure capture, release and keeper exit from one socket-relative latch pose.

    Args:
        latch_pose_in_socket: Measured latch-body pose [m, XYZW], shape (..., 7).

    Returns:
        Capture margins [m], shape (..., 3), release margin [m], shape (...), and
        keeper-exit margin [m], shape (...). Release is the greatest vertical overlap
        across both barbs; a negative value proves both are below their keepers.
    """
    barb_margins = _latch_barb_engagement_margins(latch_pose_in_socket)
    return (
        barb_margins.amin(dim=-2),
        barb_margins[..., 2].amax(dim=-1),
        latch_keeper_exit_margin(latch_pose_in_socket),
    )


def _latch_barb_engagement_margins(latch_pose_in_socket: torch.Tensor) -> torch.Tensor:
    """Measure each barb independently before reducing capture or release margins."""
    corners = latch_pose_in_socket.new_tensor(LATCH_BARB_REAR_FACES_M)
    corners = quat_apply(latch_pose_in_socket[..., None, None, 3:], corners)
    corners += latch_pose_in_socket[..., None, None, :3]
    lower, upper = corners.amin(dim=-2), corners.amax(dim=-2)
    keepers = latch_pose_in_socket.new_tensor(SOCKET_KEEPER_X_INTERVALS_M)
    axial = lower[..., 1] - SOCKET_KEEPER_DEPTH_INTERVAL_M[1]
    lateral = torch.minimum(upper[..., 0], keepers[:, 1]) - torch.maximum(lower[..., 0], keepers[:, 0])
    vertical = upper[..., 2].clamp(max=SOCKET_CONTACT_BOUNDS_M[1][2]) - lower[..., 2].clamp(
        min=SOCKET_KEEPER_UNDERSIDE_M
    )
    return torch.stack((axial, lateral, vertical), dim=-1)


@lru_cache(maxsize=1)
def local_socket_mesh() -> newton.Mesh:
    """Return one CAD-measured socket SDF in the task's socket frame [m].

    The visual CAD contains duplicate, open and non-manifold surfaces; baking
    it directly falsely fills the cavity. This closed collision proxy retains
    its measured keyhole profile, entry chamfer, lower entry support, latch
    shoulders and both front and rear guides. Flexible electrical contacts and
    the deeper lower rear step remain simplified.
    Both active jacks share this immutable mesh and its cached 0.05 mm SDF.
    The checked-in measurements do not require the render-only CAD payload.
    """
    import newton

    aperture = np.asarray(SOCKET_APERTURE_PROFILE_M)
    left, right = SOCKET_KEEPER_X_INTERVALS_M
    keeper_front, keeper_rear = SOCKET_KEEPER_DEPTH_INTERVAL_M
    lip_front, lip_rear = SOCKET_ENTRY_LIP_DEPTH_INTERVAL_M
    # Paired sections close the keeper and floor steps without overlapping solids.
    sections = (
        (-0.01275, 0.0006, False, True, False),
        (keeper_front, 0.0004, False, True, False),
        (keeper_front, 0.0004, True, True, False),
        (-0.01175, 0.0002, True, True, False),
        (lip_front, 0.0, True, True, True),
        (-0.01125, 0.0, True, True, True),
        (keeper_rear, 0.0, True, True, True),
        (keeper_rear, 0.0, False, True, True),
        (lip_rear, 0.0, False, True, True),
        (lip_rear, 0.0, False, True, False),
        (SOCKET_GUIDE_REAR_M, 0.0, False, True, False),
        (SOCKET_GUIDE_REAR_M, 0.0, False, False, False),
        (-0.00170, 0.0, False, False, False),
    )
    vertices = []
    for y, expansion, has_keeper, has_front_guide, has_entry_lip in sections:
        profile = aperture.copy()
        profile[:, 0] += np.where(profile[:, 0] < -0.00018, -expansion, expansion)
        profile[:, 1] += np.where(profile[:, 1] < 0.0, -expansion, expansion)
        if has_entry_lip:
            profile[:2, 1] = SOCKET_ENTRY_LIP_TOP_M
        # Matching wall vertices close the guide and floor steps without T-junctions.
        chamfer = 0.6 * max(0.0, keeper_front - y)
        front_guide_inner = SOCKET_GUIDE_ROOF_M[1] + chamfer
        guide_roof = SOCKET_GUIDE_ROOF_M if has_front_guide else (SOCKET_REAR_GUIDE_ROOF_M,) * 2
        # The native guide rises 0.3 mm across the 0.5 mm mouth chamfer.
        guide_outer = guide_roof[0] + chamfer
        profile[[2, 7], 1] = guide_outer
        profile[[3, 6], 1] = guide_roof[1] + chamfer
        roof = profile[4, 1]
        underside = SOCKET_KEEPER_UNDERSIDE_M if has_keeper else roof
        outline = (
            *profile[:2],
            (profile[1, 0], SOCKET_ENTRY_LIP_TOP_M),
            (profile[2, 0], min(profile[2, 1], SOCKET_REAR_GUIDE_ROOF_M + chamfer)),
            *profile[2:4],
            (profile[3, 0], front_guide_inner),
            profile[4],
            (right[1], roof),
            (right[1], underside),
            (right[0], underside),
            (right[0], roof),
            (left[1], roof),
            (left[1], underside),
            (left[0], underside),
            (left[0], roof),
            profile[5],
            (profile[6, 0], front_guide_inner),
            *profile[6:],
            (profile[7, 0], min(profile[7, 1], SOCKET_REAR_GUIDE_ROOF_M + chamfer)),
            (profile[0, 0], SOCKET_ENTRY_LIP_TOP_M),
        )
        vertices.extend((x, y, z) for x, z in outline)
    outer = np.array(
        (
            (-0.009, -0.0075),
            (0.009, -0.0075),
            (0.009, 0.0035),
            (0.009, 0.0065),
            (0.003, 0.0065),
            (-0.003, 0.0065),
            (-0.009, 0.0065),
            (-0.009, 0.0035),
        )
    )
    outer = outer[[0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 7, 7, 0]]
    ring_size = len(outline)
    front_outer, back_outer = len(vertices), len(vertices) + ring_size
    for y in (-0.01275, 0.001):
        vertices.extend((x, y, z) for x, z in outer)
    faces = []
    # Join the front annulus, chamfer, cavity walls and cabinet-facing exterior.
    rings = [(ring_size * layer, ring_size * (layer + 1)) for layer in range(len(sections) - 1)]
    rings.extend(((front_outer, 0), (back_outer, front_outer)))
    for first, second in rings:
        for index in range(ring_size):
            following = (index + 1) % ring_size
            a, b, c, d = first + index, first + following, second + following, second + index
            faces.extend(((a, b, c), (a, c, d)))
    for ring, y in ((ring_size * (len(sections) - 1), -0.00170), (back_outer, 0.001)):
        center = len(vertices)
        vertices.append((0.0, y, 0.0))
        faces.extend((center, ring + index, ring + (index + 1) % ring_size) for index in range(ring_size))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(mesh)
    if not mesh.is_watertight or not mesh.is_volume:
        raise ValueError("The measured SN2201 socket proxy must define a closed collision volume.")
    contact_mesh = newton.Mesh(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32).reshape(-1),
        compute_inertia=False,
    )
    contact_mesh.build_sdf(
        target_voxel_size=0.00005,
        narrow_band_range=(-0.002, 0.002),
        margin=0.001,
        sign_method="parity",
        cache_dir=Path.home() / ".cache/isaaclab/gb300-socket-sdf" / GB300_SIMREADY_EXTERNAL_USD_SHA256,
    )
    return contact_mesh
