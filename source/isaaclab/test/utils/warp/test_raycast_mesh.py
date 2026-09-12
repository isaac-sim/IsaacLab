# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from isaaclab.utils.warp.ops import convert_to_warp_mesh, raycast_mesh

pytestmark = pytest.mark.unit


def test_raycast_mesh_auxiliary_outputs_match_unbatched_ray_shape():
    """Distance and face-id outputs should use the leading shape of documented ``(N, 3)`` ray inputs."""
    mesh = convert_to_warp_mesh(
        points=np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        indices=np.array([[0, 1, 2]], dtype=np.int32),
        device="cpu",
    )
    ray_starts = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=torch.float32)
    ray_directions = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]], dtype=torch.float32)

    ray_hits, ray_distance, ray_normal, ray_face_id = raycast_mesh(
        ray_starts,
        ray_directions,
        mesh,
        return_distance=True,
        return_normal=True,
        return_face_id=True,
    )

    assert ray_hits.shape == (2, 3)
    assert ray_distance.shape == (2,)
    assert ray_normal.shape == (2, 3)
    assert ray_face_id.shape == (2,)
    torch.testing.assert_close(ray_distance, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(ray_face_id, torch.tensor([0, 0], dtype=torch.int32))
