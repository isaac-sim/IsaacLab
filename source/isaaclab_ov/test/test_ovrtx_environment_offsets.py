# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment offsets translate the published OVRTX body poses of whole environments."""

import numpy as np
import warp as wp
from isaaclab_ov.renderers.ovrtx_renderer_kernels import offset_environment_bodies_kernel


def test_each_body_moves_by_the_offset_of_its_environment():
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 3, :3] = [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]  # row-vector layout: translation in the last row
    transforms = wp.array(poses, dtype=wp.mat44d, device="cpu")
    body_envs = wp.array([0, 1, 1, 2], dtype=wp.int32, device="cpu")  # env 2 stands for "outside any environment"
    offsets = wp.array([[0, 0, 0], [10, 0, 0], [0, 0, 0]], dtype=wp.vec3, device="cpu")
    out = wp.empty(4, dtype=wp.mat44d, device="cpu")

    wp.launch(offset_environment_bodies_kernel, 4, [transforms, body_envs, offsets, out], device="cpu")

    result = out.numpy()
    assert result[:, 3, :3].tolist() == [[1, 1, 1], [12, 2, 2], [13, 3, 3], [4, 4, 4]]
    assert np.array_equal(result[:, :3, :3], poses[:, :3, :3])
    assert np.array_equal(transforms.numpy(), poses)  # the published buffer is shared and stays untouched
