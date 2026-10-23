# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for hiding the bodies and markers of cloned environments in the OVRTX scene."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("ovrtx")

from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: E402
from isaaclab_ov.renderers.ovrtx_renderer_kernels import hide_environment_bodies_kernel  # noqa: E402


def test_hidden_environments_drop_only_their_own_bodies():
    # Row-vector matrices keep the translation in the last row; bodies 0-1 are in env 0, 2 in env 1, 3 in no env.
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, 3, :3] = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    body_envs = wp.array([0, 0, 1, 2], dtype=wp.int32, device="cpu")
    hidden = wp.array([0, 1, 0], dtype=wp.uint8, device="cpu")  # Env 1 is hidden; the last entry is "no env".
    out = wp.empty(4, dtype=wp.mat44d, device="cpu")

    wp.launch(
        hide_environment_bodies_kernel,
        4,
        [wp.array(poses, dtype=wp.mat44d, device="cpu"), body_envs, hidden, wp.float64(100.0), out],
        device="cpu",
    )

    result = out.numpy()
    np.testing.assert_allclose(result[:, 3, 2], [3, 6, 9 - 100, 12])
    np.testing.assert_allclose(result[:, 3, :2], poses[:, 3, :2])


def _renderer(visible):
    renderer = object.__new__(OVRTXRenderer)
    renderer._env_paths = ["/World/envs/env_0", "/World/envs/env_1", "/World/envs/env_2"]
    renderer._visible_envs = None if visible is None else np.asarray(visible, dtype=bool)
    return renderer


def test_markers_keep_only_the_rows_of_visible_environments():
    marker = SimpleNamespace(
        translations=torch.arange(0.0, 12.0, 2.0)[:, None].repeat(1, 3),  # 6 markers: 2 per env
        orientations=None,
        scales=None,
        marker_indices=torch.tensor([0, 1, 0, 1, 0, 1]),
    )

    translations, orientations, scales, indices = _renderer([True, False, True])._visible_marker_rows(marker)

    assert translations[:, 0].tolist() == [0.0, 2.0, 8.0, 10.0]
    assert orientations is None and scales is None
    assert indices.tolist() == [0, 1, 0, 1]


def test_markers_are_untouched_when_all_environments_show_or_the_batch_is_global():
    marker = SimpleNamespace(
        translations=torch.zeros(4, 3), orientations=None, scales=None, marker_indices=None
    )  # 4 markers do not divide by 3 environments, so they are global.

    assert _renderer([True, False, True])._visible_marker_rows(marker)[0] is marker.translations
    assert _renderer([True, True, True])._visible_marker_rows(marker)[0] is marker.translations
    assert _renderer(None)._visible_marker_rows(marker)[0] is marker.translations
