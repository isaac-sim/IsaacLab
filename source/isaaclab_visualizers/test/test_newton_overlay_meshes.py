# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton's line and arrow overlays become thin meshes for the OVRTX view."""

import numpy as np
import pytest
import warp as wp
from isaaclab_visualizers.newton.newton_overlay_meshes import (
    DEFAULT_COLOR,
    OverlayMeshPublisher,
    build_segment_meshes,
    color_classes,
    segment_indices,
)

pytestmark = [pytest.mark.unit]


class _Renderer:
    def __init__(self):
        self.calls = []

    def log_mesh(self, path, points, indices, **kwargs):
        self.calls.append((path, points.numpy().copy(), indices.numpy().copy(), kwargs))


def _vec3s(values):
    return wp.array(np.asarray(values, np.float32), dtype=wp.vec3, device="cpu")


def test_segment_mesh_spans_the_segment_and_other_classes_collapse():
    starts, ends = _vec3s([[0, 0, 0], [1, 1, 1]]), _vec3s([[0, 0, 1], [1, 1, 2]])
    classes = wp.array([0, 1], dtype=wp.int32, device="cpu")
    points = wp.zeros(26, dtype=wp.vec3, device="cpu")
    wp.launch(build_segment_meshes, 2, [starts, ends, classes, 0, 0.01, 0.0, 0.0, points], device="cpu")
    vertices = points.numpy()
    assert vertices[:13, 2].min() == pytest.approx(0.0) and vertices[:13, 2].max() == pytest.approx(1.0)
    assert np.abs(vertices[:4, :2]).max() == pytest.approx(0.01)
    assert np.all(vertices[13:] == [1, 1, 1])  # the other class's segment is a point at its start
    assert segment_indices(2, "cpu").numpy().max() == 25


def test_arrow_head_ends_at_the_tip():
    points = wp.zeros(13, dtype=wp.vec3, device="cpu")
    classes = wp.zeros(1, dtype=wp.int32, device="cpu")
    wp.launch(
        build_segment_meshes, 1, [_vec3s([[0, 0, 0]]), _vec3s([[2, 0, 0]]), classes, 0, 0.01, 0.03, 0.2, points],
        device="cpu",
    )  # fmt: skip
    vertices = points.numpy()
    assert vertices[12] == pytest.approx([2, 0, 0])
    assert vertices[8:12, 0] == pytest.approx([1.8] * 4)


def test_color_classes_group_per_segment_colors():
    assert color_classes((1, 0, 0), 3) == (None, [(1.0, 0.0, 0.0)])
    assert color_classes(None, 3) == (None, [DEFAULT_COLOR])
    classes, colors = color_classes(_vec3s([[1, 0, 0], [0, 1, 0], [1, 0, 0]]), 3)
    assert [colors[i] for i in classes] == [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]


def test_publisher_shows_recorded_batches_and_hides_the_rest():
    renderer, publisher = _Renderer(), OverlayMeshPublisher("Viewer_1")
    starts, ends = _vec3s([[0, 0, 0]]), _vec3s([[0, 0, 1]])
    publisher.record("/model/joints", starts, ends, (0.0, 1.0, 0.0), 0.01, False, arrow=False)
    publisher.publish([renderer])
    assert [(call[0], call[3]["hidden"], call[3]["color"]) for call in renderer.calls] == [
        ("Viewer_1/overlay/model_joints_1_0", False, (0.0, 1.0, 0.0))
    ]
    renderer.calls.clear()
    publisher.publish([renderer])  # nothing recorded this frame
    assert [(call[0], call[3]["hidden"]) for call in renderer.calls] == [("Viewer_1/overlay/model_joints_1_0", True)]
    renderer.calls.clear()
    publisher.hide_all([renderer])
    assert not renderer.calls
