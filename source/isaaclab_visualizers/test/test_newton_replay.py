# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for decoding ``.viser`` recordings for replay in the Newton GL window."""

import socket

import numpy as np
import pytest
import viser
from isaaclab_visualizers.newton.newton_replay import list_recordings, load_viser_recording


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def recording_path(tmp_path):
    """A recording written by Viser itself: two instances of a triangle moving over three frames."""
    server = viser.ViserServer(host="127.0.0.1", port=_free_port(), verbose=False)
    try:
        serializer = server.get_scene_serializer()
        handle = server.scene.add_batched_meshes_simple(
            "/model/shapes/shape_0",
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
            faces=np.array([[0, 1, 2]], np.uint32),
            batched_wxyzs=np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32),
            batched_positions=np.zeros((2, 3), np.float32),
            batched_colors=np.array([[251, 165, 69], [10, 20, 30]], np.uint8),
        )
        for step in (1, 2):
            serializer.insert_sleep(0.1)
            handle.batched_positions = np.full((2, 3), float(step), np.float32)
        path = tmp_path / "scene" / "clip.viser"
        path.parent.mkdir()
        path.write_bytes(serializer.serialize())
    finally:
        server.stop()
    return path


def test_loader_reads_frames_poses_and_colors(recording_path):
    recording = load_viser_recording(str(recording_path))

    np.testing.assert_allclose(recording.times, [0.0, 0.1, 0.2])
    assert recording.duration == pytest.approx(0.2)
    (shape,) = recording.shapes
    assert shape.positions.shape == (3, 2, 3)
    np.testing.assert_allclose(shape.positions[:, 0, 0], [0.0, 1.0, 2.0])
    np.testing.assert_allclose(shape.xforms[1, 0], [1, 1, 1, 0, 0, 0, 1])  # Position, then xyzw quaternion.
    np.testing.assert_allclose(shape.colors[1], np.array([10, 20, 30]) / 255.0, atol=1e-6)
    assert shape.indices.tolist() == [0, 1, 2]


def test_frame_lookup_holds_the_last_frame(recording_path):
    recording = load_viser_recording(str(recording_path))

    assert [recording.frame_at(t) for t in (-1.0, 0.0, 0.05, 0.1, 0.19, 5.0)] == [0, 0, 0, 1, 1, 2]


def test_listing_finds_nested_files_newest_first(tmp_path):
    older, newer = tmp_path / "a" / "old.viser", tmp_path / "new.viser"
    older.parent.mkdir()
    older.write_bytes(b"x")
    newer.write_bytes(b"x")
    older.touch()
    newer.touch()
    (tmp_path / "ignored.txt").write_text("x")

    assert list_recordings(str(tmp_path)) == [newer, older]
    assert list_recordings(str(tmp_path / "missing")) == []


def test_static_shapes_are_detected_from_their_poses(recording_path):
    (shape,) = load_viser_recording(str(recording_path)).shapes
    assert not shape.static  # The fixture moves its instances.

    shape.xforms[:] = shape.xforms[:1]
    assert shape.static


def test_bounds_cover_the_first_frame_positions(recording_path):
    low, high = load_viser_recording(str(recording_path)).bounds()

    np.testing.assert_allclose(low, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(high, [0.0, 0.0, 0.0])  # Both instances start at the origin.
