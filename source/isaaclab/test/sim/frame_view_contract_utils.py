# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared FrameView contract tests.

This module defines the invariants that **every** FrameView backend
(USD, Fabric, Newton) must satisfy.  Backend test files import these tests
via ``from frame_view_contract_utils import *`` and provide a
``view_factory`` pytest fixture that builds the backend-specific scene.

The factory signature is::

    def view_factory() -> Callable[[int, str], ViewBundle]: ...

Where ``ViewBundle`` is a :class:`NamedTuple`::

    class ViewBundle(NamedTuple):
        view: BaseFrameView
        get_parent_pos: Callable[[int, str], torch.Tensor]
        set_parent_pos: Callable[[torch.Tensor, int], None]
        teardown: Callable[[], None]

- ``view``: The FrameView under test.  Must track child prims at
  :data:`CHILD_OFFSET` under parent prims/bodies.
- ``get_parent_pos(n, device)``: Read the parent prim/body positions.
- ``set_parent_pos(positions, n)``: Write the parent prim/body positions.
- ``teardown()``: Cleanup (close context, clear stage, etc.).

Tolerance policy:
    - Indexed reads (exact copy): ``atol=0``
    - Composition / decomposition through float32 transforms: ``atol=ATOL``
    - Parent position identity checks (should be untouched): ``atol=0``
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import pytest
import torch
import warp as wp

from isaaclab.utils.warp import ProxyArray

CHILD_OFFSET = (0.1, 0.0, 0.05)
"""Local offset of the child prim from its parent, shared by all backend fixtures."""

ATOL = 1e-5
"""Default absolute tolerance for float32 transform composition."""


class ViewBundle(NamedTuple):
    """Return type of the ``view_factory`` fixture."""

    view: object
    get_parent_pos: Callable
    set_parent_pos: Callable
    teardown: Callable


def _t(a):
    """Convert a wp.array or ProxyArray return to a torch.Tensor (pass-through otherwise)."""
    if isinstance(a, ProxyArray):
        return a.torch
    return wp.to_torch(a) if isinstance(a, wp.array) else a


def _wp_vec3f(data, device="cpu"):
    return wp.array([wp.vec3f(*row) for row in data], dtype=wp.vec3f, device=device)


def _wp_vec4f(data, device="cpu"):
    return wp.array([wp.vec4f(*row) for row in data], dtype=wp.vec4f, device=device)


# ==================================================================
# Contract: Getters
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_pose_equals_parent_plus_offset(device, view_factory):
    """world_pose == parent_pos + local offset (identity parent orientation)."""
    bundle = view_factory(num_envs=4, device=device)
    try:
        child_pos = _t(bundle.view.get_world_poses()[0])
        parent_pos = bundle.get_parent_pos(4, device)
        offset = torch.tensor(CHILD_OFFSET, device=device)

        torch.testing.assert_close(child_pos, parent_pos + offset.unsqueeze(0), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_pose_equals_structural_offset(device, view_factory):
    """local_pose == the authored offset (0.1, 0, 0.05) for every prim."""
    bundle = view_factory(num_envs=4, device=device)
    try:
        local_pos, local_quat = bundle.view.get_local_poses()
        expected_pos = torch.tensor(CHILD_OFFSET, device=device).expand(4, -1)
        expected_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).expand(4, -1)

        torch.testing.assert_close(_t(local_pos), expected_pos, atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(local_quat), expected_quat, atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_differs_from_world(device, view_factory):
    """local != world when parent is not at the origin.

    Asserts |world - local| > 0.5 to catch any implementation that returns
    world as local.  The parent is offset from the origin so the z-component
    alone provides > 0.5 difference.
    """
    bundle = view_factory(num_envs=2, device=device)
    try:
        world_pos = _t(bundle.view.get_world_poses()[0])
        local_pos = _t(bundle.view.get_local_poses()[0])

        diff = (world_pos - local_pos).abs().max().item()
        assert diff > 0.5, (
            f"Expected |world - local| > 0.5, got {diff:.4f}. world={world_pos.tolist()}, local={local_pos.tolist()}"
        )
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_local_stable_after_parent_move(device, view_factory):
    """Moving the parent changes world but NOT local."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        local_before = _t(bundle.view.get_local_poses()[0]).clone()
        bundle.set_parent_pos(torch.tensor([[99.0, 0.0, 0.0], [0.0, 99.0, 0.0]], device=device), 2)
        local_after = _t(bundle.view.get_local_poses()[0])

        torch.testing.assert_close(local_after, local_before, atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_world_tracks_parent_move(device, view_factory):
    """Moving the parent shifts world poses by the same amount."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        new_parent_pos = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]], device=device)
        bundle.set_parent_pos(new_parent_pos, 2)

        child_pos = _t(bundle.view.get_world_poses()[0])
        offset = torch.tensor(CHILD_OFFSET, device=device)

        torch.testing.assert_close(child_pos, new_parent_pos + offset.unsqueeze(0), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_indexed_get_returns_correct_subset(device, view_factory):
    """Indexed get (out-of-order) returns exact copies for both world and local."""
    bundle = view_factory(num_envs=5, device=device)
    try:
        all_world = _t(bundle.view.get_world_poses()[0])
        all_local = _t(bundle.view.get_local_poses()[0])

        indices_list = [4, 1, 3]
        indices = wp.array(indices_list, dtype=wp.int32, device=device)
        sub_world = _t(bundle.view.get_world_poses(indices)[0])
        sub_local = _t(bundle.view.get_local_poses(indices)[0])

        for out_i, view_i in enumerate(indices_list):
            torch.testing.assert_close(sub_world[out_i], all_world[view_i], atol=0, rtol=0)
            torch.testing.assert_close(sub_local[out_i], all_local[view_i], atol=0, rtol=0)
    finally:
        bundle.teardown()


# ==================================================================
# Contract: Setters
# ==================================================================


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_roundtrip(device, view_factory):
    """set_world_poses -> get_world_poses returns the same values."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        new_pos = _wp_vec3f([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device)
        new_quat = _wp_vec4f([[0.0, 0.0, 0.7071068, 0.7071068], [0.0, 0.0, 0.0, 1.0]], device=device)
        bundle.view.set_world_poses(new_pos, new_quat)

        ret_pos, ret_quat = bundle.view.get_world_poses()
        torch.testing.assert_close(_t(ret_pos), _t(new_pos), atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(ret_quat), _t(new_quat), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_local_roundtrip(device, view_factory):
    """set_local_poses -> get_local_poses returns the same values."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        new_pos = _wp_vec3f([[0.5, 0.3, 0.1], [0.2, 0.7, 0.4]], device=device)
        new_quat = _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2, device=device)
        bundle.view.set_local_poses(new_pos, new_quat)

        ret_pos, ret_quat = bundle.view.get_local_poses()
        torch.testing.assert_close(_t(ret_pos), _t(new_pos), atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(ret_quat), _t(new_quat), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_does_not_move_parent(device, view_factory):
    """set_world_poses must not modify the parent prim/body position."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        parent_before = bundle.get_parent_pos(2, device).clone()
        bundle.view.set_world_poses(
            _wp_vec3f([[99.0, 99.0, 99.0], [88.0, 88.0, 88.0]], device=device),
            _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2, device=device),
        )
        parent_after = bundle.get_parent_pos(2, device)

        torch.testing.assert_close(parent_after, parent_before, atol=0, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_local_does_not_move_parent(device, view_factory):
    """set_local_poses must not modify the parent prim/body position."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        parent_before = bundle.get_parent_pos(2, device).clone()
        bundle.view.set_local_poses(
            _wp_vec3f([[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], device=device),
            _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2, device=device),
        )
        parent_after = bundle.get_parent_pos(2, device)

        torch.testing.assert_close(parent_after, parent_before, atol=0, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_updates_local(device, view_factory):
    """After set_world_poses, get_local_poses reflects the new offset.

    Uses non-axis-aligned offsets to catch coordinate swap bugs.
    """
    bundle = view_factory(num_envs=2, device=device)
    try:
        parent_pos = bundle.get_parent_pos(2, device)
        desired_offset = torch.tensor([[0.3, 0.7, 0.2], [0.8, 0.1, 0.6]], device=device)
        new_world = parent_pos + desired_offset

        bundle.view.set_world_poses(
            _wp_vec3f(new_world.tolist(), device=device),
            _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2, device=device),
        )

        local_pos = _t(bundle.view.get_local_poses()[0])
        torch.testing.assert_close(local_pos, desired_offset, atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_local_updates_world(device, view_factory):
    """After set_local_poses, get_world_poses == parent + new_local.

    Uses non-axis-aligned offsets to catch coordinate swap bugs.
    """
    bundle = view_factory(num_envs=2, device=device)
    try:
        parent_pos = bundle.get_parent_pos(2, device)
        new_offset = torch.tensor([[0.4, 0.9, 0.15], [0.6, 0.2, 0.85]], device=device)
        bundle.view.set_local_poses(
            _wp_vec3f(new_offset.tolist(), device=device),
            _wp_vec4f([[0.0, 0.0, 0.0, 1.0]] * 2, device=device),
        )

        world_pos = _t(bundle.view.get_world_poses()[0])
        torch.testing.assert_close(world_pos, parent_pos + new_offset, atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_partial_position_only(device, view_factory):
    """Setting only positions: new positions written, orientations preserved."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        _, orig_quat = bundle.view.get_world_poses()
        new_pos = _wp_vec3f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        bundle.view.set_world_poses(positions=new_pos)

        ret_pos, ret_quat = bundle.view.get_world_poses()
        torch.testing.assert_close(_t(ret_pos), _t(new_pos), atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(ret_quat), _t(orig_quat), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_partial_orientation_only(device, view_factory):
    """Setting only orientations: new orientations written, positions preserved."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        orig_pos, _ = bundle.view.get_world_poses()
        new_quat = _wp_vec4f([[0.0, 0.0, 0.7071068, 0.7071068], [0.7071068, 0.0, 0.0, 0.7071068]], device=device)
        bundle.view.set_world_poses(orientations=new_quat)

        ret_pos, ret_quat = bundle.view.get_world_poses()
        torch.testing.assert_close(_t(ret_pos), _t(orig_pos), atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(ret_quat), _t(new_quat), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_local_partial_position_only(device, view_factory):
    """Setting only local translations: new translations written, orientations preserved."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        _, orig_quat = bundle.view.get_local_poses()
        new_pos = _wp_vec3f([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]], device=device)
        bundle.view.set_local_poses(translations=new_pos)

        ret_pos, ret_quat = bundle.view.get_local_poses()
        torch.testing.assert_close(_t(ret_pos), _t(new_pos), atol=ATOL, rtol=0)
        torch.testing.assert_close(_t(ret_quat), _t(orig_quat), atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_set_world_indexed_only_affects_subset(device, view_factory):
    """Indexed set_world_poses writes requested indices, leaves others untouched."""
    bundle = view_factory(num_envs=4, device=device)
    try:
        orig_pos = _t(bundle.view.get_world_poses()[0]).clone()
        indices = wp.array([1, 3], dtype=wp.int32, device=device)
        new_pos = _wp_vec3f([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], device=device)
        bundle.view.set_world_poses(positions=new_pos, indices=indices)

        updated = _t(bundle.view.get_world_poses()[0])
        torch.testing.assert_close(updated[0], orig_pos[0], atol=0, rtol=0)
        torch.testing.assert_close(updated[2], orig_pos[2], atol=0, rtol=0)
        torch.testing.assert_close(updated[1], _t(new_pos)[0], atol=ATOL, rtol=0)
        torch.testing.assert_close(updated[3], _t(new_pos)[1], atol=ATOL, rtol=0)
    finally:
        bundle.teardown()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_return_types_are_torcharray(device, view_factory):
    """Public API contract — every backend returns ProxyArray from the pose getters."""
    bundle = view_factory(num_envs=2, device=device)
    try:
        pos_full, quat_full = bundle.view.get_world_poses()
        assert isinstance(pos_full, ProxyArray), (
            f"get_world_poses()[0] must be ProxyArray, got {type(pos_full).__name__}"
        )
        assert isinstance(quat_full, ProxyArray), (
            f"get_world_poses()[1] must be ProxyArray, got {type(quat_full).__name__}"
        )

        indices = wp.array([0], dtype=wp.int32, device=bundle.view.device)
        pos_idx, quat_idx = bundle.view.get_world_poses(indices)
        assert isinstance(pos_idx, ProxyArray), (
            f"get_world_poses(indices)[0] must be ProxyArray, got {type(pos_idx).__name__}"
        )
        assert isinstance(quat_idx, ProxyArray), (
            f"get_world_poses(indices)[1] must be ProxyArray, got {type(quat_idx).__name__}"
        )

        lpos_full, lquat_full = bundle.view.get_local_poses()
        assert isinstance(lpos_full, ProxyArray), (
            f"get_local_poses()[0] must be ProxyArray, got {type(lpos_full).__name__}"
        )
        assert isinstance(lquat_full, ProxyArray), (
            f"get_local_poses()[1] must be ProxyArray, got {type(lquat_full).__name__}"
        )

        lpos_idx, lquat_idx = bundle.view.get_local_poses(indices)
        assert isinstance(lpos_idx, ProxyArray), (
            f"get_local_poses(indices)[0] must be ProxyArray, got {type(lpos_idx).__name__}"
        )
        assert isinstance(lquat_idx, ProxyArray), (
            f"get_local_poses(indices)[1] must be ProxyArray, got {type(lquat_idx).__name__}"
        )
    finally:
        bundle.teardown()
