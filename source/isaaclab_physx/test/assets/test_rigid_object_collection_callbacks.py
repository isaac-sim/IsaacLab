# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_physx.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection


def _make_collection_for_deletion_test() -> RigidObjectCollection:
    collection = object.__new__(RigidObjectCollection)
    collection.cfg = SimpleNamespace(
        rigid_objects={"object": SimpleNamespace(prim_path="/World/Table_[^/]*/Object_0")}
    )
    collection._is_initialized = True
    collection._root_view = object()
    collection._debug_vis_handle = None
    return collection


def test_prim_deletion_string_path_preserves_legacy_input() -> None:
    collection = _make_collection_for_deletion_test()

    collection._on_prim_deletion("/World/Table_0/Object_0")

    assert collection.is_initialized is False
    assert collection._root_view is None


def test_prim_deletion_event_invalidates_matching_collection() -> None:
    collection = _make_collection_for_deletion_test()

    collection._on_prim_deletion(SimpleNamespace(payload={"prim_path": "/World/Table_0/Object_0"}))

    assert collection.is_initialized is False
    assert collection._root_view is None


@pytest.mark.parametrize("event", [{"prim_path": "/"}, SimpleNamespace(payload={"prim_path": "/"})])
def test_prim_deletion_event_accepts_root_payload_forms(event) -> None:
    collection = _make_collection_for_deletion_test()

    collection._on_prim_deletion(event)

    assert collection.is_initialized is False
    assert collection._root_view is None


def test_nonmatching_prim_deletion_leaves_collection_initialized() -> None:
    collection = _make_collection_for_deletion_test()

    collection._on_prim_deletion(SimpleNamespace(payload={"prim_path": "/World/Other/Object_0"}))

    assert collection.is_initialized is True
    assert collection._root_view is not None


def test_prim_deletion_clears_callbacks_when_invalidation_fails() -> None:
    collection = _make_collection_for_deletion_test()
    collection._invalidate_initialize_callback = Mock(side_effect=RuntimeError("teardown failed"))
    collection._clear_callbacks = Mock()

    with pytest.raises(RuntimeError, match="teardown failed"):
        collection._on_prim_deletion({"prim_path": "/"})

    collection._clear_callbacks.assert_called_once_with()