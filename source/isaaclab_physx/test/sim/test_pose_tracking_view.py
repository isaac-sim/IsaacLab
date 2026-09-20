# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for automatic pose-write tracking on PhysX tensor views."""

from unittest.mock import Mock

import pytest
from isaaclab_physx.sim.views._pose_tracking_view import _PoseTrackingView


@pytest.mark.parametrize("method_name", ["set_root_transforms", "set_transforms"])
def test_successful_pose_write_marks_state_dirty(method_name: str):
    native_view = Mock()
    on_pose_write = Mock()
    view = _PoseTrackingView(native_view, on_pose_write)

    result = getattr(view, method_name)("transforms", indices="indices")

    assert result is getattr(native_view, method_name).return_value
    getattr(native_view, method_name).assert_called_once_with("transforms", indices="indices")
    on_pose_write.assert_called_once_with()


@pytest.mark.parametrize("method_name", ["set_root_transforms", "set_transforms"])
def test_failed_pose_write_does_not_mark_state_dirty(method_name: str):
    native_view = Mock()
    getattr(native_view, method_name).side_effect = RuntimeError("write failed")
    on_pose_write = Mock()
    view = _PoseTrackingView(native_view, on_pose_write)

    with pytest.raises(RuntimeError, match="write failed"):
        getattr(view, method_name)("transforms", indices="indices")

    on_pose_write.assert_not_called()


def test_other_view_operations_delegate_without_marking_state_dirty():
    native_view = Mock()
    native_view.get_transforms.return_value = "transforms"
    on_pose_write = Mock()
    view = _PoseTrackingView(native_view, on_pose_write)

    assert view.get_transforms() == "transforms"
    on_pose_write.assert_not_called()
