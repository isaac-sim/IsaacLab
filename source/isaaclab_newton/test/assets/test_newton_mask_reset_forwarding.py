# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp reset-mask forwarding to Newton asset wrench storage."""

from unittest.mock import Mock

import warp as wp
from isaaclab_newton.assets.articulation.articulation import Articulation
from isaaclab_newton.assets.rigid_object.rigid_object import RigidObject
from isaaclab_newton.assets.rigid_object_collection.rigid_object_collection import RigidObjectCollection


def test_rigid_object_forwards_reset_mask_to_wrench_composers() -> None:
    """Rigid-object resets should preserve mask selection at wrench storage."""
    asset = RigidObject.__new__(RigidObject)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    asset._instantaneous_wrench_composer = Mock()
    asset._permanent_wrench_composer = Mock()
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    asset.reset(env_mask=env_mask)

    asset._instantaneous_wrench_composer.reset.assert_called_once_with(slice(None), env_mask)
    asset._permanent_wrench_composer.reset.assert_called_once_with(slice(None), env_mask)


def test_rigid_object_collection_forwards_reset_mask_to_wrench_composers() -> None:
    """Collection resets should preserve mask selection at wrench storage."""
    asset = RigidObjectCollection.__new__(RigidObjectCollection)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    asset._instantaneous_wrench_composer = Mock()
    asset._permanent_wrench_composer = Mock()
    env_ids = [0]
    env_mask = wp.array([True, False], dtype=wp.bool, device="cpu")

    asset.reset(env_ids=env_ids, object_ids=slice(None), env_mask=env_mask)

    asset._instantaneous_wrench_composer.reset.assert_called_once_with(env_ids, env_mask)
    asset._permanent_wrench_composer.reset.assert_called_once_with(env_ids, env_mask)


def test_articulation_without_lab_actuators_keeps_mask_on_device() -> None:
    """Mask resets should not materialize IDs when no Torch actuator consumes them."""
    asset = Articulation.__new__(Articulation)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    asset.actuators = {}
    asset._has_newton_actuators = False
    asset._instantaneous_wrench_composer = Mock()
    asset._permanent_wrench_composer = Mock()
    env_mask = object()

    asset.reset(env_mask=env_mask)

    asset._instantaneous_wrench_composer.reset.assert_called_once_with(slice(None), env_mask)
