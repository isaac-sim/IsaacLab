# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Warp reset-mask forwarding to Newton asset wrench storage."""

from unittest.mock import Mock, patch

import torch
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


def test_articulation_legacy_mode_forwards_mask_to_actuators() -> None:
    """Legacy-mode masked resets should hand actuators the boolean view, not compact IDs."""
    asset = Articulation.__new__(Articulation)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    lab_actuator = Mock()
    asset.actuators = {"legs": lab_actuator}
    asset._has_newton_actuators = False
    asset._instantaneous_wrench_composer = Mock()
    asset._permanent_wrench_composer = Mock()
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    asset.reset(env_mask=env_mask)

    lab_actuator.reset.assert_not_called()
    forwarded_mask = lab_actuator.reset_mask.call_args.args[0]
    assert forwarded_mask.dtype == torch.bool
    torch.testing.assert_close(forwarded_mask, torch.tensor([False, True]))


def test_articulation_reset_capture_safe_reflects_actuator_capability() -> None:
    """Capture safety should require mask-native actuator resets in legacy mode."""
    from isaaclab.actuators import ActuatorBase, ImplicitActuator

    class _LegacyActuator(ActuatorBase):
        def reset(self, env_ids):
            pass

        def compute(self, control_action, joint_pos, joint_vel):
            return control_action

    asset = Articulation.__new__(Articulation)
    asset._has_newton_actuators = True
    asset.actuators = {}
    assert asset.reset_capture_safe

    asset._has_newton_actuators = False
    asset.actuators = {"implicit": ImplicitActuator.__new__(ImplicitActuator)}
    assert asset.reset_capture_safe

    asset.actuators = {"legacy": _LegacyActuator.__new__(_LegacyActuator)}
    assert not asset.reset_capture_safe


def test_articulation_native_mode_masked_reset_skips_lab_actuators() -> None:
    """Native-mode partial resets must not reset Lab actuator state for all environments."""
    asset = Articulation.__new__(Articulation)
    asset._initialize_handle = None
    asset._invalidate_initialize_handle = None
    asset._prim_deletion_handle = None
    lab_actuator = Mock()
    asset.actuators = {"legs": lab_actuator}
    asset._has_newton_actuators = True
    asset._instantaneous_wrench_composer = Mock()
    asset._permanent_wrench_composer = Mock()
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    adapter = Mock()

    from isaaclab_newton.physics import NewtonManager

    with patch.object(NewtonManager, "_adapter", adapter):
        asset.reset(env_mask=env_mask)

    lab_actuator.reset.assert_not_called()
    adapter.reset.assert_called_once_with(slice(None), env_mask=env_mask)
