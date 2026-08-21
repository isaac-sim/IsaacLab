# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Focused PhysX rigid-object CPU staging and cached-view tests."""

import importlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
import warp as wp

from ._imports import import_physx_module


def _rigid_object_class():
    return import_physx_module("isaaclab_physx.assets.rigid_object.rigid_object").RigidObject


def test_partial_int64_environment_selector_is_narrowed_and_staged_on_cpu() -> None:
    """PhysX CPU property writers must receive int32 indices even from public int64 selectors."""
    rigid_object = object.__new__(_rigid_object_class())
    rigid_object._device = "cuda:0"
    rigid_object._ALL_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cuda:0")
    rigid_object._cpu_env_ids_all = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    rigid_object._cpu_env_ids = wp.empty(3, dtype=wp.int32, device="cpu", pinned=True)
    rigid_object._cpu_env_ids_views = {}

    result = rigid_object._get_cpu_env_ids(torch.tensor([2, 0], dtype=torch.int64, device="cuda:0"))

    assert result.dtype == wp.int32
    assert str(result.device) == "cpu"
    np.testing.assert_array_equal(result.numpy(), [2, 0])


def test_tensor_api_float_view_is_cached_over_stable_pose_storage() -> None:
    """Repeated root-pose writes must reuse the wrapper over the stable data buffer."""
    rigid_object = object.__new__(_rigid_object_class())
    pose = wp.zeros(2, dtype=wp.transformf, device="cpu")
    rigid_object._data = SimpleNamespace(_root_link_pose_w=SimpleNamespace(data=pose))
    rigid_object._root_link_pose_w_f32 = None

    first = rigid_object._get_root_link_pose_w_f32()
    second = rigid_object._get_root_link_pose_w_f32()

    assert first is second
    assert first.ptr == pose.ptr
    assert first.shape == (2, 7)


def test_kitless_import_is_evicted_before_fresh_manager_import() -> None:
    """A stub-bound unit import must not leak into the next real PhysX module import."""
    module_name = "isaaclab_physx.assets.rigid_object.rigid_object"
    stubbed_module = import_physx_module(module_name)
    assert module_name not in sys.modules

    try:
        fresh_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Plain uv Python has no Kit ``carb`` module. Reaching that boundary proves the
        # real lazy PhysX manager replaced the unit stub.
        assert exc.name == "carb"
        assert module_name not in sys.modules
    else:
        from isaaclab_physx.physics import PhysxManager

        assert fresh_module is not stubbed_module
        assert fresh_module.SimulationManager is PhysxManager
