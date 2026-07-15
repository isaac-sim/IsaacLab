# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the lazy Isaac Sim simulation-manager patch lifecycle."""

import sys
import types
from unittest.mock import MagicMock

from isaaclab_physx._simulation_manager_patch import _SimulationManagerPatch


def test_patch_replaces_manager_after_disabling_original_callbacks(monkeypatch):
    """Replace exported symbols only after disabling the original manager."""

    class OriginalSimulationManager:
        enable_all_default_callbacks = MagicMock()

    class PhysxManager:
        pass

    isaac_events = object()
    original_module = types.ModuleType("isaacsim.core.simulation_manager")
    original_module.SimulationManager = OriginalSimulationManager
    physx_manager_module = types.ModuleType("isaaclab_physx.physics.physx_manager")
    physx_manager_module.PhysxManager = PhysxManager
    physx_manager_module.IsaacEvents = isaac_events
    monkeypatch.setitem(sys.modules, "isaacsim.core.simulation_manager", original_module)
    monkeypatch.setitem(sys.modules, "isaaclab_physx.physics.physx_manager", physx_manager_module)

    _SimulationManagerPatch().patch()

    OriginalSimulationManager.enable_all_default_callbacks.assert_called_once_with(False)
    assert original_module.SimulationManager is PhysxManager
    assert original_module.IsaacEvents is isaac_events


def test_subscribe_waits_for_kit_app(monkeypatch):
    """Leave the hook unset when the Kit module exists before its app interface."""
    kit_app_module = types.ModuleType("omni.kit.app")
    kit_app_module.get_app = MagicMock(side_effect=RuntimeError("IApp is not ready"))
    monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app_module)
    manager_patch = _SimulationManagerPatch()

    manager_patch.subscribe()

    assert manager_patch._extension_enable_hook is None


def test_subscribe_retains_one_extension_hook(monkeypatch):
    """Retain one hook and use it to patch late extension enablement."""
    extension_hook = object()
    extension_manager = MagicMock()
    extension_manager.subscribe_to_extension_enable.return_value = extension_hook
    app = MagicMock()
    app.get_extension_manager.return_value = extension_manager
    kit_app_module = types.ModuleType("omni.kit.app")
    kit_app_module.get_app = MagicMock(return_value=app)
    monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app_module)
    manager_patch = _SimulationManagerPatch()
    manager_patch.patch = MagicMock()

    manager_patch.subscribe()
    manager_patch.subscribe()

    assert manager_patch._extension_enable_hook is extension_hook
    extension_manager.subscribe_to_extension_enable.assert_called_once()
    on_enable_fn = extension_manager.subscribe_to_extension_enable.call_args.kwargs["on_enable_fn"]
    on_enable_fn("isaacsim.core.simulation_manager")
    manager_patch.patch.assert_called_once_with()


def test_disable_default_callbacks_uses_supported_api():
    """Prefer Isaac Sim's callback-control API when it is available."""
    original_class = MagicMock()

    _SimulationManagerPatch._disable_default_callbacks(original_class)

    original_class.enable_all_default_callbacks.assert_called_once_with(False)


def test_disable_default_callbacks_clears_legacy_handles():
    """Clear known callback handles when the callback-control API is absent."""
    original_class = type(
        "OriginalSimulationManager",
        (),
        {
            "_default_callback_warm_start": object(),
            "_default_callback_on_stop": object(),
            "_default_callback_stage_open": object(),
            "_default_callback_stage_close": object(),
        },
    )

    _SimulationManagerPatch._disable_default_callbacks(original_class)

    for attribute_name in _SimulationManagerPatch._DEFAULT_CALLBACK_ATTRIBUTES:
        assert getattr(original_class, attribute_name) is None
