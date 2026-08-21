# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless import helpers for PhysX asset unit tests."""

import importlib
import sys
import warnings
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import patch

_MISSING = object()


def import_physx_module(module_name: str, *, simulation_manager: type | None = None):
    """Import a PhysX asset module while replacing only its unavailable Kit boundary.

    Args:
        module_name: Fully qualified production module name.
        simulation_manager: Manager class to expose at the import boundary. A lightweight
            placeholder is created when omitted.
    """
    module_parts = module_name.split(".")
    asset_package_name = module_name.rsplit(".", 1)[0]
    asset_subtree_snapshot = {
        name: module
        for name, module in sys.modules.items()
        if name == asset_package_name or name.startswith(f"{asset_package_name}.")
    }
    ancestor_names = [".".join(module_parts[:index]) for index in range(1, len(module_parts))]
    ancestor_snapshots = {
        name: (sys.modules[name], dict(sys.modules[name].__dict__)) for name in ancestor_names if name in sys.modules
    }
    missing_ancestors = [name for name in ancestor_names if name not in sys.modules]
    for name in asset_subtree_snapshot:
        sys.modules.pop(name, None)
    cloner = ModuleType("isaaclab_physx.cloner")
    cloner.queue_physx_replication = lambda cfg: None
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = simulation_manager or type("PhysxManager", (), {})
    stubs = {"isaaclab_physx.cloner": cloner, "isaaclab_physx.physics": physics}
    import omni

    omni_physics = ModuleType("omni.physics")
    omni_physics.__path__ = []
    omni_physics.__spec__ = ModuleSpec("omni.physics", loader=None, is_package=True)
    omni_tensors = ModuleType("omni.physics.tensors")
    omni_tensors.__spec__ = ModuleSpec("omni.physics.tensors", loader=None)
    omni_physics.tensors = omni_tensors
    stubs.update({"omni.physics": omni_physics, "omni.physics.tensors": omni_tensors})
    stub_snapshots = {name: sys.modules.get(name, _MISSING) for name in stubs}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            with patch.object(omni, "physics", omni_physics, create=True):
                sys.modules.update(stubs)
                imported = importlib.import_module(module_name)
        finally:
            current_asset_modules = [
                name
                for name in sys.modules
                if name == asset_package_name or name.startswith(f"{asset_package_name}.")
            ]
            for name in current_asset_modules:
                sys.modules.pop(name, None)
            sys.modules.update(asset_subtree_snapshot)
            for name, previous_module in stub_snapshots.items():
                if previous_module is _MISSING:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = previous_module
            for name in reversed(missing_ancestors):
                sys.modules.pop(name, None)
            for module, namespace in ancestor_snapshots.values():
                module.__dict__.clear()
                module.__dict__.update(namespace)
    return imported
