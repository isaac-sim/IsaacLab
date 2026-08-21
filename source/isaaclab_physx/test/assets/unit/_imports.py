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


def import_physx_module(module_name: str):
    """Import a PhysX asset module while replacing only its unavailable Kit boundary."""
    cloner = ModuleType("isaaclab_physx.cloner")
    cloner.queue_physx_replication = lambda cfg: None
    physics = ModuleType("isaaclab_physx.physics")
    physics.PhysxManager = type("PhysxManager", (), {})
    stubs = {"isaaclab_physx.cloner": cloner, "isaaclab_physx.physics": physics}
    import omni

    omni_physics = ModuleType("omni.physics")
    omni_physics.__path__ = []
    omni_physics.__spec__ = ModuleSpec("omni.physics", loader=None, is_package=True)
    omni_tensors = ModuleType("omni.physics.tensors")
    omni_tensors.__spec__ = ModuleSpec("omni.physics.tensors", loader=None)
    omni_physics.tensors = omni_tensors
    stubs.update({"omni.physics": omni_physics, "omni.physics.tensors": omni_tensors})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with patch.object(omni, "physics", omni_physics, create=True), patch.dict(sys.modules, stubs):
            return importlib.import_module(module_name)
