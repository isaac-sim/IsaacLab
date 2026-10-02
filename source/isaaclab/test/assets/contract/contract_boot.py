# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kit and kitless bootstrap for the asset contract tests."""

import os
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import MagicMock

_kitless = "ovphysx" in os.environ.get("LD_PRELOAD", "") or (
    os.environ.get("LD_PRELOAD", "") == "" and "EXP_PATH" not in os.environ
)

if not _kitless:
    from isaaclab.test.utils import launch_test_simulation

    launch_test_simulation()
else:

    def _install_stub(module_name: str, *, is_package: bool = False) -> None:
        """Install one import-boundary module double unless the real module is already loaded."""
        if module_name in sys.modules:
            return
        stub = MagicMock()
        stub.__spec__ = ModuleSpec(module_name, loader=None, is_package=is_package)
        if is_package:
            stub.__path__ = []
        sys.modules[module_name] = stub
        if "." in module_name:
            parent_name, attribute = module_name.rsplit(".", 1)
            setattr(sys.modules[parent_name], attribute, stub)

    # Without Kit's Python runtime, ``carb`` and ``usdrt`` are missing even when ``isaaclab_physx`` is installed. Stub
    # only that runtime boundary so the contracts still import the real PhysX asset and data classes and their views.
    _install_stub("carb")
    _install_stub("usdrt", is_package=True)

    # ``omni`` is a real namespace package in kitless runs. Install missing submodules in both ``sys.modules`` and the
    # namespace attributes.
    import omni as _omni  # noqa: F401

    for _module_name, _is_package in (
        ("omni.kit", True),
        ("omni.kit.app", False),
        ("omni.physics", True),
        ("omni.physics.tensors", False),
        ("omni.physx", False),
        ("omni.timeline", False),
        ("omni.usd", False),
        ("isaacsim", True),
        ("isaacsim.core", True),
        ("isaacsim.core.simulation_manager", False),
    ):
        _install_stub(_module_name, is_package=_is_package)
    sys.modules["omni.kit.app"].get_app.return_value = None
