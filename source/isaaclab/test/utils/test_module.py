# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for module loading utilities."""

import sys

import pytest

from isaaclab.utils.module import deferred_import


def test_deferred_import_delays_module_loading():
    """Delay importing a module until one of its attributes is requested."""
    module_name = "email.quoprimime"
    sys.modules.pop(module_name, None)

    module = deferred_import(module_name)
    assert module_name not in sys.modules

    assert callable(module.header_check)
    assert module_name in sys.modules


def test_deferred_import_delays_missing_dependency_error():
    """Create a proxy for a missing optional dependency without raising eagerly."""
    module = deferred_import("isaaclab_missing_optional_dependency")

    with pytest.raises(ModuleNotFoundError, match="isaaclab_missing_optional_dependency"):
        module.some_attribute
