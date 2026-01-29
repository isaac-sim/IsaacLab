# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for mock interfaces tests.

This conftest sets up imports to allow testing the mock interfaces without
requiring Isaac Sim or the full isaaclab_physx package.

The mock interfaces are designed to be importable without Isaac Sim, so we
pre-load them into sys.modules to avoid triggering the full isaaclab_physx
package initialization.
"""

import sys
from pathlib import Path

# Get the path to the mock_interfaces module
_TEST_DIR = Path(__file__).parent
_MOCK_INTERFACES_DIR = _TEST_DIR.parent.parent / "isaaclab_physx" / "test" / "mock_interfaces"

# Add paths for direct imports
if str(_MOCK_INTERFACES_DIR) not in sys.path:
    sys.path.insert(0, str(_MOCK_INTERFACES_DIR))
if str(_MOCK_INTERFACES_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_MOCK_INTERFACES_DIR.parent))

# Pre-load mock_interfaces modules to prevent isaaclab_physx from being imported
# This allows tests to use `from isaaclab_physx.test.mock_interfaces import ...`
# without triggering the full package (which requires Isaac Sim)
import mock_interfaces
import mock_interfaces.views
import mock_interfaces.utils
import mock_interfaces.factories
import mock_interfaces.views.mock_rigid_body_view
import mock_interfaces.views.mock_articulation_view
import mock_interfaces.views.mock_rigid_contact_view
import mock_interfaces.utils.mock_shared_metatype
import mock_interfaces.utils.patching

# Create fake module entries in sys.modules for the isaaclab_physx path
# This makes `from isaaclab_physx.test.mock_interfaces import ...` work
sys.modules["isaaclab_physx"] = type(sys)("isaaclab_physx")
sys.modules["isaaclab_physx.test"] = type(sys)("isaaclab_physx.test")
sys.modules["isaaclab_physx.test.mock_interfaces"] = mock_interfaces
sys.modules["isaaclab_physx.test.mock_interfaces.views"] = mock_interfaces.views
sys.modules["isaaclab_physx.test.mock_interfaces.utils"] = mock_interfaces.utils
sys.modules["isaaclab_physx.test.mock_interfaces.factories"] = mock_interfaces.factories
sys.modules["isaaclab_physx.test.mock_interfaces.views.mock_rigid_body_view"] = (
    mock_interfaces.views.mock_rigid_body_view
)
sys.modules["isaaclab_physx.test.mock_interfaces.views.mock_articulation_view"] = (
    mock_interfaces.views.mock_articulation_view
)
sys.modules["isaaclab_physx.test.mock_interfaces.views.mock_rigid_contact_view"] = (
    mock_interfaces.views.mock_rigid_contact_view
)
sys.modules["isaaclab_physx.test.mock_interfaces.utils.mock_shared_metatype"] = (
    mock_interfaces.utils.mock_shared_metatype
)
sys.modules["isaaclab_physx.test.mock_interfaces.utils.patching"] = mock_interfaces.utils.patching
