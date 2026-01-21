# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for isaaclab_newton tests.

This conftest.py adds test subdirectories to the Python path so that local
helper modules (like mock_interface.py) can be imported by test files.
"""

import sys
from pathlib import Path

# Add test directory and subdirectories to path so local modules can be imported
test_dir = Path(__file__).parent
sys.path.insert(0, str(test_dir))
sys.path.insert(0, str(test_dir / "common"))
sys.path.insert(0, str(test_dir / "assets" / "articulation"))
sys.path.insert(0, str(test_dir / "assets" / "rigid_object"))
