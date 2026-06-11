# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pytest configuration for the isaaclab_tasks test suite.

Adds this directory to ``sys.path`` so tests located in the ``core/`` and ``contrib/``
sub-directories can import the shared helpers (``env_test_utils``, ``rendering_test_utils``)
that live at the test-suite root.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
