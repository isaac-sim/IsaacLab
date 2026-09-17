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

import pytest
import warp as wp

wp.config.enable_backward = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def ovstage_variant(request, monkeypatch):
    """Select the indirectly parametrized OVRTX stage path."""
    if request.param == "ovstage":
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    else:
        # Select legacy explicitly so both variants retain coverage when ovstage is the default.
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "0")
    return request.param
