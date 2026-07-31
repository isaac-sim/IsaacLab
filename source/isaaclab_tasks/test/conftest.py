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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture()
def enable_scene_partition(monkeypatch):
    """Set ``ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION=1`` for the duration of one test."""
    monkeypatch.setenv("ISAAC_LAB_ENABLE_ISAAC_RTX_PER_ENV_SCENE_PARTITION", "1")


@pytest.fixture()
def ovstage_variant(request, monkeypatch):
    """Select the indirectly parametrized OVRTX stage path."""
    if request.param == "ovstage":
        monkeypatch.setenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", "1")
    else:
        # Clear explicitly rather than relying on the variable being unset. An ambient
        # ISAAC_LAB_OVRTX_USE_OVSTAGE=1 would otherwise make both variants exercise the ovstage
        # path, silently dropping legacy coverage while still reporting two passing variants.
        monkeypatch.delenv("ISAAC_LAB_OVRTX_USE_OVSTAGE", raising=False)
    return request.param
