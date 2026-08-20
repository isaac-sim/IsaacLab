# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for package exports and public API boundaries."""

import subprocess
import sys


def test_public_export_resolves_private_implementation_lazily():
    """Public symbols should resolve from ``_src`` without loading them at package import time."""
    code = """
import sys

import isaaclab.assets

implementation = "isaaclab._src.assets.articulation.articulation_cfg"
assert implementation not in sys.modules
assert "ArticulationCfg" not in isaaclab.assets.__dict__

from isaaclab.assets import ArticulationCfg

assert implementation in sys.modules
assert ArticulationCfg.__module__ == implementation
assert isaaclab.assets.ArticulationCfg is ArticulationCfg
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_implementation_module_is_not_available_through_public_namespace():
    """Implementation modules should only be importable below ``isaaclab._src``."""
    code = """
import importlib.util

assert importlib.util.find_spec("isaaclab.assets.articulation.articulation_cfg") is None
assert importlib.util.find_spec("isaaclab._src.assets.articulation.articulation_cfg") is not None
"""
    subprocess.run([sys.executable, "-c", code], check=True)
