# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless unit tests for Isaac Sim prebundle import sanitization."""

import sys
from importlib.machinery import PathFinder

import pytest

from isaaclab import _deprioritize_prebundle_paths

pytestmark = pytest.mark.unit


class FastFinder:
    """Stand-in for Kit's global fast importer."""


FastFinder.__module__ = "omni.ext._impl.fast_importer"


def test_sanitizer_places_kit_fast_finder_after_path_finder(monkeypatch):
    """Pip packages must resolve before identically named Kit extension modules."""
    meta_path = [FastFinder, *(finder for finder in sys.meta_path if finder is not FastFinder)]
    monkeypatch.setattr(sys, "meta_path", meta_path)
    monkeypatch.setattr(sys, "path", ["/venv/lib/python3.12/site-packages"])

    _deprioritize_prebundle_paths()

    assert sys.meta_path.index(PathFinder) < sys.meta_path.index(FastFinder)

    sanitized_meta_path = sys.meta_path.copy()
    _deprioritize_prebundle_paths()
    assert sys.meta_path == sanitized_meta_path
