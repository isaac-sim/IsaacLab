# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for telling a stale CI image apart from a genuine defect in the PR."""

from __future__ import annotations

import sys
from pathlib import Path

_GATE_DIR = Path(__file__).resolve().parents[1]
if str(_GATE_DIR) not in sys.path:
    sys.path.insert(0, str(_GATE_DIR))

from environment_skew import detect_dependency_skew  # noqa: E402

# Verbatim from the 2026-07-28 staging run, where source pinning a newer Newton
# met an image built before SolverNotifyFlags was replaced by ModelFlags.
_SOLVER_NOTIFY_FLAGS_LOG = """
Traceback (most recent call last):
  File "/workspace/isaaclab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py", line 21, in <module>
    from newton.solvers import SolverNotifyFlags
ImportError: cannot import name 'SolverNotifyFlags' from 'newton.solvers'
"""


def test_missing_newton_symbol_is_reported_as_stale_image() -> None:
    """The crash that broke the last qualification run must be recognized."""
    skew = detect_dependency_skew(_SOLVER_NOTIFY_FLAGS_LOG)

    assert skew is not None
    assert skew.package == "newton"
    assert skew.symbol == "SolverNotifyFlags"
    assert "no `SolverNotifyFlags`" in skew.describe()


def test_missing_image_module_is_reported_as_stale_image() -> None:
    """A package the image should provide but does not is also image skew."""
    skew = detect_dependency_skew("ModuleNotFoundError: No module named 'newton.solvers.kamino'")

    assert skew is not None
    assert skew.package == "newton"
    assert skew.symbol is None
    assert "is not installed" in skew.describe()


def test_missing_module_attribute_is_reported_as_stale_image() -> None:
    """Warp exposing no such attribute means the installed Warp predates the pin."""
    skew = detect_dependency_skew("AttributeError: module 'warp' has no attribute 'sparse_matmul'")

    assert skew is not None
    assert skew.package == "warp"
    assert skew.symbol == "sparse_matmul"


def test_missing_isaaclab_symbol_is_not_excused() -> None:
    """Isaac Lab source is mounted from the PR, so its own broken import is a real defect."""
    log = "ImportError: cannot import name 'ArticulationCfg' from 'isaaclab.assets'"

    assert detect_dependency_skew(log) is None


def test_clean_log_reports_no_skew() -> None:
    """A run that never raised an import error is not skew."""
    assert detect_dependency_skew("Step Frametimes: 3.1 3.0 3.2") is None
    assert detect_dependency_skew(None) is None
