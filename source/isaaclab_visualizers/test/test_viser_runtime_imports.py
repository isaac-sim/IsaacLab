# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Viser runtime dependency errors."""

import pytest
from isaaclab_visualizers.viser.viser_visualizer import NewtonViewerViser


def test_missing_viser_runtime_recommends_uv_extra(monkeypatch: pytest.MonkeyPatch):
    original_error = ImportError("viser package is required for ViewerViser. Install with: pip install viser")

    def _raise_missing_viser():
        raise original_error

    monkeypatch.setattr(NewtonViewerViser, "_get_viser", staticmethod(_raise_missing_viser))

    with pytest.raises(ImportError, match=r"uv run --extra viser <command>") as exc_info:
        NewtonViewerViser()

    assert exc_info.value.__cause__ is original_error
    assert "pip install viser" not in str(exc_info.value)
