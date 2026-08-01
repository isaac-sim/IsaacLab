# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for optional OVRTX runtime imports."""

import importlib
import sys

import pytest


def test_import_ovrtx_reports_uv_extra_when_runtime_missing(monkeypatch):
    """Missing ``ovrtx`` imports raise the uv-managed installation hint."""
    monkeypatch.delitem(sys.modules, "isaaclab_ov.renderers.ovrtx_renderer", raising=False)
    monkeypatch.setitem(sys.modules, "ovrtx", None)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module("isaaclab_ov.renderers.ovrtx_renderer")

    message = str(exc_info.value)
    assert "uv run --extra ovrtx" in message
    assert "./isaaclab.sh" not in message
    assert exc_info.value.__cause__.name == "ovrtx"
