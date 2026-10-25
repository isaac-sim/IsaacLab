# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :mod:`isaaclab_teleop.script_input`: teleop devices for scripts."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import isaaclab_teleop.device_selection as device_selection
import isaaclab_teleop.script_input as script_input
import pytest
from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV
from isaaclab_teleop.script_input import open_teleop_device

pytestmark = pytest.mark.unit


def test_without_isaacteleop_opening_a_device_raises(monkeypatch):
    monkeypatch.setattr(device_selection, "_isaacteleop_available", lambda: False)
    with pytest.raises(ImportError, match="teleop extra"):
        with open_teleop_device(SimpleNamespace(name="keyboard", sim_device="cpu")):
            pass


def test_with_isaacteleop_the_pipeline_runs_headset_free(monkeypatch):
    monkeypatch.setattr(device_selection, "_isaacteleop_available", lambda: True)
    monkeypatch.setattr(script_input, "pipeline_from_device_cfg", lambda cfg, sim_device: f"pipeline:{cfg.name}")
    device = MagicMock()
    device.__enter__.return_value = device
    with patch("isaaclab_teleop.isaac_teleop_device.create_isaac_teleop_device", return_value=device) as create:
        with open_teleop_device(SimpleNamespace(name="keyboard", sim_device="cpu")) as opened:
            assert opened is device
            device.request_start.assert_called_once()
            device.__exit__.assert_not_called()
    create.assert_called_once_with(
        "pipeline:keyboard", use_kit_xr_bridge=False, cloudxr_env_file=CLOUDXR_STANDALONE_ENV
    )
    device.__exit__.assert_called_once()
