# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :mod:`isaaclab_teleop.script_input`: teleop devices for scripts."""

from __future__ import annotations

from unittest.mock import MagicMock

import isaaclab_teleop.device_selection as device_selection
import isaaclab_teleop.script_input as script_input
import pytest
from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV, IsaacTeleopCfg
from isaaclab_teleop.script_input import open_teleop_device

from isaaclab.devices import Se3KeyboardCfg

pytestmark = pytest.mark.unit


def test_without_isaacteleop_opening_a_device_raises(monkeypatch):
    monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ImportError, match="teleop extra"):
        with open_teleop_device(Se3KeyboardCfg(), visualizers=()):
            pass


def test_with_isaacteleop_the_pipeline_runs_headset_free(monkeypatch):
    cfg = IsaacTeleopCfg()
    monkeypatch.setattr(script_input, "select_teleop_cfg", lambda env_cfg, name, defaults, sim_device: cfg)
    device = MagicMock()
    device.__enter__.return_value = device
    create = MagicMock(return_value=device)
    monkeypatch.setattr(script_input, "create_teleop_device", create)
    visualizers = [object()]
    with open_teleop_device(Se3KeyboardCfg(), visualizers) as opened:
        assert opened is device
        device.request_start.assert_called_once()
        device.__exit__.assert_not_called()
    create.assert_called_once_with(
        cfg, use_kit_xr_bridge=False, cloudxr_env_file=CLOUDXR_STANDALONE_ENV, visualizers=visualizers
    )
    device.__exit__.assert_called_once()
