# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for :mod:`isaaclab_teleop.script_input`: script hotkeys and teleop devices."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import isaaclab_teleop.script_input as script_input
import pytest
from isaaclab_teleop.isaac_teleop_cfg import CLOUDXR_STANDALONE_ENV
from isaaclab_teleop.script_input import add_key_callbacks, open_teleop_device

from isaaclab.visualizers import KeyEventSource

pytestmark = pytest.mark.unit


@pytest.fixture
def source():
    """A visualizer window's keyboard, read by :func:`add_key_callbacks` through a fake simulation."""
    key_event_source = KeyEventSource()
    sim = SimpleNamespace(
        visualizers=[SimpleNamespace(key_event_source=key_event_source), SimpleNamespace(key_event_source=None)]
    )
    with patch("isaaclab.sim.SimulationContext.instance", return_value=sim):
        yield key_event_source


def test_key_callbacks_fire_once_per_press(source):
    pressed = []
    add_key_callbacks({"N": lambda: pressed.append("N"), "1": lambda: pressed.append("1"), "Escape": lambda: 0})

    source._emit_key("KeyN", True)
    source._emit_key("KeyN", True)  # key repeat
    source._emit_key("KeyN", False)
    source._emit_key("KeyN", True)
    source._emit_key("Digit1", True)
    source._emit_key("KeyM", True)
    assert pressed == ["N", "N", "1"]


def test_focus_loss_ends_a_held_key(source):
    pressed = []
    add_key_callbacks({"N": lambda: pressed.append("N")})
    source._emit_key("KeyN", True)
    source._emit_focus_lost()
    source._emit_key("KeyN", True)
    assert pressed == ["N", "N"]


def test_removing_the_key_callbacks_unsubscribes(source):
    remove = add_key_callbacks({"N": lambda: None})
    assert len(source._subscriptions) == 1
    remove()
    assert source._subscriptions == []


def test_without_isaacteleop_the_device_config_is_instantiated(monkeypatch):
    monkeypatch.setattr(script_input, "_isaacteleop_available", lambda: False)
    device = object()
    with patch("isaaclab.utils.instantiate", return_value=device) as instantiate:
        with open_teleop_device("cfg") as opened:
            assert opened is device
    instantiate.assert_called_once_with("cfg")


def test_with_isaacteleop_the_pipeline_runs_headset_free(monkeypatch):
    monkeypatch.setattr(script_input, "_isaacteleop_available", lambda: True)
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
