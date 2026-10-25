# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for :mod:`isaaclab_teleop.device_selection`: device configs -> Isaac Capture pipelines."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import isaaclab_teleop.device_selection as device_selection
import pytest
from isaaclab_teleop import IsaacTeleopCfg
from isaaclab_teleop.device_selection import create_teleop_device, select_teleop_cfg

from isaaclab.devices import (
    DeviceBase,
    DeviceCfg,
    DevicesCfg,
    Se2GamepadCfg,
    Se2KeyboardCfg,
    Se2SpaceMouseCfg,
    Se3GamepadCfg,
    Se3KeyboardCfg,
    Se3SpaceMouseCfg,
)

pytestmark = pytest.mark.unit


BUILDERS = {
    Se3KeyboardCfg: "se3_keyboard_teleop_cfg",
    Se2KeyboardCfg: "se2_keyboard_teleop_cfg",
    Se3GamepadCfg: "se3_gamepad_teleop_cfg",
    Se2GamepadCfg: "se2_gamepad_teleop_cfg",
    Se3SpaceMouseCfg: "se3_spacemouse_teleop_cfg",
    Se2SpaceMouseCfg: "se2_spacemouse_teleop_cfg",
}
"""Each device config type and the builder that must serve it."""


@pytest.fixture
def recorded(monkeypatch):
    """Record which builder ``device_selection`` calls, and with what, without changing its routing."""
    calls = []
    for cfg_type, (routed, fields) in list(device_selection._PIPELINES.items()):

        def builder(name=routed.__name__, **kwargs):
            calls.append((name, kwargs))
            return f"pipeline:{name}"

        monkeypatch.setitem(device_selection._PIPELINES, cfg_type, (builder, fields))
    monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: object())
    return calls


def _env_cfg(isaac_teleop=None, **devices):
    return SimpleNamespace(isaac_teleop=isaac_teleop, teleop_devices=DevicesCfg(devices=devices))


DEFAULTS = {"keyboard": Se3KeyboardCfg(pos_sensitivity=0.2), "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.3)}


class TestPipelineFromDeviceCfg:
    @pytest.mark.parametrize(
        ("cfg_type", "source_name"),
        [
            (Se2KeyboardCfg, "KeyboardSource"),
            (Se3KeyboardCfg, "KeyboardSource"),
            (Se2GamepadCfg, "GamepadSource"),
            (Se3GamepadCfg, "GamepadSource"),
            (Se2SpaceMouseCfg, "SpaceMouseSource"),
            (Se3SpaceMouseCfg, "SpaceMouseSource"),
        ],
    )
    def test_every_legacy_device_reads_its_own_device(self, cfg_type, source_name):
        nodes = pytest.importorskip("isaacteleop.retargeting_engine.deviceio_source_nodes")

        cfg = select_teleop_cfg(_env_cfg(), "device", {"device": cfg_type()}, "cpu")

        assert isinstance(cfg, IsaacTeleopCfg)
        assert len(nodes.find_sources(cfg.pipeline_builder(), getattr(nodes, source_name))) == 1

    @pytest.mark.parametrize(
        ("device_cfg", "expected"),
        [
            (
                Se3KeyboardCfg(pos_sensitivity=0.7, rot_sensitivity=0.6, gripper_term=False),
                {"pos_sensitivity": 0.7, "rot_sensitivity": 0.6, "gripper_term": False},
            ),
            (
                Se2KeyboardCfg(v_x_sensitivity=0.1, v_y_sensitivity=0.2, omega_z_sensitivity=0.3),
                {"v_x_sensitivity": 0.1, "v_y_sensitivity": 0.2, "omega_z_sensitivity": 0.3},
            ),
            (
                Se3GamepadCfg(pos_sensitivity=0.7, rot_sensitivity=0.6, dead_zone=0.05, gripper_term=False),
                {"pos_sensitivity": 0.7, "rot_sensitivity": 0.6, "dead_zone": 0.05, "gripper_term": False},
            ),
            (
                Se2GamepadCfg(v_x_sensitivity=0.1, v_y_sensitivity=0.2, omega_z_sensitivity=0.3, dead_zone=0.05),
                {"v_x_sensitivity": 0.1, "v_y_sensitivity": 0.2, "omega_z_sensitivity": 0.3, "dead_zone": 0.05},
            ),
            (
                Se3SpaceMouseCfg(pos_sensitivity=0.7, rot_sensitivity=0.6, gripper_term=False),
                {"pos_sensitivity": 0.7, "rot_sensitivity": 0.6, "gripper_term": False},
            ),
            (
                Se2SpaceMouseCfg(v_x_sensitivity=0.1, v_y_sensitivity=0.2, omega_z_sensitivity=0.3),
                {"v_x_sensitivity": 0.1, "v_y_sensitivity": 0.2, "omega_z_sensitivity": 0.3},
            ),
        ],
        ids=lambda value: type(value).__name__ if not isinstance(value, dict) else "",
    )
    def test_config_reaches_its_builder_with_every_setting(self, recorded, device_cfg, expected):
        select_teleop_cfg(_env_cfg(), "device", {"device": device_cfg}, "cuda:0")

        assert recorded == [(BUILDERS[type(device_cfg)], {**expected, "sim_device": "cuda:0"})]

    def test_unsupported_config_has_no_pipeline(self):
        cfg = DeviceCfg()
        assert select_teleop_cfg(_env_cfg(), "device", {"device": cfg}, "cpu") is cfg


class TestSelectIsaacTeleopCfg:
    def test_without_isaacteleop_selects_native_keyboard(self, recorded, monkeypatch):
        monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: None)
        cfg = Se3KeyboardCfg(pos_sensitivity=0.9)
        env_cfg = _env_cfg(isaac_teleop="declared", keyboard=cfg)
        assert select_teleop_cfg(env_cfg, None, DEFAULTS, "cpu") is env_cfg.teleop_devices.devices["keyboard"]

    def test_declared_pipeline_wins_without_a_device_name(self, recorded):
        assert select_teleop_cfg(_env_cfg(isaac_teleop="declared"), None, DEFAULTS, "cpu") == "declared"
        assert recorded == []

    def test_default_keyboard_without_a_declared_pipeline(self, recorded):
        assert select_teleop_cfg(_env_cfg(), None, DEFAULTS, "cpu") == "pipeline:se3_keyboard_teleop_cfg"
        assert recorded[0][1]["pos_sensitivity"] == 0.2

        recorded.clear()
        env_cfg = _env_cfg(keyboard=Se3KeyboardCfg(pos_sensitivity=0.9))
        select_teleop_cfg(env_cfg, None, DEFAULTS, "cpu")
        assert recorded[0][1]["pos_sensitivity"] == 0.9

    def test_environment_device_wins_over_the_default(self, recorded):
        env_cfg = _env_cfg(isaac_teleop="declared", spacemouse=Se3SpaceMouseCfg(pos_sensitivity=0.9))

        assert select_teleop_cfg(env_cfg, "spacemouse", DEFAULTS, "cpu") == "pipeline:se3_spacemouse_teleop_cfg"
        assert recorded[0][1]["pos_sensitivity"] == 0.9

    def test_default_serves_a_name_the_environment_lacks(self, recorded):
        assert select_teleop_cfg(_env_cfg(), "spacemouse", DEFAULTS, "cpu") == "pipeline:se3_spacemouse_teleop_cfg"
        assert recorded[0][1]["pos_sensitivity"] == 0.3

    def test_unsupported_device_retains_its_config_and_unknown_name_raises(self, recorded):
        env_cfg = _env_cfg(handtracking=DeviceCfg())

        assert (
            select_teleop_cfg(env_cfg, "handtracking", DEFAULTS, "cpu")
            is env_cfg.teleop_devices.devices["handtracking"]
        )
        with pytest.raises(ValueError, match="joystick"):
            select_teleop_cfg(env_cfg, "joystick", DEFAULTS, "cpu")


class _NativeDevice(DeviceBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.callbacks = {}

    def reset(self):
        pass

    def add_callback(self, key, callback):
        self.callbacks[key] = callback


def test_creation_uses_the_resolved_config_for_both_implementations():
    """Native callbacks and explicitly supplied Capture windows reach the selected device."""
    cfg = DeviceCfg(class_type=_NativeDevice)

    def callback():
        pass

    native = create_teleop_device(cfg, callbacks={"R": callback}, use_kit_xr_bridge=False)
    assert native.cfg is cfg and native.callbacks == {"R": callback}

    from isaaclab_teleop.isaac_teleop_device import IsaacTeleopDevice

    windows = []
    capture = create_teleop_device(IsaacTeleopCfg(), visualizers=windows, use_kit_xr_bridge=False)
    assert isinstance(capture, IsaacTeleopDevice)
    assert capture._session_lifecycle._visualizers is windows


def test_capture_creation_failure_does_not_switch_to_a_native_device(monkeypatch):
    from isaaclab_teleop import isaac_teleop_device

    def fail(*args, **kwargs):
        raise RuntimeError("Capture initialization failed")

    monkeypatch.setattr(isaac_teleop_device, "create_isaac_teleop_device", fail)
    with pytest.raises(RuntimeError, match="Capture initialization failed"):
        create_teleop_device(IsaacTeleopCfg())


def test_window_dependencies_and_device_creation_have_one_owner():
    """Keep global window discovery and repeated device-selection branches out of the session/scripts."""
    root = Path(__file__).resolve().parents[3]
    session = ast.parse((root / "source/isaaclab_teleop/isaaclab_teleop/session_lifecycle.py").read_text())
    assert not any(isinstance(node, ast.Name) and node.id == "SimulationContext" for node in ast.walk(session))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("_attached_visualizers", "_keyboard_detachers")
        for node in ast.walk(session)
    )
    for script in ("scripts/environments/teleoperation/teleop_se3_agent.py", "scripts/tools/record_demos.py"):
        tree = ast.parse((root / script).read_text())
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name in ("_builtin_device_cfgs", "setup_teleop_device")
            for node in ast.walk(tree)
        )
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        for name in ("select_teleop_cfg", "create_teleop_device"):
            assert sum(node.func.id == name for node in calls) == 1, (script, name)
        factory = next(node for node in calls if node.func.id == "create_teleop_device")
        assert any(keyword.arg == "visualizers" for keyword in factory.keywords), script
