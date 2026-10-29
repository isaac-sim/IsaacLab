# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public device pipelines, configuration selection and creation."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import isaaclab_teleop
import isaaclab_teleop.device_selection as device_selection
import numpy as np
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


def _env_cfg(isaac_teleop=None, **devices):
    return SimpleNamespace(isaac_teleop=isaac_teleop, teleop_devices=DevicesCfg(devices=devices))


def _action(cfg, payload):
    """Run the actual Capture graph with an encoded device snapshot."""
    from isaacteleop.retargeting_engine.interface import TensorGroup

    pipeline = cfg.pipeline_builder()
    (source,) = pipeline.get_leaf_nodes()
    input_name, group_type = next(iter(source.input_spec().items()))
    group = TensorGroup(group_type)
    group[0] = payload
    return pipeline, pipeline.execute_pipeline({source.name: {input_name: group}})


class _CustomKeyboardCfg(Se3KeyboardCfg):
    pass


@pytest.mark.parametrize(
    ("device", "cfg_type", "builder_name", "default_action", "custom_action"),
    [
        ("Keyboard", Se2KeyboardCfg, "se2_keyboard_teleop_cfg", [0.8, 0.4, 1.0], [0.1, 0.2, 0.3]),
        ("Keyboard", Se3KeyboardCfg, "se3_keyboard_teleop_cfg", [0.4, 0, 0, 0, 0.8, 0, -1], [0.7, 0, 0, 0, 0.6, 0]),
        ("Gamepad", Se2GamepadCfg, "se2_gamepad_teleop_cfg", [1.0, 0, 0.5], [0.1, 0.02, 0.15]),
        (
            "Gamepad",
            Se3GamepadCfg,
            "se3_gamepad_teleop_cfg",
            [1.0, 0, 0.25, 0, 0, 0.8, -1],
            [0.7, 0.07, 0.175, 0, 0, 0.3],
        ),
        ("SpaceMouse", Se2SpaceMouseCfg, "se2_spacemouse_teleop_cfg", [0.8, 0.1, 0.5], [0.1, 0.05, 0.15]),
        (
            "SpaceMouse",
            Se3SpaceMouseCfg,
            "se3_spacemouse_teleop_cfg",
            [0.4, 0.1, 0.2, 0.4, 0, 0, -1],
            [0.7, 0.175, 0.35, 0.3, 0, 0],
        ),
    ],
)
def test_device_pipelines(device, cfg_type, builder_name, default_action, custom_action):
    """Public defaults and converted overrides produce the right ordered float32 actions."""
    nodes = pytest.importorskip("isaacteleop.retargeting_engine.deviceio_source_nodes")
    from isaacteleop.schema import (
        GamepadInputState,
        GamepadSnapshot,
        KeyAction,
        KeyboardOutput,
        KeyEvent,
        Point,
        SpaceMouseOutput,
    )
    from isaacteleop.teleop_session_manager import BUNDLED_PLUGINS_DIR

    payload = {
        # W/T, arrows up/left, numpad 7; K closes the gripper.
        "Keyboard": KeyboardOutput([17, 20, 103, 105, 71], [KeyEvent(0, 37, KeyAction.PRESS)]),
        "Gamepad": GamepadSnapshot(
            GamepadInputState(
                west_click=True,
                thumbstick_left_y=1.0,
                thumbstick_left_x=0.1,
                thumbstick_right_y=0.25,
                thumbstick_right_x=0.5,
            )
        ),
        "SpaceMouse": SpaceMouseOutput(Point(0.25, 1.0, -0.5), Point(0.0, 0.5, 0.0), [1, 1], connected=True),
    }[device]
    builder = getattr(isaaclab_teleop, builder_name)
    cfg = builder()
    assert cfg.sim_device == "cpu" and cfg.teleoperation_active_default is True
    assert cfg.app_name == f"IsaacLab{device}Se{2 if len(default_action) == 3 else 3}"
    if device == "Keyboard":
        assert cfg.plugins == []
    else:
        assert [(p.plugin_name, p.plugin_root_id, p.search_paths, p.required) for p in cfg.plugins] == [
            (device.lower(), device.lower(), [BUNDLED_PLUGINS_DIR], True)
        ]
    pipeline, result = _action(cfg, payload)
    assert len(nodes.find_sources(pipeline, getattr(nodes, device + "Source"))) == 1
    assert pipeline.output_types()["action"].types[0].shape == (len(default_action),)
    assert result["action"][0].dtype == np.float32
    np.testing.assert_allclose(result["action"][0], default_action, atol=1e-6)
    if device == "SpaceMouse":
        assert result["spacemouse_buttons"][0][1] == 1

    settings = (
        dict(v_x_sensitivity=0.1, v_y_sensitivity=0.2, omega_z_sensitivity=0.3)
        if len(custom_action) == 3
        else dict(pos_sensitivity=0.7, rot_sensitivity=0.6, gripper_term=False)
    )
    if device == "Gamepad":
        settings["dead_zone"] = 0.05
    legacy_type = _CustomKeyboardCfg if cfg_type is Se3KeyboardCfg else cfg_type
    legacy_cfg = legacy_type(**settings, teleoperation_active_default=False)
    assert legacy_cfg.class_type is None
    with pytest.warns(FutureWarning, match=rf"deprecated in Isaac Lab 3\.2.*{builder_name}"):
        cfg = select_teleop_cfg(_env_cfg(), "device", {"device": legacy_cfg}, "cuda:0")
    assert cfg.sim_device == "cuda:0" and cfg.teleoperation_active_default is False
    _, result = _action(cfg, payload)
    assert result["action"][0].dtype == np.float32
    np.testing.assert_allclose(result["action"][0], custom_action, atol=1e-6)


@pytest.mark.parametrize("capture_available", [False, True])
@pytest.mark.parametrize(
    ("requested_device", "env_devices", "sensitivity"),
    [
        (None, {}, 0.2),
        (None, {"keyboard": Se3KeyboardCfg(pos_sensitivity=0.9)}, 0.9),
        ("spacemouse", {}, 0.3),
        ("spacemouse", {"spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.9)}, 0.9),
    ],
)
def test_environment_device_precedence(monkeypatch, capture_available, requested_device, env_devices, sensitivity):
    if capture_available:
        pytest.importorskip("isaacteleop")
    monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: object() if capture_available else None)
    defaults = {"keyboard": Se3KeyboardCfg(pos_sensitivity=0.2), "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=0.3)}
    # Explicit names override the declared pipeline; an unnamed request still uses the env's keyboard.
    env = _env_cfg(IsaacTeleopCfg() if requested_device is not None or not capture_available else None, **env_devices)
    if not capture_available:
        # keyboard, gamepad and SpaceMouse need Isaac Capture; there is no fallback to the native devices
        with pytest.raises(ImportError, match="teleop extra"):
            select_teleop_cfg(env, requested_device, defaults, "cpu")
        return
    with pytest.warns(FutureWarning, match="is deprecated"):
        cfg = select_teleop_cfg(env, requested_device, defaults, "cpu")
    from isaacteleop.schema import KeyboardOutput, Point, SpaceMouseOutput

    payload = (
        KeyboardOutput([17])
        if requested_device is None
        else SpaceMouseOutput(translation=Point(0, 1, 0), connected=True)
    )
    _, result = _action(cfg, payload)
    assert result["action"][0][0] == pytest.approx(sensitivity)


def test_declared_pipeline_and_unsupported_native_devices(monkeypatch):
    monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: object())
    env = _env_cfg(isaac_teleop=IsaacTeleopCfg(), handtracking=DeviceCfg())
    assert select_teleop_cfg(env, None, {}, "cpu") is env.isaac_teleop
    assert select_teleop_cfg(env, "keyboard", {"keyboard": env.isaac_teleop}, "cpu") is env.isaac_teleop
    assert select_teleop_cfg(env, "handtracking", {}, "cpu") is env.teleop_devices.devices["handtracking"]
    monkeypatch.setattr(device_selection.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ImportError, match="teleop extra"):
        select_teleop_cfg(env, None, {}, "cpu")
    with pytest.raises(ValueError, match="joystick"):
        select_teleop_cfg(env, "joystick", {}, "cpu")


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
    cfg = IsaacTeleopCfg(target_frame_prim_path="{ENV_REGEX_NS}/Robot")
    cfg.xr_cfg.anchor_prim_path = "{ENV_REGEX_NS}/Robot"
    capture = create_teleop_device(cfg, visualizers=windows, env_prim_path="/Lab/Cells/Cell0", use_kit_xr_bridge=False)
    assert isinstance(capture, IsaacTeleopDevice)
    assert capture._session_lifecycle._visualizers is windows
    assert capture._cfg.target_frame_prim_path == capture._cfg.xr_cfg.anchor_prim_path == "/Lab/Cells/Cell0/Robot"
    assert cfg.target_frame_prim_path == cfg.xr_cfg.anchor_prim_path == "{ENV_REGEX_NS}/Robot"


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
    package = root / "source/isaaclab_teleop/isaaclab_teleop"
    pipelines = ast.parse((package / "device_pipelines.py").read_text())
    assert not any(
        isinstance(node, ast.ImportFrom) and (node.module or "").startswith("isaaclab.devices")
        for node in ast.walk(pipelines)
    )
    packing_calls = sum(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TensorReorderer"
        for node in ast.walk(pipelines)
    )
    assert packing_calls == 1
    session = ast.parse((root / "source/isaaclab_teleop/isaaclab_teleop/session_lifecycle.py").read_text())
    assert not any(isinstance(node, ast.Name) and node.id == "SimulationContext" for node in ast.walk(session))
    assert not any(
        isinstance(node, ast.Attribute) and node.attr in ("_attached_visualizers", "_keyboard_detachers")
        for node in ast.walk(session)
    )
    for script in ("entrypoints/teleop_se3_agent.py", "entrypoints/record_demos.py"):
        tree = ast.parse((package / script).read_text())
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name in ("_builtin_device_cfgs", "setup_teleop_device")
            for node in ast.walk(tree)
        )
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        for name in ("select_teleop_cfg", "create_teleop_device"):
            assert sum(node.func.id == name for node in calls) == 1, (script, name)
        factory = next(node for node in calls if node.func.id == "create_teleop_device")
        assert {"visualizers", "env_prim_path"} <= {keyword.arg for keyword in factory.keywords}, script
