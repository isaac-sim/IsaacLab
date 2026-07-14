# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import pytest

pytestmark = pytest.mark.isaacsim_ci

from isaaclab.devices import DVRKOpenXRDevice, DVRKOpenXRDeviceCfg, create_teleop_device
from isaaclab.devices.openxr.openxr_device import OpenXRDevice
from isaaclab.devices.openxr.retargeters import (
    DVRKPSMRetargeter,
    DVRKPSMRetargeterCfg,
    DVRKPSMSideRetargeterCfg,
)


def _retargeter_cfg() -> DVRKPSMRetargeterCfg:
    side = DVRKPSMSideRetargeterCfg(
        home_position=(0.0, 0.0, 0.2),
        home_orientation=(0.0, 0.0, 0.0, 1.0),
        workspace_lower=(-0.1, -0.1, 0.1),
        workspace_upper=(0.1, 0.1, 0.3),
        jaw_open=(-0.5, 0.5),
        jaw_closed=(-0.09, 0.09),
    )
    return DVRKPSMRetargeterCfg(left=side, right=side, sim_device="cpu")


@pytest.fixture
def retargeter() -> DVRKPSMRetargeter:
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")
    return DVRKPSMRetargeter(_retargeter_cfg())


@pytest.fixture
def patched_openxr_init(mocker):
    def initialise(device, cfg, retargeters=None):
        device._retargeters = retargeters or []
        device._required_features = set()
        device._additional_callbacks = {}

    return mocker.patch.object(OpenXRDevice, "__init__", autospec=True, side_effect=initialise)


def test_configuration_selects_dvrk_device_class():
    assert DVRKOpenXRDeviceCfg().class_type is DVRKOpenXRDevice


def test_constructor_requires_exactly_one_dvrk_retargeter(patched_openxr_init):
    with pytest.raises(ValueError, match="exactly one DVRKPSMRetargeter"):
        DVRKOpenXRDevice(DVRKOpenXRDeviceCfg(), retargeters=[])


def test_factory_constructs_dvrk_device_retargeter_and_callbacks(patched_openxr_init):
    pytest.importorskip("isaacteleop.retargeters.DVRK.control")

    def callback():
        pass

    device = create_teleop_device(
        "motion_controllers",
        {
            "motion_controllers": DVRKOpenXRDeviceCfg(
                teleoperation_active_default=False,
                retargeters=[_retargeter_cfg()],
            )
        },
        callbacks={"RESET": callback},
    )

    assert isinstance(device, DVRKOpenXRDevice)
    assert len(device._retargeters) == 1
    assert isinstance(device._retargeters[0], DVRKPSMRetargeter)
    assert device._retargeters[0].session_active is False
    assert device._additional_callbacks["RESET"] is callback


@pytest.mark.parametrize("active_default", (False, True))
def test_constructor_initialises_explicit_session_state(retargeter, patched_openxr_init, mocker, active_default: bool):
    start = mocker.spy(retargeter, "start")
    stop = mocker.spy(retargeter, "stop")

    DVRKOpenXRDevice(
        DVRKOpenXRDeviceCfg(teleoperation_active_default=active_default),
        retargeters=[retargeter],
    )

    assert retargeter.session_active is active_default
    assert start.call_count == int(active_default)
    assert stop.call_count == int(not active_default)


def test_start_stop_and_reset_preserve_application_callbacks_and_order(retargeter, patched_openxr_init, mocker):
    events: list[str] = []
    mocker.patch.object(retargeter, "start", side_effect=lambda: events.append("internal start"))
    mocker.patch.object(retargeter, "stop", side_effect=lambda: events.append("internal stop"))
    mocker.patch.object(retargeter, "reset", side_effect=lambda: events.append("internal reset"))
    mocker.patch.object(OpenXRDevice, "reset", autospec=True, side_effect=lambda _: events.append("openxr reset"))
    device = DVRKOpenXRDevice(
        DVRKOpenXRDeviceCfg(teleoperation_active_default=False),
        retargeters=[retargeter],
    )
    events.clear()  # Ignore the constructor's explicit inactive transition.
    device.add_callback("START", lambda: events.append("application start"))
    device.add_callback("STOP", lambda: events.append("application stop"))
    device.add_callback("RESET", lambda: events.append("application reset"))

    device._on_teleop_command(SimpleNamespace(payload={"message": "start"}))
    assert events == ["internal start", "application start"]

    events.clear()
    device._on_teleop_command(SimpleNamespace(payload={"message": "stop"}))
    assert events == ["internal stop", "application stop"]

    events.clear()
    device._on_teleop_command(SimpleNamespace(payload={"message": "reset"}))
    assert events == ["application reset", "internal reset", "openxr reset"]


def test_direct_reset_forwards_once_before_openxr_reset(retargeter, patched_openxr_init, mocker):
    events: list[str] = []
    mocker.patch.object(retargeter, "reset", side_effect=lambda: events.append("internal reset"))
    mocker.patch.object(OpenXRDevice, "reset", autospec=True, side_effect=lambda _: events.append("openxr reset"))
    device = DVRKOpenXRDevice(DVRKOpenXRDeviceCfg(), retargeters=[retargeter])

    device.reset()

    assert events == ["internal reset", "openxr reset"]


def test_task_enumeration_without_isaacteleop_only_errors_when_factory_constructs_device(monkeypatch):
    control_imports: list[str] = []
    real_import = builtins.__import__

    def import_without_control(name, *args, **kwargs):
        if name == "isaacteleop.retargeters.DVRK.control":
            control_imports.append(name)
            raise ImportError("dependency deliberately absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_control)
    tasks_package = importlib.import_module("isaaclab_tasks")
    if getattr(tasks_package, "__file__", None) is None:
        pytest.skip("isaaclab_tasks extension is unavailable in this simulator lane")
    gym = importlib.import_module("gymnasium")
    task_registration = importlib.import_module("isaaclab_tasks.manager_based.manipulation.needle_pass.config.dvrk")
    if "Isaac-NeedlePass-dVRK-IK-Abs-v0" not in gym.registry:
        importlib.reload(task_registration)

    assert tasks_package.__name__ == "isaaclab_tasks"
    assert gym.spec("Isaac-NeedlePass-dVRK-IK-Abs-v0").entry_point == "isaaclab.envs:ManagerBasedRLEnv"
    assert control_imports == []

    device_configs = {
        "motion_controllers": DVRKOpenXRDeviceCfg(
            teleoperation_active_default=False,
            retargeters=[_retargeter_cfg()],
        )
    }
    with pytest.raises(ModuleNotFoundError, match="Isaac Teleop.*PR 769"):
        create_teleop_device("motion_controllers", device_configs)
    assert control_imports == ["isaacteleop.retargeters.DVRK.control"]
