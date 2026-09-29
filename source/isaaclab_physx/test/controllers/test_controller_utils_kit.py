# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test Isaac Lab controller utilities that resolve Isaac Sim extensions."""

from isaaclab.test.utils import launch_test_simulation

launch_test_simulation()

import os
import sys
import xml.etree.ElementTree as ET
from types import ModuleType

import pytest

import omni.kit.app

from isaaclab.controllers.utils import convert_usd_to_urdf, resolve_rmpflow_path

pytestmark = pytest.mark.integration


def _mock_module(monkeypatch, name: str) -> ModuleType:
    module = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _mock_kit_app(monkeypatch):
    enabled_extensions = []

    class MockExtensionManager:
        def is_extension_enabled(self, _name):
            return _name in enabled_extensions

        def set_extension_enabled_immediate(self, name, enabled):
            if enabled:
                enabled_extensions.append(name)
            return True

        def get_enabled_extension_id(self, name):
            if name in enabled_extensions:
                return name
            return ""

        def get_extension_path(self, extension_id):
            return f"/extensions/{extension_id}"

    class MockApp:
        def get_extension_manager(self):
            return MockExtensionManager()

    monkeypatch.setattr(omni.kit.app, "get_app", MockApp)
    return enabled_extensions


def test_convert_usd_to_urdf_uses_isaacsim_exporter(monkeypatch, tmp_path):
    """Test that USD-to-URDF conversion uses Isaac Sim's URDF exporter."""
    enabled_extensions = _mock_kit_app(monkeypatch)
    for module_name in ("isaacsim.asset", "isaacsim.asset.exporter"):
        _mock_module(monkeypatch, module_name).__path__ = []

    converter_args = {}

    class MockUsdToUrdfConverter:
        def __init__(self, **kwargs):
            converter_args.update(kwargs)

        def convert(self, output_path):
            with open(output_path, "w") as file:
                file.write(
                    '<robot><joint name="j" type="revolute">'
                    '<limit lower="-inf" upper="inf" velocity="inf"/></joint></robot>'
                )

    urdf_module = _mock_module(monkeypatch, "isaacsim.asset.exporter.urdf")
    urdf_module.UsdToUrdfConverter = MockUsdToUrdfConverter

    urdf_path, mesh_path = convert_usd_to_urdf("/assets/gr1.usd", str(tmp_path))

    assert enabled_extensions == ["isaacsim.asset.exporter.urdf"]
    assert converter_args == {
        "stage": "/assets/gr1.usd",
        "root_prim_path": None,
        "mesh_dir_name": "../meshes",
        "mesh_path_prefix": "../meshes/",
        "visualize_collision_meshes": False,
    }
    assert (urdf_path, mesh_path) == (str(tmp_path / "urdf" / "gr1.urdf"), str(tmp_path / "meshes"))
    limit = ET.parse(urdf_path).find("joint/limit")
    assert limit.attrib["lower"] == "-inf"
    assert limit.attrib["upper"] == "inf"
    assert limit.attrib["effort"] == "0."
    assert limit.attrib["velocity"] == "0."


def test_resolve_rmpflow_path_uses_installed_motion_generation_extension(monkeypatch, tmp_path):
    """Test that RMPFlow sentinel paths resolve from the installed extension directory."""
    enabled_extensions = _mock_kit_app(monkeypatch)
    ext_dir = tmp_path / "extsDeprecated" / "isaacsim.robot_motion.motion_generation"
    config_dir = ext_dir / "motion_policy_configs" / "franka" / "rmpflow"
    config_dir.mkdir(parents=True)
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))

    resolved_path = resolve_rmpflow_path("rmpflow_ext:motion_policy_configs/franka/rmpflow/config.yaml")

    assert enabled_extensions == []
    assert resolved_path == str(config_dir / "config.yaml")


def test_franka_rmpflow_config_resolves_motion_generation_path():
    """Test that the trimmed app resolves Franka RMPFlow config files from the disabled extension."""
    from isaaclab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    assert not extension_manager.is_extension_enabled("isaacsim.robot_motion.motion_generation")

    resolved_path = resolve_rmpflow_path(FRANKA_RMPFLOW_CFG.config_file)

    assert os.path.isabs(resolved_path)
    assert os.path.exists(resolved_path)
    assert not extension_manager.is_extension_enabled("isaacsim.robot_motion.motion_generation")
