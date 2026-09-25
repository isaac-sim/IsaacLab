# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for Isaac Lab controller utilities."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from types import ModuleType

import pytest
import torch

import omni.kit.app

from isaaclab.controllers.utils import (
    change_revolute_to_fixed,
    change_revolute_to_fixed_regex,
    convert_usd_to_urdf,
    resolve_rmpflow_path,
)
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.io.torchscript import load_torchscript_model

pytestmark = pytest.mark.integration


@pytest.fixture
def mock_urdf_content():
    """Create mock URDF content for testing."""
    return """<?xml version="1.0"?>
<robot name="test_robot">
    <link name="base_link">
        <visual>
            <geometry>
                <box size="0.1 0.1 0.1"/>
            </geometry>
        </visual>
    </link>

    <joint name="base_to_shoulder" type="revolute">
        <parent link="base_link"/>
        <child link="shoulder_link"/>
        <origin xyz="0 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="shoulder_link">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.2"/>
            </geometry>
        </visual>
    </link>

    <joint name="shoulder_to_elbow" type="revolute">
        <parent link="shoulder_link"/>
        <child link="elbow_link"/>
        <origin xyz="0 0 0.2" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="elbow_link">
        <visual>
            <geometry>
                <cylinder radius="0.04" length="0.15"/>
            </geometry>
        </visual>
    </link>

    <joint name="elbow_to_wrist" type="revolute">
        <parent link="elbow_link"/>
        <child link="wrist_link"/>
        <origin xyz="0 0 0.15" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>
    </joint>

    <link name="wrist_link">
        <visual>
            <geometry>
                <sphere radius="0.03"/>
            </geometry>
        </visual>
    </link>

    <joint name="wrist_to_gripper" type="fixed">
        <parent link="wrist_link"/>
        <child link="gripper_link"/>
        <origin xyz="0 0 0.05" rpy="0 0 0"/>
    </joint>

    <link name="gripper_link">
        <visual>
            <geometry>
                <box size="0.02 0.02 0.02"/>
            </geometry>
        </visual>
    </link>
</robot>"""


@pytest.fixture
def test_urdf_file(mock_urdf_content):
    """Create a temporary URDF file for testing."""
    # Create a temporary directory for test files
    test_dir = tempfile.mkdtemp()

    # Create the test URDF file
    test_urdf_path = os.path.join(test_dir, "test_robot.urdf")
    with open(test_urdf_path, "w") as f:
        f.write(mock_urdf_content)

    yield test_urdf_path

    # Clean up the temporary directory and all its contents
    import shutil

    shutil.rmtree(test_dir)


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


# =============================================================================
# Test cases for change_revolute_to_fixed and change_revolute_to_fixed_regex functions
# =============================================================================


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


@pytest.mark.parametrize("convert", [change_revolute_to_fixed, change_revolute_to_fixed_regex])
def test_single_joint_conversion(convert, test_urdf_file, mock_urdf_content):
    """Converting one revolute joint changes only its type and preserves the rest of the file."""
    convert(test_urdf_file, ["shoulder_to_elbow"])

    expected = mock_urdf_content.replace(
        '<joint name="shoulder_to_elbow" type="revolute">', '<joint name="shoulder_to_elbow" type="fixed">'
    )
    assert expected != mock_urdf_content
    assert _read(test_urdf_file) == expected


def test_mixed_existent_and_non_existent_joints(test_urdf_file, mock_urdf_content):
    """Test converting a mix of existent and non-existent joints."""
    # Try to convert both existent and non-existent joints
    fixed_joints = ["base_to_shoulder", "non_existent_joint", "elbow_to_wrist"]
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    modified_content = _read(test_urdf_file)

    # Check that existent joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content
    assert '<joint name="base_to_shoulder" type="revolute">' not in modified_content
    assert '<joint name="elbow_to_wrist" type="revolute">' not in modified_content

    # Check that the joint between them remains unchanged
    assert '<joint name="shoulder_to_elbow" type="revolute">' in modified_content

    # Check that non-existent joint didn't cause issues
    assert '<joint name="non_existent_joint" type="fixed">' not in modified_content


@pytest.mark.parametrize("fixed_joints", [["non_existent_joint"], [], ["wrist_to_gripper"]])
def test_non_existent_joint(fixed_joints, test_urdf_file, mock_urdf_content):
    """Missing, empty, and already-fixed joint lists leave the file unchanged."""
    change_revolute_to_fixed(test_urdf_file, fixed_joints)

    assert _read(test_urdf_file) == mock_urdf_content


@pytest.mark.parametrize("convert", [change_revolute_to_fixed, change_revolute_to_fixed_regex])
def test_file_not_found(convert, test_urdf_file):
    """Test behavior when URDF file doesn't exist."""
    non_existent_path = os.path.join(os.path.dirname(test_urdf_file), "non_existent.urdf")

    with pytest.raises(FileNotFoundError):
        convert(non_existent_path, ["base_to_shoulder"])


def test_regex_pattern_matching(test_urdf_file, mock_urdf_content):
    """Test converting joints using regex patterns."""
    # Test converting joints that contain "to" in their name
    fixed_joints = [r".*to.*"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    modified_content = _read(test_urdf_file)

    # Check that all joints with "to" in the name were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="shoulder_to_elbow" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that the fixed joint remains unchanged
    assert '<joint name="wrist_to_gripper" type="fixed">' in modified_content


def test_regex_multiple_patterns(test_urdf_file, mock_urdf_content):
    """Test converting joints using multiple regex patterns."""
    # Test converting joints that start with "base" or end with "wrist"
    fixed_joints = [r"^base.*", r".*wrist$"]
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    modified_content = _read(test_urdf_file)

    # Check that matching joints were converted
    assert '<joint name="base_to_shoulder" type="fixed">' in modified_content
    assert '<joint name="elbow_to_wrist" type="fixed">' in modified_content

    # Check that non-matching joints remain unchanged
    assert '<joint name="shoulder_to_elbow" type="revolute">' in modified_content


@pytest.mark.parametrize("fixed_joints", [[r"^nonexistent.*"], [], [r".*TO.*"], [r".*gripper.*"]])
def test_regex_no_matches(fixed_joints, test_urdf_file, mock_urdf_content):
    """Unmatched, empty, case-mismatched, and already-fixed patterns leave the file unchanged."""
    change_revolute_to_fixed_regex(test_urdf_file, fixed_joints)

    assert _read(test_urdf_file) == mock_urdf_content


# =============================================================================
# Test cases for load_torchscript_model function
# =============================================================================


@pytest.fixture
def policy_model_path():
    """Path to the test TorchScript model."""
    _policy_path = f"{ISAACLAB_NUCLEUS_DIR}/Policies/Agile/agile_locomotion.pt"
    return retrieve_file_path(_policy_path)


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def test_load_torchscript_model_success(policy_model_path, device):
    """Test successful loading of a TorchScript model."""
    model = load_torchscript_model(policy_model_path, device=device)

    # Check that model was loaded successfully
    assert model is not None
    assert isinstance(model, torch.nn.Module)

    # Check that model is in evaluation mode
    assert model.training is False


def test_load_torchscript_model_file_not_found():
    """Test behavior when TorchScript model file doesn't exist."""
    non_existent_path = "non_existent_model.pt"

    # Should raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_torchscript_model(non_existent_path)


@pytest.mark.parametrize("content", [b"invalid torchscript content", b""])
def test_load_torchscript_model_invalid_file(content, tmp_path):
    """Invalid or empty TorchScript files are handled gracefully and return None."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(content)

    assert load_torchscript_model(str(model_path)) is None
