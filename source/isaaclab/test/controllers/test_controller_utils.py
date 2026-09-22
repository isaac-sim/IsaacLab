# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test cases for Isaac Lab controller utilities."""

import os
import sys
import xml.etree.ElementTree as ET
from types import ModuleType

import pytest
import torch

import isaaclab.controllers.utils as controller_utils
from isaaclab.controllers.utils import (
    change_revolute_to_fixed,
    change_revolute_to_fixed_regex,
    convert_usd_to_urdf,
    resolve_rmpflow_path,
)
from isaaclab.utils.io.torchscript import load_torchscript_model

pytestmark = pytest.mark.unit

_REVOLUTE_JOINTS = ("base_to_shoulder", "shoulder_to_elbow", "elbow_to_wrist")
_FIXED_JOINT = "wrist_to_gripper"


def _joint_tag(name: str, joint_type: str) -> str:
    return f'<joint name="{name}" type="{joint_type}">'


_URDF_CONTENT = "\n".join(
    [
        '<?xml version="1.0"?>',
        '<robot name="test_robot">',
        '    <link name="base_link"/>',
        *[
            f"    {_joint_tag(name, 'revolute')}\n"
            f'        <parent link="link_{index}"/>\n'
            f'        <child link="link_{index + 1}"/>\n'
            '        <limit lower="-3.14" upper="3.14" effort="100" velocity="1"/>\n'
            "    </joint>"
            for index, name in enumerate(_REVOLUTE_JOINTS)
        ],
        f"    {_joint_tag(_FIXED_JOINT, 'fixed')}",
        '        <parent link="link_3"/>',
        '        <child link="gripper_link"/>',
        "    </joint>",
        "</robot>",
    ]
)


@pytest.fixture
def urdf_file(tmp_path):
    path = tmp_path / "test_robot.urdf"
    path.write_text(_URDF_CONTENT)
    return str(path)


@pytest.mark.parametrize(
    ("convert", "fixed_joints", "expected_fixed"),
    [
        (change_revolute_to_fixed, ["shoulder_to_elbow"], {"shoulder_to_elbow"}),
        (
            change_revolute_to_fixed,
            ["base_to_shoulder", "missing", "elbow_to_wrist"],
            {"base_to_shoulder", "elbow_to_wrist"},
        ),
        (change_revolute_to_fixed, [_FIXED_JOINT], set()),
        (change_revolute_to_fixed, [], set()),
        (change_revolute_to_fixed_regex, ["shoulder_to_elbow"], {"shoulder_to_elbow"}),
        (change_revolute_to_fixed_regex, [r"^base.*", r".*wrist$"], {"base_to_shoulder", "elbow_to_wrist"}),
        (change_revolute_to_fixed_regex, [r".*TO.*"], set()),
        (change_revolute_to_fixed_regex, [r".*to.*"], set(_REVOLUTE_JOINTS)),
        (change_revolute_to_fixed_regex, [r".*gripper.*"], set()),
    ],
)
def test_change_revolute_to_fixed(urdf_file, convert, fixed_joints, expected_fixed):
    """Only the selected revolute joints become fixed; everything else in the file is preserved."""
    convert(urdf_file, fixed_joints)

    with open(urdf_file) as file:
        content = file.read()

    expected = _URDF_CONTENT
    for name in expected_fixed:
        expected = expected.replace(_joint_tag(name, "revolute"), _joint_tag(name, "fixed"))
    assert content == expected


@pytest.mark.parametrize("convert", [change_revolute_to_fixed, change_revolute_to_fixed_regex])
def test_change_revolute_to_fixed_missing_file(tmp_path, convert):
    with pytest.raises(FileNotFoundError):
        convert(str(tmp_path / "missing.urdf"), ["base_to_shoulder"])


def test_convert_usd_to_urdf_uses_isaacsim_exporter(monkeypatch, tmp_path):
    """The exporter is configured for Pinocchio and its infinite limits are sanitized."""
    enabled_extensions = []
    monkeypatch.setattr(controller_utils, "enable_extension", enabled_extensions.append)
    converter_args = {}

    class FakeUsdToUrdfConverter:
        def __init__(self, **kwargs):
            converter_args.update(kwargs)

        def convert(self, output_path):
            with open(output_path, "w") as file:
                file.write(
                    '<robot><joint name="j" type="revolute">'
                    '<limit lower="-inf" upper="inf" velocity="inf"/></joint></robot>'
                )

    for module_name in ("isaacsim.asset", "isaacsim.asset.exporter", "isaacsim.asset.exporter.urdf"):
        module = ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)
    sys.modules["isaacsim.asset.exporter.urdf"].UsdToUrdfConverter = FakeUsdToUrdfConverter

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
    assert limit.attrib == {"lower": "-inf", "upper": "inf", "effort": "0.", "velocity": "0."}


def test_resolve_rmpflow_path_uses_extension_directory(monkeypatch, tmp_path):
    """Sentinel paths resolve from the extension directory without enabling the Kit extension."""
    monkeypatch.setattr(controller_utils, "enable_extension", lambda name: pytest.fail(f"enabled {name}"))
    config_dir = tmp_path / "extsDeprecated" / "isaacsim.robot_motion.motion_generation" / "motion_policy_configs"
    (config_dir / "franka" / "rmpflow").mkdir(parents=True)
    monkeypatch.setenv("ISAAC_PATH", str(tmp_path))

    resolved_path = resolve_rmpflow_path("rmpflow_ext:motion_policy_configs/franka/rmpflow/config.yaml")

    assert resolved_path == str(config_dir / "franka" / "rmpflow" / "config.yaml")
    assert resolve_rmpflow_path("/absolute/config.yaml") == "/absolute/config.yaml"


def test_franka_rmpflow_config_resolves_from_installed_isaacsim(monkeypatch):
    """The shipped Franka RMPFlow config resolves to an existing file from the installed Isaac Sim package."""
    from isaaclab.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG

    monkeypatch.setattr(controller_utils, "enable_extension", lambda name: pytest.fail(f"enabled {name}"))
    monkeypatch.delenv("ISAAC_PATH", raising=False)

    resolved_path = resolve_rmpflow_path(FRANKA_RMPFLOW_CFG.config_file)

    assert resolved_path.startswith("/")
    assert resolved_path.endswith(FRANKA_RMPFLOW_CFG.config_file.split(":", 1)[1])
    assert os.path.exists(resolved_path)


@pytest.fixture(scope="module")
def torchscript_model_path(tmp_path_factory):
    model = torch.jit.script(torch.nn.Linear(3, 2))
    path = tmp_path_factory.mktemp("torchscript") / "model.pt"
    torch.jit.save(model, str(path))
    return str(path)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")),
    ],
)
def test_load_torchscript_model(torchscript_model_path, device):
    model = load_torchscript_model(torchscript_model_path, device=device)

    assert isinstance(model, torch.nn.Module)
    assert not model.training
    output = model(torch.ones(1, 3, device=device))
    assert output.shape == (1, 2)
    assert output.device.type == device


def test_load_torchscript_model_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_torchscript_model(str(tmp_path / "missing.pt"))
    # invalid archives are reported by returning None instead of raising
    invalid = tmp_path / "invalid.pt"
    invalid.write_bytes(b"not a torchscript archive")
    assert load_torchscript_model(str(invalid)) is None
    empty = tmp_path / "empty.pt"
    empty.touch()
    assert load_torchscript_model(str(empty)) is None
