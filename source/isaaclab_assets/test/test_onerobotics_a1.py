# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-fidelity and portability tests for the OneRobotics A1 assets."""

import hashlib
import inspect
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaaclab_assets import ONEROBOTICS_A1_CFG, ONEROBOTICS_A1_UNIMANUAL_CFG
from isaaclab_assets.robots import onerobotics

_JOINT_NAMES = [f"joint{index}-a1_r" for index in range(1, 8)]
_JOINT_LIMITS = [
    (-1.04, 3.14),
    (-3.14, 0.26),
    (-2.76, 2.76),
    (-1.92, 1.92),
    (-2.23, 2.23),
    (-1.57, 1.57),
    (-2.76, 2.76),
]
_URDF_EFFORT_LIMITS = [15.0] * 3 + [3.0] * 4
_URDF_VELOCITY_LIMITS = [30.0] * 7
_HOME_POSITION = [0.0, -0.6, 0.0, 1.0, 0.0, 0.5, 0.0]
_ASSET_SHA256 = {
    "a1_r.urdf": "6c1ea095f90018bd642656af206dd42282c09f6ad2d14be0c9f550e00fa9ebf7",
    "Link1.STL": "d54268edba96a905bb4e0d39614233d822eb447ff5f2eedb2047d4308c1ecfa6",
    "Link2.STL": "742ed9c8a22f857e9adf97fde3fb91ccfed97f99e6406b9777618c3f79d11441",
    "Link3.STL": "a184d61f04bf25fcdf3b6982978562cfd74c61a25f3746b5863fd0961f613e71",
    "Link4.STL": "7a7eabd15b78fa2168759dc3aa261dab1f898433f5bca08f74170681d1238daf",
    "Link5.STL": "75fbe9a5eb0b30f12dc7c77a392d8c567f44625c0fd2a0d96e23890b103ceeb6",
    "Link6.STL": "265449ea87514fd30e9a9810cc25bea20bdb426e78c2485508577f9d0d0d73d2",
    "Link7.STL": "2579284183b7fb457134c9271198575d89c42fa33c8cdbdc9a0a3d9edad2d3ac",
    "base_link.STL": "27b1d1796af11757a3aec995b77f933cd25329e05f3ab5a8845d61813dd6e551",
}


@pytest.fixture(scope="module")
def unimanual_urdf() -> tuple[Path, ET.Element]:
    """Return the resolved review-stage right-arm URDF and parsed root."""
    urdf_path = Path(ONEROBOTICS_A1_UNIMANUAL_CFG.spawn.asset_path)
    return urdf_path, ET.parse(urdf_path).getroot()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asset_source_is_portable_attributed_and_aliased(unimanual_urdf: tuple[Path, ET.Element]):
    """The public review source has attribution, a local override, and a stable alias."""
    urdf_path, _ = unimanual_urdf
    source = inspect.getsource(onerobotics)

    assert urdf_path.is_file()
    assert ONEROBOTICS_A1_CFG is ONEROBOTICS_A1_UNIMANUAL_CFG
    assert "ONEROBOTICS_A1_ASSET_DIR" in source
    assert "https://github.com/katazen/onerobot_h1.git" in source
    assert "fe8df949c4c8c891e17a7c102b255db40af28df9" in source
    assert "CC BY 4.0" in source
    assert "OneRobotics" in source
    assert "/home/" not in source


def test_unimanual_topology_and_limits(unimanual_urdf: tuple[Path, ET.Element]):
    """The source asset is the current fixed-base, 7-joint A1 right arm."""
    _, root = unimanual_urdf
    links = root.findall("link")
    joints = root.findall("joint")

    assert root.attrib["name"] == "a1_r_urdf"
    assert [link.attrib["name"] for link in links] == ["base_link", *[f"Link{i}" for i in range(1, 8)]]
    assert [joint.attrib["name"] for joint in joints] == _JOINT_NAMES
    assert [joint.attrib["type"] for joint in joints] == ["revolute"] * 7

    for index, (joint, expected_limit) in enumerate(zip(joints, _JOINT_LIMITS)):
        limit = joint.find("limit")
        assert limit is not None
        assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == pytest.approx(expected_limit)
        assert float(limit.attrib["effort"]) == _URDF_EFFORT_LIMITS[index]
        assert float(limit.attrib["velocity"]) == _URDF_VELOCITY_LIMITS[index]


def test_unimanual_mass_inertia_collision_and_mesh_fidelity(unimanual_urdf: tuple[Path, ET.Element]):
    """Mass, inertia, collision, mesh hashes, and the Link7 limitation are explicit."""
    urdf_path, root = unimanual_urdf
    total_mass = 0.0
    mesh_paths: set[Path] = set()

    for link in root.findall("link"):
        inertial = link.find("inertial")
        assert inertial is not None
        mass = float(inertial.find("mass").attrib["value"])
        inertia = inertial.find("inertia")
        assert inertia is not None
        inertia_values = {name: float(inertia.attrib[name]) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")}
        assert mass > 0.0 and math.isfinite(mass)
        assert all(math.isfinite(value) for value in inertia_values.values())
        assert all(inertia_values[name] > 0.0 for name in ("ixx", "iyy", "izz"))
        total_mass += mass

        visual_mesh = link.find("visual/geometry/mesh")
        collision_mesh = link.find("collision/geometry/mesh")
        assert visual_mesh is not None and collision_mesh is not None
        assert visual_mesh.attrib["filename"] == collision_mesh.attrib["filename"]
        mesh_path = (urdf_path.parent / visual_mesh.attrib["filename"]).resolve()
        assert mesh_path.is_file()
        mesh_paths.add(mesh_path)

    assert total_mass == pytest.approx(2.711257798161839)
    assert {path.name for path in mesh_paths} == set(_ASSET_SHA256) - {"a1_r.urdf"}
    assert _sha256(urdf_path) == _ASSET_SHA256[urdf_path.name]
    for mesh_path in mesh_paths:
        assert _sha256(mesh_path) == _ASSET_SHA256[mesh_path.name]

    source_text = urdf_path.read_text()
    assert "Link7" in source_text
    assert "placeholder" in source_text
    assert "TODO" in source_text


def test_unimanual_actuator_overlay_matches_confirmed_hardware():
    """The Isaac Lab overlay encodes the confirmed motors, gains, and home pose."""
    cfg = ONEROBOTICS_A1_UNIMANUAL_CFG
    assert cfg.spawn.fix_base
    assert cfg.spawn.self_collision
    assert not cfg.spawn.activate_contact_sensors
    assert cfg.spawn.rigid_props.disable_gravity is False
    assert cfg.spawn.rigid_props.max_depenetration_velocity == 5.0
    assert cfg.spawn.articulation_props.enabled_self_collisions
    assert cfg.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert cfg.spawn.joint_drive.gains.stiffness == {".*joint[1-4].*": 150.0, ".*joint[5-7].*": 40.0}
    assert cfg.spawn.joint_drive.gains.damping == {".*joint[1-4].*": 4.0, ".*joint[5-7].*": 1.0}
    assert list(cfg.init_state.joint_pos.values()) == _HOME_POSITION
    assert cfg.soft_joint_pos_limit_factor == 1.0

    assert pytest.approx(2.193e-5) == onerobotics._A1_MOTOR_ROTOR_INERTIA
    assert pytest.approx(48.19) == onerobotics._A1_4340_GEAR_RATIO
    assert pytest.approx(10.0) == onerobotics._A1_4310_GEAR_RATIO
    assert pytest.approx(0.050927514873) == onerobotics._A1_4340_ARMATURE
    assert pytest.approx(0.002193) == onerobotics._A1_4310_ARMATURE
    assert (onerobotics._A1_4340_RATED_EFFORT, onerobotics._A1_4340_PEAK_EFFORT) == (15.0, 26.859)
    assert (onerobotics._A1_4310_RATED_EFFORT, onerobotics._A1_4310_PEAK_EFFORT) == (3.0, 5.975)
    assert pytest.approx(2.6179938779914944) == onerobotics._A1_4340_RATED_SPEED
    assert pytest.approx(12.566370614359172) == onerobotics._A1_4310_RATED_SPEED

    motor_4340 = cfg.actuators["arm_4340"]
    assert motor_4340.joint_names_expr == [".*joint[1-3].*"]
    assert (motor_4340.actuator_effort_limit, motor_4340.joint_effort_limit) == (15.0, 26.859)
    assert motor_4340.actuator_velocity_limit == pytest.approx(2.6179938779914944)
    assert motor_4340.joint_velocity_limit == pytest.approx(2.6179938779914944)
    assert (motor_4340.stiffness, motor_4340.damping) == (150.0, 4.0)
    assert motor_4340.armature == pytest.approx(0.050927514873)

    motor_4310 = cfg.actuators["arm_4310"]
    assert motor_4310.joint_names_expr == [".*joint[4-7].*"]
    assert (motor_4310.actuator_effort_limit, motor_4310.joint_effort_limit) == (3.0, 5.975)
    assert motor_4310.actuator_velocity_limit == pytest.approx(12.566370614359172)
    assert motor_4310.joint_velocity_limit == pytest.approx(12.566370614359172)
    assert motor_4310.stiffness == {".*joint4.*": 150.0, ".*joint[5-7].*": 40.0}
    assert motor_4310.damping == {".*joint4.*": 4.0, ".*joint[5-7].*": 1.0}
    assert motor_4310.armature == pytest.approx(0.002193)
