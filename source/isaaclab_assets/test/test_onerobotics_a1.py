# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-fidelity and portability tests for the OneRobotics A1 asset configuration."""

import hashlib
import inspect
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaaclab_assets import ONEROBOTICS_A1_CFG
from isaaclab_assets.robots import onerobotics

_JOINT_NAMES = [f"joint{index}-a1_r" for index in range(1, 8)]
_JOINT_LIMITS = [
    (-1.04, 3.14),
    (-3.14, 0.26),
    (-2.75, 2.75),
    (-1.92, 1.92),
    (-2.75, 2.75),
    (-1.57, 1.57),
    (-2.75, 2.75),
]
_HOME_POSITION = [0.0, -0.6, 0.0, 1.0, 0.0, 0.5, 0.0]
_ASSET_SHA256 = {
    "a1_right.urdf": "16b5bdd2e1a4812563854d727ede7b5bd2110b41f77820de053880119a941207",
    "Link_R1.STL": "ce57c22a5b46aed102bc1623cd435d12dbd0f50fc8a3ba089791bf013f98691a",
    "Link_R2.STL": "76b84009a598a72cb180c057b5ba65bcbf0b96800da30de41166d34062d80051",
    "Link_R3.STL": "40e8737ce7a4f8cfbc37ad04c4c6e9f2ba9b0a81a9341cdfb1c3dd46af1ff0c2",
    "Link_R4.STL": "8901305677c45690ab0fc251292135e07ad54a59ba7aeeb89d7eb33620137ef3",
    "Link_R5.STL": "9986ccc5b3554b46b636436f93b1dcf540e745a41504e0902277ba53af9fae48",
    "Link_R6.STL": "2b18f213a95c2606a680e8d35ca9ef20417f78fccee3bac9fe15b83ed75caa05",
    "Link_R7.STL": "f05fa5c4aa376a4c2a5404fad8997872cfa47a433640f9829b9a814cc617e06a",
    "base_link.STL": "4e87134bbbb060353cd775f7cc4ae27c6eb95341d895a2ed1d2cdde76c772237",
}


@pytest.fixture(scope="module")
def urdf() -> tuple[Path, ET.Element]:
    """Return the resolved review-stage URDF path and parsed XML root."""
    urdf_path = Path(ONEROBOTICS_A1_CFG.spawn.asset_path)
    return urdf_path, ET.parse(urdf_path).getroot()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_asset_source_is_portable_and_attributed(urdf: tuple[Path, ET.Element]):
    """The config uses the public review source without a workstation path dependency."""
    urdf_path, _ = urdf
    source = inspect.getsource(onerobotics)

    assert urdf_path.is_file()
    assert "ONEROBOTICS_A1_ASSET_DIR" in source
    assert "https://github.com/katazen/onerobot_h1.git" in source
    assert "CC BY 4.0" in source
    assert "OneRobotics" in source
    assert "/home/" not in source


def test_canonical_model_topology_and_limits(urdf: tuple[Path, ET.Element]):
    """The source asset remains the canonical 7-joint right arm with exact limits."""
    _, root = urdf
    links = root.findall("link")
    joints = root.findall("joint")

    assert root.attrib["name"] == "a1_r_urdf"
    assert [link.attrib["name"] for link in links] == ["base_link", *[f"Link{i}" for i in range(1, 8)]]
    assert [joint.attrib["name"] for joint in joints] == _JOINT_NAMES
    assert [joint.attrib["type"] for joint in joints] == ["revolute"] * 7

    for joint, expected_limit in zip(joints, _JOINT_LIMITS):
        limit = joint.find("limit")
        assert limit is not None
        assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == pytest.approx(expected_limit)
        assert float(limit.attrib["effort"]) == 0.0
        assert float(limit.attrib["velocity"]) == 0.0


def test_mass_inertia_collision_and_mesh_fidelity(urdf: tuple[Path, ET.Element]):
    """Mass, inertia, collision, and mesh declarations are complete and finite."""
    urdf_path, root = urdf
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

    assert total_mass == pytest.approx(2.2889299585050775)
    assert {path.name for path in mesh_paths} == set(_ASSET_SHA256) - {"a1_right.urdf"}
    assert _sha256(urdf_path) == _ASSET_SHA256[urdf_path.name]
    for mesh_path in mesh_paths:
        assert _sha256(mesh_path) == _ASSET_SHA256[mesh_path.name]


def test_articulation_and_actuator_overlay_matches_source():
    """The Isaac Lab overlay preserves the validated home pose and actuator values."""
    cfg = ONEROBOTICS_A1_CFG
    assert cfg.spawn.fix_base
    assert cfg.spawn.self_collision
    assert not cfg.spawn.activate_contact_sensors
    assert cfg.spawn.rigid_props.disable_gravity is False
    assert cfg.spawn.rigid_props.max_depenetration_velocity == 5.0
    assert cfg.spawn.articulation_props.enabled_self_collisions
    assert cfg.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert cfg.spawn.joint_drive.gains.stiffness == {".*joint[1-4].*": 60.0, ".*joint[5-7].*": 30.0}
    assert cfg.spawn.joint_drive.gains.damping == {".*joint[1-4].*": 6.0, ".*joint[5-7].*": 3.0}
    assert list(cfg.init_state.joint_pos.values()) == _HOME_POSITION
    assert cfg.soft_joint_pos_limit_factor == 1.0

    proximal = cfg.actuators["arm_proximal"]
    distal = cfg.actuators["arm_distal"]
    assert (proximal.joint_effort_limit, proximal.joint_velocity_limit) == (30.0, 3.0)
    assert (proximal.stiffness, proximal.damping, proximal.armature) == (60.0, 6.0, 0.05)
    assert (distal.joint_effort_limit, distal.joint_velocity_limit) == (12.0, 3.0)
    assert (distal.stiffness, distal.damping, distal.armature) == (30.0, 3.0, 0.05)
