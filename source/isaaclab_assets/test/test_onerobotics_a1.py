# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Source-fidelity and portability tests for the OneRobotics A1 assets."""

import hashlib
import inspect
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from isaaclab_assets import ONEROBOTICS_A1_BIMANUAL_CFG, ONEROBOTICS_A1_CFG, ONEROBOTICS_A1_UNIMANUAL_CFG
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

_BIMANUAL_LINK_NAMES = [
    "base_link",
    *[f"Link_r{index}" for index in range(8)],
    *[f"Link_l{index}" for index in range(8)],
]
_BIMANUAL_JOINT_NAMES = [
    "joint_r0",
    *[f"joint_r{index}" for index in range(1, 8)],
    "joint_l0",
    *[f"joint_l{index}" for index in range(1, 8)],
]
_BIMANUAL_ACTIVE_JOINT_NAMES = [
    *[f"joint_r{index}" for index in range(1, 8)],
    *[f"joint_l{index}" for index in range(1, 8)],
]
_BIMANUAL_JOINT_LIMITS = [
    (-1.04, 3.14),
    (-3.14, 0.26),
    (-2.75, 2.75),
    (-1.91, 1.91),
    (-2.75, 2.75),
    (-1.57, 1.57),
    (-2.75, 2.75),
    (-3.14, 1.04),
    (-0.26, 3.14),
    (-2.75, 2.75),
    (-1.91, 1.91),
    (-2.75, 2.75),
    (-1.57, 1.57),
    (-2.75, 2.75),
]
_BIMANUAL_LINK_MASSES = {
    "base_link": 1.71792346199407,
    "Link_r0": 0.771006476050805,
    "Link_r1": 0.699008573524528,
    "Link_r2": 0.678339582027986,
    "Link_r3": 0.667439620002479,
    "Link_r4": 0.507385590081131,
    "Link_r5": 0.533144427415355,
    "Link_r6": 0.519324512857312,
    "Link_r7": 0.0177506448635755,
    "Link_l0": 0.771130628935356,
    "Link_l1": 0.699012569536138,
    "Link_l2": 0.678292070540836,
    "Link_l3": 0.667439620457598,
    "Link_l4": 0.507385586657591,
    "Link_l5": 0.533144415193751,
    "Link_l6": 0.519324489008105,
    "Link_l7": 0.0177506448635751,
}
_BIMANUAL_ASSET_SHA256 = {
    "a1_bimanual_stand.urdf": "4719154dd2395498a646f60a468ccdd3c72e9f9754478c4df74eb495e605ef9d",
    "base_link.STL": "9ccfdff7ec77b26d294fa597fa20116b3d6745fbdb2faa22f0886f8f60dfa331",
    "Link_l0.STL": "7500d6bdfbbf3f7614ee750131d30527adf6b15d6fca8ccc0da297af0e9411fa",
    "Link_l1.STL": "e46d1540441ece5c1b98de4dd91778f4a19278f736e3b0a83698cca1616861b6",
    "Link_l2.STL": "6e98b5242edcd3b403b28569de4d29f1ee42d9903cfffc0f7f47459c34117d95",
    "Link_l3.STL": "5a2470eb295db4d157b9d1473b93a018353146e0690d343a7cfc5c839b47bf58",
    "Link_l4.STL": "25632bdaead8a7c240f89e59708d29e3759b8a59ca40a8e51d6396502e0850d6",
    "Link_l5.STL": "3118c3ffa2c32efc1da7e86391063f692ba67beaa9a1b92802eb878023dce099",
    "Link_l6.STL": "2db7a7493390fb7ea64af9745d879cb9251d73038c440304cdbe4a75b9f6a942",
    "Link_l7.STL": "55daadcea869470cf395d8da89f696f9a45062f084125112aac161e25b78d9e6",
    "Link_r0.STL": "d0b89cb9a695e3221ba75383e6f296108157743ed898c45e753b4e2810c5df5f",
    "Link_r1.STL": "ff7aaf297c2e1a29397962eda08b076a7cb6c2a345f97a3b7e1560efc5cfe718",
    "Link_r2.STL": "fbf5d3d142ed7500cb46d0d405a304dab584b17afa2d5e819337912f28abd394",
    "Link_r3.STL": "596eb5c561896a8d977b8ebbed0faf036e3ed102650e12f47c0b534e873784ab",
    "Link_r4.STL": "f22bfef640bb741e8dd96075bd508206bf8f71888dcde25d8474aa3eb98811f7",
    "Link_r5.STL": "ffb18138033907e33c9d5abbbeb0405d63c16b5b9e7c45ca9d6244986e57885e",
    "Link_r6.STL": "7bec003f813fe7e8685ba68940078d21e56badebfe304f0b0524c13fe406e255",
    "Link_r7.STL": "95e5017e0c308196c0a981a08c26e5d1f8b298affeae675a0bf3316916779f61",
}


@pytest.fixture(scope="module")
def unimanual_urdf() -> tuple[Path, ET.Element]:
    """Return the resolved review-stage right-arm URDF and parsed root."""
    urdf_path = Path(ONEROBOTICS_A1_UNIMANUAL_CFG.spawn.asset_path)
    return urdf_path, ET.parse(urdf_path).getroot()


@pytest.fixture(scope="module")
def bimanual_urdf() -> tuple[Path, ET.Element]:
    """Return the resolved review-stage bimanual URDF and parsed root."""
    urdf_path = Path(ONEROBOTICS_A1_BIMANUAL_CFG.spawn.asset_path)
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
    assert "004905e528bdbf26d00b9826c64741c2a48a1089" in source
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


def test_bimanual_topology_limits_and_portability(bimanual_urdf: tuple[Path, ET.Element]):
    """The source asset is one portable, fixed-stand, 14-joint bimanual tree."""
    urdf_path, root = bimanual_urdf
    links = root.findall("link")
    joints = root.findall("joint")
    active_joints = [joint for joint in joints if joint.attrib["type"] != "fixed"]

    assert root.attrib["name"] == "A1_URDF"
    assert [link.attrib["name"] for link in links] == _BIMANUAL_LINK_NAMES
    assert [joint.attrib["name"] for joint in joints] == _BIMANUAL_JOINT_NAMES
    assert [joint.attrib["name"] for joint in active_joints] == _BIMANUAL_ACTIVE_JOINT_NAMES
    assert [joint.attrib["name"] for joint in joints if joint.attrib["type"] == "fixed"] == [
        "joint_r0",
        "joint_l0",
    ]
    assert len(links) == 17
    assert len(joints) == 16
    assert len(active_joints) == 14
    assert not any(re.search("gripper|finger", name, re.IGNORECASE) for name in _BIMANUAL_JOINT_NAMES)

    link_names = set(_BIMANUAL_LINK_NAMES)
    child_links: set[str] = set()
    children_by_parent: dict[str, list[str]] = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        assert parent in link_names and child in link_names
        assert child not in child_links
        child_links.add(child)
        children_by_parent.setdefault(parent, []).append(child)
    assert link_names - child_links == {"base_link"}

    reachable = {"base_link"}
    pending = ["base_link"]
    while pending:
        parent = pending.pop()
        for child in children_by_parent.get(parent, []):
            assert child not in reachable
            reachable.add(child)
            pending.append(child)
    assert reachable == link_names

    for joint, expected_limit in zip(active_joints, _BIMANUAL_JOINT_LIMITS):
        limit = joint.find("limit")
        axis = joint.find("axis")
        assert limit is not None and axis is not None
        assert (float(limit.attrib["lower"]), float(limit.attrib["upper"])) == pytest.approx(expected_limit)
        # The CAD export uses zero placeholders; the Isaac Lab overlay below supplies hardware limits.
        assert float(limit.attrib["effort"]) == 0.0
        assert float(limit.attrib["velocity"]) == 0.0
        axis_xyz = [float(value) for value in axis.attrib["xyz"].split()]
        assert math.sqrt(sum(value**2 for value in axis_xyz)) == pytest.approx(1.0)

    source_text = urdf_path.read_text()
    assert "/home/" not in source_text
    assert "package://" not in source_text
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\]", source_text)


def test_bimanual_mass_inertia_collision_and_mesh_fidelity(bimanual_urdf: tuple[Path, ET.Element]):
    """All 17 bodies retain finite physical data and their audited source meshes."""
    urdf_path, root = bimanual_urdf
    mesh_paths: set[Path] = set()
    measured_masses: dict[str, float] = {}

    for link in root.findall("link"):
        inertial = link.find("inertial")
        visual_mesh = link.find("visual/geometry/mesh")
        collision_mesh = link.find("collision/geometry/mesh")
        assert inertial is not None and visual_mesh is not None and collision_mesh is not None
        assert visual_mesh.attrib["filename"] == collision_mesh.attrib["filename"]

        mass = float(inertial.find("mass").attrib["value"])
        inertia = {
            name: float(inertial.find("inertia").attrib[name]) for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
        }
        assert mass == pytest.approx(_BIMANUAL_LINK_MASSES[link.attrib["name"]], abs=1.0e-12)
        assert all(math.isfinite(value) for value in (mass, *inertia.values()))

        # Sylvester's criterion verifies that the symmetric inertia tensor is positive definite.
        det_2 = inertia["ixx"] * inertia["iyy"] - inertia["ixy"] ** 2
        det_3 = (
            inertia["ixx"] * (inertia["iyy"] * inertia["izz"] - inertia["iyz"] ** 2)
            - inertia["ixy"] * (inertia["ixy"] * inertia["izz"] - inertia["iyz"] * inertia["ixz"])
            + inertia["ixz"] * (inertia["ixy"] * inertia["iyz"] - inertia["iyy"] * inertia["ixz"])
        )
        assert inertia["ixx"] > 0.0 and det_2 > 0.0 and det_3 > 0.0
        measured_masses[link.attrib["name"]] = mass

        mesh_reference = Path(visual_mesh.attrib["filename"])
        assert not mesh_reference.is_absolute()
        mesh_path = (urdf_path.parent / mesh_reference).resolve()
        mesh_path.relative_to(urdf_path.parent.resolve())
        assert mesh_path.is_file()
        assert mesh_path.name == f"{link.attrib['name']}.STL"
        mesh_paths.add(mesh_path)

    assert len(mesh_paths) == 17
    assert sum(measured_masses.values()) == pytest.approx(10.504802914010192, abs=1.0e-12)
    assert sum(measured_masses[f"Link_r{index}"] for index in range(8)) == pytest.approx(4.393399426823172, abs=1.0e-12)
    assert sum(measured_masses[f"Link_l{index}"] for index in range(8)) == pytest.approx(4.39348002519295, abs=1.0e-12)
    assert _sha256(urdf_path) == _BIMANUAL_ASSET_SHA256[urdf_path.name]
    assert {path.name for path in mesh_paths} == set(_BIMANUAL_ASSET_SHA256) - {urdf_path.name}
    for mesh_path in mesh_paths:
        assert _sha256(mesh_path) == _BIMANUAL_ASSET_SHA256[mesh_path.name]


def test_bimanual_actuator_overlay_matches_confirmed_hardware():
    """Both arms have source-zero state and exact, non-overlapping hardware overlays."""
    cfg = ONEROBOTICS_A1_BIMANUAL_CFG
    assert cfg.spawn.fix_base
    assert not cfg.spawn.merge_fixed_joints
    assert cfg.spawn.self_collision
    assert not cfg.spawn.activate_contact_sensors
    assert cfg.spawn.rigid_props.disable_gravity is False
    assert cfg.spawn.rigid_props.max_depenetration_velocity == 5.0
    assert cfg.spawn.articulation_props.enabled_self_collisions
    assert cfg.spawn.articulation_props.solver_position_iteration_count == 8
    assert cfg.spawn.articulation_props.solver_velocity_iteration_count == 0
    assert cfg.spawn.joint_drive.gains.stiffness == {"joint_[rl][1-4]": 150.0, "joint_[rl][5-7]": 40.0}
    assert cfg.spawn.joint_drive.gains.damping == {"joint_[rl][1-4]": 4.0, "joint_[rl][5-7]": 1.0}
    assert cfg.init_state.joint_pos == {"joint_[rl][1-7]": 0.0}
    assert cfg.soft_joint_pos_limit_factor == 1.0

    motor_4340 = cfg.actuators["arm_4340"]
    assert motor_4340.joint_names_expr == ["joint_[rl][1-3]"]
    assert (motor_4340.actuator_effort_limit, motor_4340.joint_effort_limit) == (15.0, 26.859)
    assert motor_4340.actuator_velocity_limit == pytest.approx(2.6179938779914944)
    assert motor_4340.joint_velocity_limit == pytest.approx(2.6179938779914944)
    assert (motor_4340.stiffness, motor_4340.damping) == (150.0, 4.0)
    assert motor_4340.armature == pytest.approx(0.050927514873)

    motor_4310 = cfg.actuators["arm_4310"]
    assert motor_4310.joint_names_expr == ["joint_[rl][4-7]"]
    assert (motor_4310.actuator_effort_limit, motor_4310.joint_effort_limit) == (3.0, 5.975)
    assert motor_4310.actuator_velocity_limit == pytest.approx(12.566370614359172)
    assert motor_4310.joint_velocity_limit == pytest.approx(12.566370614359172)
    assert motor_4310.stiffness == {"joint_[rl]4": 150.0, "joint_[rl][5-7]": 40.0}
    assert motor_4310.damping == {"joint_[rl]4": 4.0, "joint_[rl][5-7]": 1.0}
    assert motor_4310.armature == pytest.approx(0.002193)

    actuator_matches = {
        joint_name: [
            actuator_name
            for actuator_name, actuator in cfg.actuators.items()
            if any(re.fullmatch(expression, joint_name) for expression in actuator.joint_names_expr)
        ]
        for joint_name in _BIMANUAL_JOINT_NAMES
    }
    assert all(len(actuator_matches[joint_name]) == 1 for joint_name in _BIMANUAL_ACTIVE_JOINT_NAMES)
    assert actuator_matches["joint_r0"] == []
    assert actuator_matches["joint_l0"] == []
    assert sum(matches == ["arm_4340"] for matches in actuator_matches.values()) == 6
    assert sum(matches == ["arm_4310"] for matches in actuator_matches.values()) == 8

    for joint_name in _BIMANUAL_ACTIVE_JOINT_NAMES:
        joint_index = int(joint_name[-1])
        expected_stiffness = 150.0 if joint_index <= 4 else 40.0
        expected_damping = 4.0 if joint_index <= 4 else 1.0
        stiffness_matches = [
            value
            for expression, value in cfg.spawn.joint_drive.gains.stiffness.items()
            if re.fullmatch(expression, joint_name)
        ]
        damping_matches = [
            value
            for expression, value in cfg.spawn.joint_drive.gains.damping.items()
            if re.fullmatch(expression, joint_name)
        ]
        init_matches = [
            value for expression, value in cfg.init_state.joint_pos.items() if re.fullmatch(expression, joint_name)
        ]
        assert stiffness_matches == [expected_stiffness]
        assert damping_matches == [expected_damping]
        assert init_matches == [0.0]
