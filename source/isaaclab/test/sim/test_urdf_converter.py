# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run the converter from the standalone importer wheel when installed; otherwise launch Isaac Sim."""

from isaaclab.app import AppLauncher
from isaaclab.utils.version import standalone_importers_available

# Prefer kit-less; fall back to Kit when the standalone importers are not usable.
_USE_KIT = not standalone_importers_available() and AppLauncher.is_available()
simulation_app = AppLauncher(headless=True).app if _USE_KIT else None

"""Rest everything follows."""

import math
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

if _USE_KIT:
    import omni.kit.app

from pxr import Usd, UsdPhysics

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# The Kit-less container mounts the checkout read-only, so ``usd_dir`` goes under ``tmp_path``.
pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci, pytest.mark.kitless]

_URDF_IMPORTER_EXTENSION = "isaacsim.asset.importer.urdf"

# Portable Franka URDF for the kitless path (the Kit path uses the importer extension's bundled
# ``panda_arm_hand.urdf``). Both expose the same 7 revolute + 2 prismatic joint structure.
_REPO_FRANKA_URDF = os.path.join(
    os.path.dirname(isaaclab.__file__), "controllers", "config", "data", "lula_franka_gen.urdf"
)
_URDF_FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdfs")
# Fixed-joint fixture for the merge tests: 7 links / 6 joints (3 fixed, 1 continuous, 2 prismatic).
_MERGE_JOINTS_URDF = os.path.join(_URDF_FIXTURES, "test_merge_joints.urdf")
# Fixed-joint-only fixture: the importer writes no PhysX data for it, so its "Physics" variant set
# offers no "physx" variant, so requesting it must fail.
_FIXED_ONLY_URDF = os.path.join(_URDF_FIXTURES, "test_fixed_only.urdf")

# Franka Panda: 7 revolute arm joints and 2 prismatic finger joints.
_NUM_ARM_JOINTS, _NUM_FINGER_JOINTS = 7, 2
_DEG = math.pi / 180.0


def _enable_importer_extension() -> str:
    manager = omni.kit.app.get_app().get_extension_manager()
    if not manager.is_extension_enabled(_URDF_IMPORTER_EXTENSION):
        manager.set_extension_enabled_immediate(_URDF_IMPORTER_EXTENSION, True)
    return manager.get_extension_path(manager.get_enabled_extension_id(_URDF_IMPORTER_EXTENSION))


@pytest.fixture
def sim_config(tmp_path):
    """A stage to spawn into and a Franka converter config writing under ``tmp_path``."""
    stage = sim_utils.create_new_stage()
    if _USE_KIT:
        # Kit path: enable the importer extension and use its bundled Franka asset.
        extension_path = _enable_importer_extension()
        asset_path = f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
        sim = SimulationContext(SimulationCfg(dt=0.01))
    else:
        # Kitless path: the converter loads the importer from the standalone wheel. Spawning and
        # inspecting prims needs a USD stage but neither physics nor Kit, so the plain stage above
        # stands in for the simulation context.
        asset_path = _REPO_FRANKA_URDF
        sim = SimpleNamespace(stage=stage)
    config = UrdfConverterCfg(
        asset_path=asset_path,
        fix_base=True,
        force_usd_conversion=True,
        usd_dir=str(tmp_path / "usd"),
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    yield sim, config
    if _USE_KIT:
        sim._disable_app_control_on_stop_handle = True  # prevent timeout
        sim.stop()
        sim.clear_instance()


def _assert_spawnable(sim, usd_path: str, prim_path: str = "/World/Robot") -> None:
    assert os.path.exists(usd_path)
    sim_utils.create_prim(prim_path, usd_path=usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


def _joint_drives(usd_path: str) -> dict[str, tuple[str, str | None, float, float]]:
    """Map joint name to ``(drive instance, drive type, stiffness, damping)`` for revolute and prismatic joints.

    Values are read straight from ``UsdPhysics.DriveAPI`` so the checks do not depend on a running
    physics simulation. USD stores revolute gains per degree, prismatic gains per meter.
    """
    stage = Usd.Stage.Open(usd_path)
    drives = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            instance = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            instance = "linear"
        else:
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, instance)
        drives[prim.GetName()] = (
            instance,
            drive.GetTypeAttr().Get(),
            drive.GetStiffnessAttr().Get(),
            drive.GetDampingAttr().Get(),
        )
    return drives


def _physics_variant(usd_path: str) -> tuple[str, int, int]:
    """Return the authored ``"Physics"`` variant selection plus the composed joint and articulation-root counts.

    Everything is read while the stage is still referenced, since USD objects do not keep it alive.
    """
    stage = Usd.Stage.Open(usd_path)
    prims = list(stage.Traverse())
    selection = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics").GetVariantSelection()
    joints = sum(1 for prim in prims if prim.IsA(UsdPhysics.Joint))
    roots = sum(1 for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI))
    return selection, joints, roots


def test_lazy_conversion_cache(sim_config):
    """Conversion is skipped for an unchanged asset and config, and re-run when the config changes."""
    sim, config = sim_config
    config.force_usd_conversion = False

    converter = UrdfConverter(config)
    created = os.stat(converter.usd_path).st_mtime_ns
    _assert_spawnable(sim, converter.usd_path)
    stage = Usd.Stage.Open(converter.usd_path)
    assert any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in stage.Traverse())

    assert os.stat(UrdfConverter(config).usd_path).st_mtime_ns == created

    config.fix_base = not config.fix_base
    assert os.stat(UrdfConverter(config).usd_path).st_mtime_ns != created


@pytest.mark.parametrize(
    ("drive_type", "target_type", "stiffness", "damping", "expected"),
    [
        # uniform position gains: revolute joints convert to per-degree units, prismatic stay per meter
        pytest.param(
            None,
            "position",
            42.0,
            4.2,
            lambda kind: (None, 42.0 * _DEG, 4.2 * _DEG) if kind == "angular" else (None, 42.0, 4.2),
            id="uniform_position",
        ),
        pytest.param(
            "acceleration",
            None,
            100.0,
            10.0,
            lambda kind: ("acceleration", None, None),
            id="acceleration",
        ),
        pytest.param(None, "none", None, None, lambda kind: (None, 0.0, 0.0), id="target_none_zeros_gains"),
        pytest.param(
            {"panda_joint[1-7]": "acceleration", "panda_finger": "force"},
            "position",
            {"panda_joint[1-7]": 100.0, "panda_finger": 200.0},
            {"panda_joint[1-7]": 10.0, "panda_finger": 20.0},
            lambda kind: ("acceleration", 100.0 * _DEG, 10.0 * _DEG) if kind == "angular" else ("force", 200.0, 20.0),
            id="per_joint_dicts",
        ),
    ],
)
def test_joint_drive_overrides(sim_config, drive_type, target_type, stiffness, damping, expected):
    """``JointDriveCfg`` settings, uniform or per joint pattern, land in every joint's ``DriveAPI``.

    ``expected(kind)`` yields ``(drive type, stiffness, damping)``; ``None`` skips that check.
    """
    sim, config = sim_config
    if drive_type is not None:
        config.joint_drive.drive_type = drive_type
    if target_type is not None:
        config.joint_drive.target_type = target_type
    config.joint_drive.gains.stiffness = stiffness
    config.joint_drive.gains.damping = damping

    drives = _joint_drives(UrdfConverter(config).usd_path)

    kinds = [kind for kind, *_ in drives.values()]
    assert kinds.count("angular") == _NUM_ARM_JOINTS
    assert kinds.count("linear") == _NUM_FINGER_JOINTS
    for name, (kind, actual_type, actual_stiffness, actual_damping) in drives.items():
        expected_type, expected_stiffness, expected_damping = expected(kind)
        if expected_type is not None:
            assert actual_type == expected_type, name
        if expected_stiffness is not None:
            assert actual_stiffness == pytest.approx(expected_stiffness, abs=1e-3), name
            assert actual_damping == pytest.approx(expected_damping, abs=1e-3), name


@pytest.mark.parametrize("fix_base", [True, False])
def test_fix_base(sim_config, fix_base):
    """``fix_base`` adds a fixed joint anchoring a rigid body link, and only then."""
    _, config = sim_config
    config.fix_base = fix_base

    stage = Usd.Stage.Open(UrdfConverter(config).usd_path)

    fixed_joints = [UsdPhysics.FixedJoint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    if fix_base:
        assert any(joint.GetBody1Rel().GetTargets() for joint in fixed_joints)
    else:
        assert not any(prim.GetName() == "fix_base_joint" for prim in stage.Traverse())


def test_importer_options_produce_spawnable_usd(sim_config):
    """Pass-through importer options run end to end and still yield a spawnable articulation.

    Collision and mesh approximation schemas are applied before the asset transformer restructures
    the USD, and the deprecated options only warn, so the observable contract is a valid output.
    """
    sim, config = sim_config
    config.merge_fixed_joints = True
    config.collision_from_visuals = True
    config.collision_type = "Convex Decomposition"
    config.link_density = 500.0
    config.convert_mimic_joints_to_normal_joints = True
    config.replace_cylinders_with_capsules = True
    config.root_link_name = "some_link"

    converter = UrdfConverter(config)

    stage = Usd.Stage.Open(converter.usd_path)
    assert any(prim.HasAPI(UsdPhysics.MassAPI) for prim in stage.Traverse())
    _assert_spawnable(sim, converter.usd_path)


def test_natural_frequency_gains_deprecation(sim_config):
    """``NaturalFrequencyGainsCfg`` warns but the conversion still succeeds."""
    sim, config = sim_config
    config.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(natural_frequency=10.0)

    with pytest.warns(DeprecationWarning, match="NaturalFrequencyGainsCfg"):
        converter = UrdfConverter(config)

    _assert_spawnable(sim, converter.usd_path)


def test_self_collision(sim_config):
    """``self_collision=True`` enables self-collisions on the Newton articulation root.

    The Isaac Sim importer's ``enable_self_collision`` writes the ``newton:selfCollisionEnabled``
    attribute on prims tagged as articulation roots.
    """
    _, config = sim_config
    config.self_collision = True

    stage = Usd.Stage.Open(UrdfConverter(config).usd_path)

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]
    assert articulation_roots, "Expected at least one articulation root in the converted USD"
    assert any(bool(prim.GetAttribute("newton:selfCollisionEnabled").Get()) for prim in articulation_roots)


def test_merge_fixed_joints_xml(tmp_path):
    """The importer's ``merge_fixed_joints`` collapses fixed joints and re-parents visuals and joints.

    ``test_merge_joints.urdf`` has 7 links and 6 joints (3 fixed, 1 continuous, 2 prismatic); merging
    leaves 4 links and the 3 moving joints.
    """
    if _USE_KIT:
        _enable_importer_extension()
    from isaacsim.asset.importer.urdf.impl.urdf_utils import merge_fixed_joints

    output_path = str(tmp_path / "merged.urdf")
    merge_fixed_joints(_MERGE_JOINTS_URDF, output_path)
    root = ET.parse(output_path).getroot()

    links = {link.get("name"): link for link in root.findall("link")}
    joints = root.findall("joint")
    assert set(links) == {"root_link", "link_1", "finger_link_1", "finger_link_2"}
    assert sorted(joint.get("type") for joint in joints) == ["continuous", "prismatic", "prismatic"]
    # visuals of merged links move to their new parent: base_link -> root_link, link_2 + palm_link -> link_1
    assert len(links["root_link"].findall("visual")) >= 1
    assert len(links["link_1"].findall("visual")) == 3
    # the finger joints were parented to palm_link, which merged into link_1
    for joint in joints:
        if joint.find("child").get("link").startswith("finger_link"):
            assert joint.find("parent").get("link") == "link_1"


@pytest.mark.parametrize("variant", [None, "mujoco"], ids=["default_physics", "override_mujoco"])
def test_physics_variant_selection(sim_config, variant):
    """The converter selects the portable ``"physics"`` variant by default, or the requested one.

    The importer leaves its ``"Physics"`` variant set unselected, which composes the asset without
    joints, articulation roots, or mass properties.
    """
    _, config = sim_config
    if variant is not None:
        config.physics_variant = variant

    selection, joints, roots = _physics_variant(UrdfConverter(config).usd_path)

    assert selection == (variant or "physics")
    assert joints > 0 and roots > 0


def test_physics_variant_raises_again_on_retry(tmp_path):
    """A conversion which failed on the variant does not count as cached.

    The converter skips conversion when the asset hash matches, so recording the hash before the
    variant is settled would make an identical retry return the asset the importer selected.
    """
    config = UrdfConverterCfg(
        asset_path=_FIXED_ONLY_URDF, fix_base=True, usd_dir=str(tmp_path / "usd"), physics_variant="physx"
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="no 'physx' physics variant"):
            UrdfConverter(config)
