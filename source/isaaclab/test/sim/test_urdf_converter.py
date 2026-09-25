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
import warnings
from types import SimpleNamespace

import pytest

if _USE_KIT:
    import omni.kit.app

import isaaclab
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# The Kit-less container mounts the checkout read-only, so ``usd_dir`` goes under ``tmp_path``.
pytestmark = [pytest.mark.integration, pytest.mark.kitless]


# Portable Franka URDF for the kitless path (the Kit path uses the importer extension's bundled
# ``panda_arm_hand.urdf``). Both expose the same 7 revolute + 2 prismatic joint structure.
_REPO_FRANKA_URDF = os.path.join(
    os.path.dirname(isaaclab.__file__), "controllers", "config", "data", "lula_franka_gen.urdf"
)

# Fixed-joint fixture for the merge tests: 7 links / 6 joints (3 fixed, 1 continuous, 2 prismatic).
# Kept beside the tests so they are hermetic and run on either importer backend.
_MERGE_JOINTS_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdfs", "test_merge_joints.urdf")

# Fixed-joint-only fixture: the importer writes no PhysX data for it, so its "Physics" variant set
# offers no "physx" variant, so requesting it must fail.
_FIXED_ONLY_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "urdfs", "test_fixed_only.urdf")


# Create a fixture for setup and teardown
@pytest.fixture
def sim_config():
    stage = sim_utils.create_new_stage()
    if _USE_KIT:
        # Kit path: enable the importer extension and use its bundled Franka asset.
        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("isaacsim.asset.importer.urdf"):
            manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
        extension_id = manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
        extension_path = manager.get_extension_path(extension_id)
        asset_path = f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf"
        # Load kit helper
        sim = SimulationContext(SimulationCfg(dt=0.01))
    else:
        # Kitless path: the converter loads the importer from the standalone wheel. Spawning and
        # inspecting prims needs a USD stage but neither physics nor Kit, so the plain stage above
        # stands in for the simulation context.
        asset_path = _REPO_FRANKA_URDF
        sim = SimpleNamespace(stage=stage)
    # default configuration
    config = UrdfConverterCfg(
        asset_path=asset_path,
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
        ),
    )
    yield sim, config
    # Teardown
    if _USE_KIT:
        sim._disable_app_control_on_stop_handle = True  # prevent timeout
        sim.stop()
        sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_no_change(sim_config):
    """Call conversion twice. This should not generate a new USD file."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = config
    new_config.usd_dir = urdf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_config_change(sim_config, tmp_path):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""

    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_config_change")
    os.makedirs(output_dir, exist_ok=True)

    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # change the config
    new_config = config
    new_config.fix_base = not config.fix_base
    # define the usd directory
    new_config.usd_dir = output_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_create_prim_from_usd(sim_config):
    """Call conversion and create a prim from it."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_config_drive_type(sim_config, tmp_path):
    """Verify that ``target_type='position'``, a uniform drive type, and uniform PD gains are written into
    every joint's DriveAPI.

    Reads the converter's USD output directly via :class:`pxr.UsdPhysics.DriveAPI` so the assertion does
    not depend on a running PhysX simulation. Revolute joints are checked in N·m/deg (the USD storage
    convention) and prismatic joints in N/m.
    """
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_converter")
    os.makedirs(output_dir, exist_ok=True)

    stiffness = 42.0
    damping = 4.2

    config.force_usd_conversion = True
    config.joint_drive.drive_type = "acceleration"
    config.joint_drive.target_type = "position"
    config.joint_drive.gains.stiffness = stiffness
    config.joint_drive.gains.damping = damping
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    revolute_count = 0
    prismatic_count = 0
    for prim in stage.Traverse():
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        if not (is_revolute or is_prismatic):
            continue
        instance_name = "angular" if is_revolute else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
        actual_stiffness = drive.GetStiffnessAttr().Get()
        actual_damping = drive.GetDampingAttr().Get()
        assert drive.GetTypeAttr().Get() == "acceleration", f"Joint {prim.GetName()}: expected 'acceleration' drive"

        if is_revolute:
            expected_stiffness = stiffness * math.pi / 180.0
            expected_damping = damping * math.pi / 180.0
            revolute_count += 1
        else:
            expected_stiffness = stiffness
            expected_damping = damping
            prismatic_count += 1

        assert abs(actual_stiffness - expected_stiffness) < 1e-4, (
            f"Joint {prim.GetName()}: expected stiffness {expected_stiffness}, got {actual_stiffness}"
        )
        assert abs(actual_damping - expected_damping) < 1e-4, (
            f"Joint {prim.GetName()}: expected damping {expected_damping}, got {actual_damping}"
        )

    # Franka Panda has 7 revolute arm joints and 2 prismatic finger joints.
    assert revolute_count == 7, f"Expected 7 revolute joints, got {revolute_count}"
    assert prismatic_count == 2, f"Expected 2 prismatic joints, got {prismatic_count}"


@pytest.mark.isaacsim_ci
def test_merge_fixed_joints_converter(sim_config, tmp_path):
    """Test the full URDF converter pipeline with merge_fixed_joints enabled.

    ``test_merge_joints.urdf`` has 7 links and 6 joints (3 fixed, 1 continuous, 2 prismatic); merging
    leaves 4 links (root_link, link_1, finger_link_1, finger_link_2) and the 3 movable joints.
    """
    sim, config = sim_config
    # Create directory to dump results
    output_dir = os.path.join(str(tmp_path), "urdf_converter_merge")
    os.makedirs(output_dir, exist_ok=True)

    # use a URDF that has fixed joints
    config.asset_path = _MERGE_JOINTS_URDF
    config.merge_fixed_joints = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir

    urdf_converter = UrdfConverter(config)

    # check the USD file was created
    assert os.path.exists(urdf_converter.usd_path), f"USD file not found at: {urdf_converter.usd_path}"

    # create a prim from it and verify it's valid
    prim_path = "/World/MergedRobot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()

    # the fixed-joint children are merged into their parents; the movable joints remain
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)
    prims = list(stage.Traverse())
    names = {prim.GetName() for prim in prims}
    assert not names & {"base_link", "link_2", "palm_link"}, f"Merged links still present: {sorted(names)}"
    assert {"root_link", "link_1", "finger_link_1", "finger_link_2"} <= names
    movable = [p for p in prims if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
    assert len(movable) == 3, f"Expected 3 movable joints, got {[p.GetName() for p in movable]}"


@pytest.mark.isaacsim_ci
def test_fix_base_creates_fixed_joint(sim_config, tmp_path):
    """Verify that fix_base=True creates a FixedJoint in the output USD."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_fix_base")
    os.makedirs(output_dir, exist_ok=True)

    config.fix_base = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # search for a FixedJoint in the output
    fixed_joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed_joints) > 0, "Expected at least one FixedJoint from fix_base=True"

    # the first FixedJoint should target a rigid body link via body1
    fj = UsdPhysics.FixedJoint(fixed_joints[0])
    body1_targets = fj.GetBody1Rel().GetTargets()
    assert len(body1_targets) > 0, "FixedJoint should target a rigid body link via body1"


@pytest.mark.isaacsim_ci
def test_no_fix_base(sim_config, tmp_path):
    """Verify that fix_base=False does not create a fix_base_joint."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_no_fix_base")
    os.makedirs(output_dir, exist_ok=True)

    config.fix_base = False
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    # there should be no prim named "fix_base_joint"
    fix_base_prims = [p for p in stage.Traverse() if p.GetName() == "fix_base_joint"]
    assert len(fix_base_prims) == 0, "Expected no fix_base_joint when fix_base=False"


@pytest.mark.isaacsim_ci
def test_collision_from_visuals(sim_config, tmp_path):
    """Verify that collision_from_visuals with a non-default collision type runs and produces valid output.

    Note: CollisionAPI is applied on the intermediate stage before the asset transformer
    restructures the USD.  The transformer may not preserve CollisionAPI in the final
    output, so this test verifies the pipeline executes successfully rather than
    inspecting the final USD for CollisionAPI schemas.
    """
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_collision_visuals")
    os.makedirs(output_dir, exist_ok=True)

    config.collision_from_visuals = True
    config.collision_type = "Convex Decomposition"
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    assert os.path.exists(urdf_converter.usd_path), "USD file should exist after conversion"

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_self_collision(sim_config, tmp_path):
    """Verify that ``self_collision=True`` enables self-collision on the Newton articulation root.

    The Isaac Sim importer's ``enable_self_collision`` writes the ``newton:selfCollisionEnabled``
    attribute on prims tagged as articulation roots (``UsdPhysics.ArticulationRootAPI``,
    ``PhysicsArticulationRootAPI``, or ``NewtonArticulationRootAPI``).
    """
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_self_collision")
    os.makedirs(output_dir, exist_ok=True)

    config.self_collision = True
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    articulation_roots = [
        prim
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        or prim.HasAPI("PhysicsArticulationRootAPI")
        or prim.HasAPI("NewtonArticulationRootAPI")
    ]
    assert articulation_roots, "Expected at least one articulation root in the converted USD"

    found_self_collision = False
    for prim in articulation_roots:
        sc_attr = prim.GetAttribute("newton:selfCollisionEnabled")
        if sc_attr and sc_attr.HasValue() and sc_attr.Get():
            found_self_collision = True
            break

    assert found_self_collision, "Expected ``newton:selfCollisionEnabled`` to be True on a Newton articulation root"


@pytest.mark.isaacsim_ci
def test_target_type_none_zeros_gains(sim_config, tmp_path):
    """Verify that ``target_type='none'`` zeros the DriveAPI stiffness and damping on every joint."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_target_none")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.target_type = "none"
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    joint_count = 0
    for prim in stage.Traverse():
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        is_prismatic = prim.IsA(UsdPhysics.PrismaticJoint)
        if not (is_revolute or is_prismatic):
            continue
        instance_name = "angular" if is_revolute else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
        assert abs(drive.GetStiffnessAttr().Get()) < 1e-6, (
            f"Joint {prim.GetName()}: expected zero stiffness, got {drive.GetStiffnessAttr().Get()}"
        )
        assert abs(drive.GetDampingAttr().Get()) < 1e-6, (
            f"Joint {prim.GetName()}: expected zero damping, got {drive.GetDampingAttr().Get()}"
        )
        joint_count += 1

    assert joint_count > 0, "No joints found in the output USD"


@pytest.mark.isaacsim_ci
def test_per_joint_dict_gains(sim_config, tmp_path):
    """Verify that per-joint dict-based gains and drive types are applied correctly."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_dict_gains")
    os.makedirs(output_dir, exist_ok=True)

    arm_stiffness = 100.0
    finger_stiffness = 200.0
    arm_damping = 10.0
    finger_damping = 20.0

    config.force_usd_conversion = True
    config.joint_drive.target_type = "position"
    config.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
        stiffness={
            "panda_joint[1-7]": arm_stiffness,
            "panda_finger": finger_stiffness,
        },
        damping={
            "panda_joint[1-7]": arm_damping,
            "panda_finger": finger_damping,
        },
    )
    config.joint_drive.drive_type = {
        "panda_joint[1-7]": "acceleration",
        "panda_finger": "force",
    }
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    # inspect the USD directly rather than going through PhysX to verify per-joint values
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(urdf_converter.usd_path)

    arm_joint_count = 0
    finger_joint_count = 0
    for prim in stage.Traverse():
        if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)):
            continue
        name = prim.GetName()
        is_revolute = prim.IsA(UsdPhysics.RevoluteJoint)
        instance_name = "angular" if is_revolute else "linear"
        drive = UsdPhysics.DriveAPI.Get(prim, instance_name)
        stiffness_attr = drive.GetStiffnessAttr()
        damping_attr = drive.GetDampingAttr()

        if "panda_joint" in name and "finger" not in name:
            # arm joint (revolute) — USD stores in Nm/deg, so expected = value * pi/180
            expected_s = arm_stiffness * math.pi / 180.0
            expected_d = arm_damping * math.pi / 180.0
            assert abs(stiffness_attr.Get() - expected_s) < 0.01, (
                f"Arm joint {name}: expected stiffness ~{expected_s}, got {stiffness_attr.Get()}"
            )
            assert abs(damping_attr.Get() - expected_d) < 0.01, (
                f"Arm joint {name}: expected damping ~{expected_d}, got {damping_attr.Get()}"
            )
            assert drive.GetTypeAttr().Get() == "acceleration", f"Arm joint {name}: expected 'acceleration'"
            arm_joint_count += 1
        elif "finger" in name:
            # finger joint (prismatic) — USD stores directly in N/m
            assert abs(stiffness_attr.Get() - finger_stiffness) < 0.01, (
                f"Finger joint {name}: expected stiffness {finger_stiffness}, got {stiffness_attr.Get()}"
            )
            assert abs(damping_attr.Get() - finger_damping) < 0.01, (
                f"Finger joint {name}: expected damping {finger_damping}, got {damping_attr.Get()}"
            )
            assert drive.GetTypeAttr().Get() == "force", f"Finger joint {name}: expected 'force'"
            finger_joint_count += 1

    assert arm_joint_count == 7, f"Expected 7 arm joints, got {arm_joint_count}"
    assert finger_joint_count == 2, f"Expected 2 finger joints, got {finger_joint_count}"


@pytest.mark.isaacsim_ci
def test_natural_frequency_gains_deprecation(sim_config, tmp_path):
    """Verify that NaturalFrequencyGainsCfg emits a DeprecationWarning and conversion still succeeds."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_nat_freq")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.joint_drive.gains = UrdfConverterCfg.JointDriveCfg.NaturalFrequencyGainsCfg(
        natural_frequency=10.0,
    )
    config.usd_dir = output_dir

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        urdf_converter = UrdfConverter(config)
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(dep_warnings) >= 1, "Expected DeprecationWarning for NaturalFrequencyGainsCfg"
        assert "NaturalFrequencyGainsCfg" in str(dep_warnings[0].message)

    # conversion should still succeed
    assert os.path.exists(urdf_converter.usd_path), "USD file should be created despite deprecation"

    # verify we can spawn from the output
    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)
    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_link_density(sim_config, tmp_path):
    """Verify that link_density applies density to links without an explicit mass.

    In ``test_merge_joints.urdf``, ``link_2`` has no inertial and the finger links have no mass, so they
    get the configured density; ``link_1`` and ``palm_link`` have an explicit mass and keep it.
    """
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_link_density")
    os.makedirs(output_dir, exist_ok=True)

    config.asset_path = _MERGE_JOINTS_URDF
    config.merge_fixed_joints = False
    config.link_density = 500.0
    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    from pxr import Usd

    stage = Usd.Stage.Open(urdf_converter.usd_path)
    densities = {
        prim.GetName(): prim.GetAttribute("physics:density").Get()
        for prim in stage.Traverse()
        if prim.GetAttribute("physics:density").HasAuthoredValue()
    }
    for link_name in ("link_2", "finger_link_1", "finger_link_2"):
        assert densities.get(link_name) == pytest.approx(500.0), f"{link_name}: densities={densities}"
    for link_name in ("link_1", "palm_link"):
        assert link_name not in densities, f"{link_name} has an explicit mass: densities={densities}"


@pytest.mark.isaacsim_ci
def test_unsupported_features_warn(sim_config, tmp_path, caplog):
    """Verify that deprecated config options emit warnings without failing."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_deprecated_warn")
    os.makedirs(output_dir, exist_ok=True)

    config.convert_mimic_joints_to_normal_joints = True
    config.replace_cylinders_with_capsules = True
    config.root_link_name = "some_link"
    config.force_usd_conversion = True
    config.usd_dir = output_dir

    # conversion should succeed despite deprecated options, warning once per option
    with caplog.at_level("WARNING"):
        urdf_converter = UrdfConverter(config)
    assert os.path.exists(urdf_converter.usd_path), "USD file should be created despite deprecated options"
    for option in ("convert_mimic_joints_to_normal_joints", "replace_cylinders_with_capsules", "root_link_name"):
        assert f"'{option}' is no longer supported" in caplog.text, option


def _physics_variant(usd_path: str) -> tuple[str, list[str]]:
    """Return the authored ``"Physics"`` variant selection and the available variants.

    Both are read while the stage is still referenced, since USD objects do not keep it alive.
    """
    from pxr import Usd

    stage = Usd.Stage.Open(usd_path)
    variant_set = stage.GetDefaultPrim().GetVariantSets().GetVariantSet("Physics")
    return variant_set.GetVariantSelection(), variant_set.GetVariantNames()


def _count_physics(usd_path: str) -> tuple[int, int]:
    """Return the number of joints and articulation roots composed from the USD file at ``usd_path``.

    The stage is held in a local for the whole traversal, since prims do not keep it alive.
    """
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(usd_path)
    prims = list(stage.Traverse())
    joints = sum(1 for prim in prims if prim.IsA(UsdPhysics.Joint))
    roots = sum(1 for prim in prims if prim.HasAPI(UsdPhysics.ArticulationRootAPI))
    return joints, roots


@pytest.mark.isaacsim_ci
def test_physics_variant_selected_by_default(sim_config, tmp_path):
    """Verify that the converter selects the backend-portable ``"physics"`` variant by default.

    The importer leaves its ``"Physics"`` variant set unselected, which composes the asset without
    joints, articulation roots, or mass properties. Without a selection this asserts 0 joints.
    """
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_physics_variant_default")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)

    selection, _ = _physics_variant(urdf_converter.usd_path)
    assert selection == "physics"

    joints, roots = _count_physics(urdf_converter.usd_path)
    assert joints > 0, "Expected the converted USD to compose joints"
    assert roots > 0, "Expected the converted USD to compose an articulation root"


@pytest.mark.isaacsim_ci
def test_physics_variant_override(sim_config, tmp_path):
    """Verify that ``physics_variant`` selects the requested variant instead of the default."""
    sim, config = sim_config
    output_dir = os.path.join(str(tmp_path), "urdf_physics_variant_override")
    os.makedirs(output_dir, exist_ok=True)

    config.force_usd_conversion = True
    config.usd_dir = output_dir
    config.physics_variant = "mujoco"
    urdf_converter = UrdfConverter(config)

    selection, _ = _physics_variant(urdf_converter.usd_path)
    assert selection == "mujoco"

    joints, roots = _count_physics(urdf_converter.usd_path)
    assert joints > 0, "Expected the converted USD to compose joints"
    assert roots > 0, "Expected the converted USD to compose an articulation root"


@pytest.mark.isaacsim_ci
def test_physics_variant_raises_again_on_retry(tmp_path):
    """Verify that a conversion which failed on the variant does not count as cached.

    The converter skips conversion when the asset hash matches, so recording the hash before the
    variant is settled would make an identical retry return the asset the importer selected.
    """
    output_dir = os.path.join(str(tmp_path), "urdf_physics_variant_missing_retry")
    os.makedirs(output_dir, exist_ok=True)

    config = UrdfConverterCfg(
        asset_path=_FIXED_ONLY_URDF,
        fix_base=True,
        usd_dir=output_dir,
        physics_variant="physx",
    )

    for _ in range(2):
        with pytest.raises(ValueError, match="no 'physx' physics variant"):
            UrdfConverter(config)
