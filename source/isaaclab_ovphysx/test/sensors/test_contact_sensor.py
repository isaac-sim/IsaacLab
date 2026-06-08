# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX ContactSensor.

Run via ``./isaaclab.sh -p -m pytest``; the ovphysx wheel is now invocable
through the standard Kit Python entrypoint, so the older kitless
``./scripts/run_ovphysx.sh`` wrapper is no longer required.

The OVPhysX runtime binds device mode (CPU vs GPU) at the C++ layer on the
first ``ovphysx.PhysX(device=...)`` construction and cannot swap it without a
process restart.  Full coverage therefore requires two separate pytest
invocations -- once with ``-k 'cpu'`` and once with ``-k 'cuda:0'``.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts the manager's
:exc:`RuntimeError` by ``pytest.skip``-ing on the unlocked device so
single-device runs finish cleanly.

Two v1-unsupported feature tests are kept but marked ``@pytest.mark.skip``:

* :func:`test_friction_reporting` — requires ``track_friction_forces``; see
  issue #5325 and ``docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md``.
* :func:`test_invalid_prim_paths_config` — requires ``track_friction_forces``
  (used to configure the scene); same issue.
* :func:`test_invalid_max_contact_points_config` — requires
  ``track_friction_forces``; same issue.

The ``disable_contact_processing`` PhysX/Kit setting is not available in the
kitless OVPhysX flow; :func:`test_cube_contact_time` and
:func:`test_sphere_contact_time` therefore drop that parametrize axis and run
once per device.
"""

from __future__ import annotations

from dataclasses import MISSING
from enum import Enum

import pytest
import torch
import warp as wp
from flaky import flaky

# The CI isaaclab_ov* pattern unintentionally collects isaaclab_ovphysx tests,
# but the ovphysx wheel is not installed in that environment. Skip gracefully
# so the isaaclab_ov CI pipeline is not blocked by an unrelated dependency.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ovphysx.assets import RigidObject  # noqa: E402
from isaaclab_ovphysx.physics import OvPhysxCfg  # noqa: E402
from isaaclab_ovphysx.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

wp.init()

# ---------------------------------------------------------------------------
# Device-lock autouse fixture
# ---------------------------------------------------------------------------

_LOCKED_DEVICE: list[str | None] = [None]
"""Device the session pins to on the first parametrized test that runs."""


@pytest.fixture(autouse=True)
def _ovphysx_skip_other_device(request):
    """Skip parametrized tests on the device the session is not pinned to.

    See the module docstring for the wheel's process-global device-mode lock.
    """
    callspec = getattr(request.node, "callspec", None)
    device = callspec.params.get("device") if callspec is not None else None
    if device is None:
        # Test does not parametrize on device.
        return
    locked = _LOCKED_DEVICE[0]
    if locked is None:
        _LOCKED_DEVICE[0] = device
        return
    if device != locked:
        pytest.skip(
            f"ovphysx process-global device lock is held by '{locked}'; cannot run '{device}' "
            "tests in the same session.  Run pytest twice (once per device) for full coverage."
        )


# ---------------------------------------------------------------------------
# Simulation context helper
# ---------------------------------------------------------------------------


def _ovphysx_sim_context(device: str, **kwargs):
    """Wrapper around :func:`build_simulation_context` that injects OVPhysX cfg.

    PhysX tests pass ``device=device`` directly and let
    :func:`build_simulation_context` build a default :class:`SimulationCfg`.
    OVPhysX needs ``physics=OvPhysxCfg()`` set on the cfg so the manager
    dispatches to OVPhysX rather than PhysX, so we build the cfg here and
    pass it through.  ``gravity_enabled`` is consumed locally (it is ignored
    by ``build_simulation_context`` once a ``sim_cfg`` is provided).
    ``add_ground_plane``, ``auto_add_lighting``, and other kwargs continue
    to flow through ``build_simulation_context`` as before.
    """
    dt = kwargs.pop("dt", 1.0 / 60.0)
    gravity_enabled = kwargs.pop("gravity_enabled", True)
    gravity = (0.0, 0.0, -9.81) if gravity_enabled else (0.0, 0.0, 0.0)
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt, gravity=gravity)
    return build_simulation_context(device=device, sim_cfg=sim_cfg, **kwargs)


##
# Custom helper classes.
##


class ContactTestMode(Enum):
    """Enum to declare the type of contact sensor test to execute."""

    IN_CONTACT = 0
    """Enum to test the condition where the test object is in contact with the ground plane."""
    NON_CONTACT = 1
    """Enum to test the condition where the test object is not in contact with the ground plane (air time)."""


@configclass
class ContactSensorRigidObjectCfg(RigidObjectCfg):
    """Configuration for rigid objects used for the contact sensor test.

    This contains the expected values in the configuration to simplify test fixtures.
    """

    contact_pose: torch.Tensor = MISSING
    """6D pose of the rigid object under test when it is in contact with the ground surface."""
    non_contact_pose: torch.Tensor = MISSING
    """6D pose of the rigid object under test when it is not in contact."""


@configclass
class ContactSensorSceneCfg(InteractiveSceneCfg):
    """Configuration of the scene used by the contact sensor test."""

    terrain: TerrainImporterCfg = MISSING
    """Terrain configuration within the scene."""

    shape: ContactSensorRigidObjectCfg = MISSING
    """RigidObject contact prim configuration."""

    contact_sensor: ContactSensorCfg = MISSING
    """Contact sensor configuration."""

    shape_2: ContactSensorRigidObjectCfg = None
    """RigidObject contact prim configuration. Defaults to None, i.e. not included in the scene.

    This is a second prim used for testing contact filtering.
    """

    contact_sensor_2: ContactSensorCfg = None
    """Contact sensor configuration. Defaults to None, i.e. not included in the scene.

    This is a second contact sensor used for testing contact filtering.
    """


##
# Scene entity configurations.
##


CUBE_CFG = ContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.6, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, -1.0, 1.0)),
    contact_pose=torch.tensor([0, -1.0, 0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, -1.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cube prim."""

SPHERE_CFG = ContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Sphere",
    spawn=sim_utils.SphereCfg(
        radius=0.25,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 1.0, 1.0)),
    contact_pose=torch.tensor([0, 1.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, 1.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the sphere prim."""

CYLINDER_CFG = ContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cylinder",
    spawn=sim_utils.CylinderCfg(
        radius=0.5,
        height=0.01,
        axis="Y",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.4, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0.0, 1.0)),
    contact_pose=torch.tensor([0, 0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([0, 0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cylinder prim."""

CAPSULE_CFG = ContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Capsule",
    spawn=sim_utils.CapsuleCfg(
        radius=0.25,
        height=0.5,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.5)),
    contact_pose=torch.tensor([1.0, 0.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([1.0, 0.0, 1.5, 1, 0, 0, 0]),
)
"""Configuration of the capsule prim."""

CONE_CFG = ContactSensorRigidObjectCfg(
    prim_path="/World/Objects/Cone",
    spawn=sim_utils.ConeCfg(
        radius=0.5,
        height=0.5,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
        ),
        activate_contact_sensors=True,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.2, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(-1.0, 0.0, 1.0)),
    contact_pose=torch.tensor([-1.0, 0.0, 0.0, 1, 0, 0, 0]),
    non_contact_pose=torch.tensor([-1.0, 0.0, 1.0, 1, 0, 0, 0]),
)
"""Configuration of the cone prim."""

FLAT_TERRAIN_CFG = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
"""Configuration of the flat ground plane."""

COBBLESTONE_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        seed=0,
        size=(3.0, 3.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "random_rough": HfRandomUniformTerrainCfg(
                proportion=1.0, noise_range=(0.0, 0.05), noise_step=0.01, border_width=0.25
            ),
        },
    ),
)
"""Configuration of the generated mesh terrain."""

##
# Shared test constants.
##

_SIM_DT = 0.0025
"""Simulation time-step [s] used across all contact sensor tests."""

_DURATIONS = [_SIM_DT, _SIM_DT * 2, _SIM_DT * 32, _SIM_DT * 128]
"""Contact/air durations [s] exercised by the timing tests."""

_TERRAINS = [FLAT_TERRAIN_CFG, COBBLESTONE_TERRAIN_CFG]
"""Terrain configurations exercised by the timing tests."""

##
# Tests.
##


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@flaky(max_runs=5, min_passes=1)
@pytest.mark.isaacsim_ci
def test_cube_contact_time(device):
    """Checks contact sensor values for contact time and air time for a cube collision primitive."""
    _run_contact_sensor_test(CUBE_CFG, _SIM_DT, device, _TERRAINS, _DURATIONS)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@flaky(max_runs=5, min_passes=1)
@pytest.mark.isaacsim_ci
def test_sphere_contact_time(device):
    """Checks contact sensor values for contact time and air time for a sphere collision primitive."""
    _run_contact_sensor_test(SPHERE_CFG, _SIM_DT, device, _TERRAINS, _DURATIONS)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 6, 24])
@pytest.mark.isaacsim_ci
def test_cube_stack_contact_filtering(device, num_envs):
    """Checks contact sensor reporting for filtering stacked cube prims."""
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=True) as sim:
        # Instance new scene for the current terrain and contact prim.
        # OVPhysX uses fnmatch globs (not regex), so ``Env_*`` rather than ``Env_.*``.
        scene_cfg = ContactSensorSceneCfg(num_envs=num_envs, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        # -- cube 1
        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_1")
        scene_cfg.shape.init_state.pos = (0, -1.0, 1.0)
        # -- cube 2 (on top of cube 1)
        scene_cfg.shape_2 = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_2")
        scene_cfg.shape_2.init_state.pos = (0, -1.0, 1.525)
        # -- contact sensor 1
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_2"],
        )
        # -- contact sensor 2
        scene_cfg.contact_sensor_2 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_1"],
        )
        scene = InteractiveScene(scene_cfg)

        # Play the simulation
        sim.reset()

        contact_sensor: ContactSensor = scene["contact_sensor"]
        contact_sensor_2: ContactSensor = scene["contact_sensor_2"]

        # Check that the filter binding was created for each sensor
        assert contact_sensor.contact_view.filter_count == 1
        assert contact_sensor_2.contact_view.filter_count == 1

        # Let the scene settle and accumulate contacts
        scene.reset()
        for _ in range(500):
            _perform_sim_step(sim, scene, _SIM_DT)

        # Check values for cube 2 — cube 1 is the only collision for cube 2
        torch.testing.assert_close(
            contact_sensor_2.data.force_matrix_w.torch[:, :, 0],
            contact_sensor_2.data.net_forces_w.torch,
        )
        # Check that forces are opposite and equal
        torch.testing.assert_close(
            contact_sensor_2.data.force_matrix_w.torch[:, :, 0],
            -contact_sensor.data.force_matrix_w.torch[:, :, 0],
        )
        # Check values are non-zero (contacts are happening and are getting reported)
        assert contact_sensor_2.data.net_forces_w.torch.sum().item() > 0.0
        assert contact_sensor.data.net_forces_w.torch.sum().item() > 0.0


@pytest.mark.isaacsim_ci
def test_no_contact_reporting():
    """Test that OVPhysX contact sensor returns zero forces when no filter is configured.

    Without ``filter_prim_paths_expr``, the ``force_matrix_w`` buffer is not
    populated (no per-partner breakdown is available), and ``net_forces_w``
    should still reflect the aggregate contact force.  This test verifies the
    simpler "unfiltered, CPU-only" path by using CPU and letting the scene
    settle: with no filter the ``force_matrix_w`` sum is expected to be zero
    (the buffer is not allocated).

    Note:
        The PhysX variant of this test forcibly disables contact processing via
        a Carbonite setting (``/physics/disableContactProcessing``).  That
        setting is not available in the kitless OVPhysX flow; instead we test
        that a sensor with no filter has a zero ``force_matrix_w``.
    """
    with _ovphysx_sim_context(device="cpu", dt=_SIM_DT, add_lighting=True) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=2, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        # -- cube 1
        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_1")
        scene_cfg.shape.init_state.pos = (0, -1.0, 1.0)
        # -- cube 2 (on top of cube 1)
        scene_cfg.shape_2 = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_2")
        scene_cfg.shape_2.init_state.pos = (0, -1.0, 1.525)
        # No filter paths — force_matrix_w will not be allocated.
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_1",
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[],
        )
        scene_cfg.contact_sensor_2 = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_2",
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[],
        )
        scene = InteractiveScene(scene_cfg)

        # Play the simulation
        sim.reset()

        contact_sensor: ContactSensor = scene["contact_sensor"]
        contact_sensor_2: ContactSensor = scene["contact_sensor_2"]

        # Let the scene settle
        scene.reset()
        for _ in range(500):
            _perform_sim_step(sim, scene, _SIM_DT)

        # Without filter_prim_paths_expr the force_matrix_w buffer is not allocated;
        # its sum should be zero (or the tensor is None).
        fm1 = contact_sensor.data.force_matrix_w
        fm2 = contact_sensor_2.data.force_matrix_w
        if fm1 is not None:
            assert fm1.torch.sum().item() == 0.0
        if fm2 is not None:
            assert fm2.torch.sum().item() == 0.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.isaacsim_ci
def test_multi_body_per_sensor_indexing(device, num_envs):
    """Ground-truth body-index check for a single sensor that resolves to two bodies.

    OVPhysX :class:`ContactBinding` returns sensors in **pattern-major** order
    (``[env_0/body_0, env_1/body_0, …, env_0/body_1, env_1/body_1, …]``),
    whereas the inherited PhysX kernel formula assumes env-major
    (``[env_0/body_0, env_0/body_1, …, env_1/body_0, …]``).  Single-body
    sensors don't disambiguate the two layouts, so this test exercises the
    multi-body discovery path with one cube on the ground and one floating
    above it.  After the scene settles, only the bottom cube should report a
    non-zero net force.  An env-major bug would attribute that force to the
    wrong (env, body) slot — caught here.
    """
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=True) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=num_envs, env_spacing=2.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        # -- Cube_low: on the ground, will report contact forces
        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_low")
        scene_cfg.shape.init_state.pos = (0.0, 0.0, 0.25)
        # -- Cube_high: floating well above the ground, should remain in air
        scene_cfg.shape_2 = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube_high")
        scene_cfg.shape_2.init_state.pos = (0.0, 1.5, 3.0)
        # Single ContactSensor that matches BOTH cubes via a regex glob.
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube_.*",
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
            filter_prim_paths_expr=[],
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        contact_sensor: ContactSensor = scene["contact_sensor"]

        # Sanity: the sensor discovered exactly two bodies, one per cube.
        assert contact_sensor.body_names is not None
        assert sorted(contact_sensor.body_names) == ["Cube_high", "Cube_low"]
        low_idx = contact_sensor.body_names.index("Cube_low")
        high_idx = contact_sensor.body_names.index("Cube_high")

        # Let physics settle and accumulate stable contacts on Cube_low.
        scene.reset()
        for _ in range(200):
            _perform_sim_step(sim, scene, _SIM_DT)

        # Net force readout: shape (num_envs, num_sensors=2, 3) after .torch.
        net_forces = contact_sensor.data.net_forces_w.torch
        assert net_forces.shape == (num_envs, 2, 3)
        low_force_mag = net_forces[:, low_idx, :].abs().sum().item()
        high_force_mag = net_forces[:, high_idx, :].abs().sum().item()
        # Cube_low rests on the ground: non-zero contact force per env.
        assert low_force_mag > 0.0, "Cube_low (on ground) should report contact force"
        # Cube_high floats: net force is zero (no contact).
        assert high_force_mag == 0.0, (
            f"Cube_high (in air) should report zero contact force, got sum-abs={high_force_mag:.6f}."
            " A non-zero value here usually means body indices are scrambled —"
            " e.g. a Cube_low contact was attributed to Cube_high because the kernel"
            " assumed env-major instead of pattern-major flat-buffer layout."
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_sensor_print(device):
    """Test sensor print is working correctly."""
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        scene_cfg.shape = CUBE_CFG
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=scene_cfg.shape.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=3,
        )
        scene = InteractiveScene(scene_cfg)
        # Play the simulator
        sim.reset()
        # print info
        print(scene.sensors["contact_sensor"])


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_contact_sensor_threshold(device):
    """Test that the contact sensor USD threshold attribute is set to 0.0."""
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        scene_cfg.shape = CUBE_CFG
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=scene_cfg.shape.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=3,
        )
        scene = InteractiveScene(scene_cfg)
        # Play the simulator
        sim.reset()

        stage = get_current_stage()
        prim_path = scene_cfg.shape.prim_path
        prim = stage.GetPrimAtPath(prim_path)

        # Ensure the contact sensor was created properly
        contact_sensor = scene["contact_sensor"]
        assert contact_sensor is not None, "Contact sensor was not created"

        # Check if the prim has contact report API and verify threshold is close to 0.0
        if "PhysxContactReportAPI" in prim.GetAppliedSchemas():
            threshold_attr = prim.GetAttribute("physxContactReport:threshold")
            if threshold_attr.IsValid():
                threshold_value = threshold_attr.Get()
                assert pytest.approx(threshold_value, abs=1e-6) == 0.0, (
                    f"Expected USD threshold to be close to 0.0, but got {threshold_value}"
                )


@pytest.mark.skip(
    reason=(
        "ovphysx ContactSensor v1 does not support track_friction_forces; "
        "see issue #5325 and docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md"
    )
)
@pytest.mark.parametrize("grav_dir", [(-10.0, 0.0, -0.1), (0.0, -10.0, -0.1)])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_friction_reporting(device, grav_dir):
    """Test friction force reporting for contact sensors.

    This test places a contact sensor enabled cube onto a ground plane under different gravity directions.
    It then compares the normalized friction force dir with the direction of gravity to ensure they are aligned.
    """
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), dt=_SIM_DT, device=device, gravity=grav_dir)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        scene_cfg.shape = CUBE_CFG

        filter_prim_paths_expr = [scene_cfg.terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"]

        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=scene_cfg.shape.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=3,
            track_friction_forces=True,
            filter_prim_paths_expr=filter_prim_paths_expr,
        )

        scene = InteractiveScene(scene_cfg)
        sim.reset()

        scene["contact_sensor"].reset()
        shape: RigidObject = scene["shape"]
        shape.write_root_pose_to_sim_index(
            root_pose=torch.tensor([0, 0.0, CUBE_CFG.spawn.size[2] / 2.0, 1, 0, 0, 0], device=device).unsqueeze(0)
        )

        # step sim once to compute friction forces
        _perform_sim_step(sim, scene, _SIM_DT)

        # check that forces are being reported match expected friction forces
        expected_friction, _, _, _ = scene["contact_sensor"].contact_view.get_friction_data(dt=_SIM_DT)
        expected_friction_torch = wp.to_torch(expected_friction)
        reported_friction = scene["contact_sensor"].data.friction_forces_w.torch[0, 0, :]

        torch.testing.assert_close(expected_friction_torch.sum(dim=0), reported_friction[0], atol=1e-6, rtol=1e-5)

        # check that friction force direction opposes gravity direction
        grav = torch.tensor(grav_dir, device=device)
        norm_reported_friction = reported_friction / reported_friction.norm()
        norm_gravity = grav / grav.norm()
        dot = torch.dot(norm_reported_friction[0], norm_gravity)

        torch.testing.assert_close(torch.abs(dot), torch.tensor(1.0, device=device), atol=1e-4, rtol=1e-3)


@pytest.mark.skip(
    reason=(
        "ovphysx ContactSensor v1 does not support track_friction_forces; "
        "see issue #5325 and docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md"
    )
)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_invalid_prim_paths_config(device):
    """Test that a ValueError is raised when track_friction_forces=True and filter_prim_paths_expr is empty."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), dt=_SIM_DT, device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        scene_cfg.shape = CUBE_CFG

        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=scene_cfg.shape.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=3,
            track_friction_forces=True,
            filter_prim_paths_expr=[],
        )

        try:
            _ = InteractiveScene(scene_cfg)
            sim.reset()
            assert False, "Expected ValueError due to invalid contact sensor configuration."
        except ValueError:
            pass


@pytest.mark.skip(
    reason=(
        "ovphysx ContactSensor v1 does not support track_friction_forces; "
        "see issue #5325 and docs/superpowers/specs/2026-04-27-ovphysx-contact-api-gaps.md"
    )
)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_invalid_max_contact_points_config(device):
    """Test that a ValueError is raised when track_friction_forces=True and max_contact_data_count_per_prim=0."""
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), dt=_SIM_DT, device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        scene_cfg.shape = CUBE_CFG
        filter_prim_paths_expr = [scene_cfg.terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"]

        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=scene_cfg.shape.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=3,
            track_friction_forces=True,
            filter_prim_paths_expr=filter_prim_paths_expr,
            max_contact_data_count_per_prim=0,
        )

        try:
            _ = InteractiveScene(scene_cfg)
            sim.reset()
            assert False, "Expected ValueError due to invalid contact sensor configuration."
        except ValueError:
            pass


##
# Internal helpers.
##


def _run_contact_sensor_test(
    shape_cfg: ContactSensorRigidObjectCfg,
    sim_dt: float,
    device: str,
    terrains: list[TerrainImporterCfg],
    durations: list[float],
):
    """Run contact sensor timing tests for a single device across all terrain combinations.

    Args:
        shape_cfg: Configuration of the rigid body used as contact primitive.
        sim_dt: Simulation time-step [s].
        device: Compute device (e.g. ``"cuda:0"`` or ``"cpu"``).
        terrains: List of terrain configurations to iterate over.
        durations: Contact / air durations [s] to exercise.

    Note:
        Unlike the PhysX variant, this helper never enables
        ``track_contact_points`` or ``track_friction_forces`` because those
        APIs are not yet available in the ovphysx v1 contact sensor (see
        issue #5325).  The ``test_contact_data`` path is therefore always
        ``False``.  The ``disable_contact_processing`` PhysX/Kit setting is
        also not available in the kitless flow and is omitted.
    """
    for terrain in terrains:
        with _ovphysx_sim_context(device=device, dt=sim_dt, add_lighting=True) as sim:
            scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
            scene_cfg.terrain = terrain
            scene_cfg.shape = shape_cfg

            scene_cfg.contact_sensor = ContactSensorCfg(
                prim_path=shape_cfg.prim_path,
                track_pose=True,
                debug_vis=False,
                update_period=0.0,
                track_air_time=True,
                history_length=3,
                track_contact_points=False,
                track_friction_forces=False,
                filter_prim_paths_expr=[],
            )
            scene = InteractiveScene(scene_cfg)

            # Play the simulation
            sim.reset()

            # Run contact time and air time tests
            _test_sensor_contact(
                shape=scene["shape"],
                sensor=scene["contact_sensor"],
                mode=ContactTestMode.IN_CONTACT,
                sim=sim,
                scene=scene,
                sim_dt=sim_dt,
                durations=durations,
            )
            _test_sensor_contact(
                shape=scene["shape"],
                sensor=scene["contact_sensor"],
                mode=ContactTestMode.NON_CONTACT,
                sim=sim,
                scene=scene,
                sim_dt=sim_dt,
                durations=durations,
            )


def _test_sensor_contact(
    shape: RigidObject,
    sensor: ContactSensor,
    mode: ContactTestMode,
    sim: SimulationContext,
    scene: InteractiveScene,
    sim_dt: float,
    durations: list[float],
):
    """Test for the contact sensor.

    This test sets the contact prim to a pose either in contact or out of contact with the ground plane for
    a known duration. Once the contact duration has elapsed, the data stored inside the contact sensor
    associated with the contact prim is checked against the expected values.

    This process is repeated for all elements in ``durations``, where each successive contact timing test
    is punctuated by setting the contact prim to the complement of the desired contact mode for 1 sim time-step.

    Args:
        shape: The contact prim used for the contact sensor test.
        sensor: The sensor reporting data to be verified by the contact sensor test.
        mode: The contact test mode: either contact with ground plane or air time.
        sim: The active simulation context.
        scene: The interactive scene.
        sim_dt: Simulation time-step [s].
        durations: Contact / air durations [s] to exercise.
    """
    # reset the test state
    sensor.reset()
    expected_last_test_contact_time = 0
    expected_last_reset_contact_time = 0

    # set poses for shape for a given contact sensor test mode.
    # desired contact mode to set for a given duration.
    test_pose = None
    # complement of the desired contact mode used to reset the contact sensor.
    reset_pose = None
    if mode == ContactTestMode.IN_CONTACT:
        test_pose = shape.cfg.contact_pose
        reset_pose = shape.cfg.non_contact_pose
    elif mode == ContactTestMode.NON_CONTACT:
        test_pose = shape.cfg.non_contact_pose
        reset_pose = shape.cfg.contact_pose
    else:
        raise ValueError("Received incompatible contact sensor test mode")

    for idx in range(len(durations)):
        current_test_time = 0
        duration = durations[idx]
        while current_test_time < duration:
            # set object states to contact the ground plane
            shape.write_root_pose_to_sim_index(root_pose=torch.tensor(test_pose, device=shape.device).unsqueeze(0))
            # perform simulation step
            _perform_sim_step(sim, scene, sim_dt)
            # increment contact time
            current_test_time += sim_dt
        # set last contact time to the previous desired contact duration plus the extra dt allowance.
        expected_last_test_contact_time = durations[idx - 1] + sim_dt if idx > 0 else 0
        # Check the data inside the contact sensor
        if mode == ContactTestMode.IN_CONTACT:
            _check_prim_contact_state_times(
                sensor=sensor,
                expected_air_time=0.0,
                expected_contact_time=durations[idx],
                expected_last_contact_time=expected_last_test_contact_time,
                expected_last_air_time=expected_last_reset_contact_time,
                dt=duration + sim_dt,
            )
        elif mode == ContactTestMode.NON_CONTACT:
            _check_prim_contact_state_times(
                sensor=sensor,
                expected_air_time=durations[idx],
                expected_contact_time=0.0,
                expected_last_contact_time=expected_last_reset_contact_time,
                expected_last_air_time=expected_last_test_contact_time,
                dt=duration + sim_dt,
            )

        # switch the contact mode for 1 dt step before the next contact test begins.
        shape.write_root_pose_to_sim_index(root_pose=torch.tensor(reset_pose, device=shape.device).unsqueeze(0))
        # perform simulation step
        _perform_sim_step(sim, scene, sim_dt)
        # set the last air time to 2 sim_dt steps, because last_air_time and last_contact_time
        # adds an additional sim_dt to the total time spent in the previous contact mode for uncertainty in
        # when the contact switch happened in between a dt step.
        expected_last_reset_contact_time = 2 * sim_dt


def _test_friction_forces(shape: RigidObject, sensor: ContactSensor, mode: ContactTestMode) -> None:
    """Verify friction force values reported by the contact sensor.

    This helper is only called from skipped tests (requires ``track_friction_forces``
    which is not supported in ovphysx v1).

    Args:
        shape: The contact prim used for the contact sensor test.
        sensor: The sensor reporting data to be verified.
        mode: The contact test mode.
    """
    if not sensor.cfg.track_friction_forces:
        assert sensor._data.friction_forces_w is None
        return

    # check shape of the friction_forces_w tensor (wp.to_torch expands vec3f -> float32 trailing dim)
    num_bodies = sensor.num_bodies
    friction_torch = sensor._data.friction_forces_w.torch
    assert friction_torch.shape == (sensor.num_instances // num_bodies, num_bodies, 1, 3)
    # compare friction forces
    if mode == ContactTestMode.IN_CONTACT:
        assert torch.any(torch.abs(friction_torch) > 1e-5).item()
        friction_forces, _, buffer_count, buffer_start_indices = sensor.contact_view.get_friction_data(
            dt=sensor._sim_physics_dt
        )
        friction_forces_t = wp.to_torch(friction_forces)
        buffer_count_t = wp.to_torch(buffer_count).to(torch.int32)
        buffer_start_t = wp.to_torch(buffer_start_indices).to(torch.int32)
        for i in range(sensor.num_instances * num_bodies):
            for j in range(sensor.contact_view.filter_count):
                start_index_ij = buffer_start_t[i, j]
                count_ij = buffer_count_t[i, j]
                force = torch.sum(friction_forces_t[start_index_ij : (start_index_ij + count_ij), :], dim=0)
                env_idx = i // num_bodies
                body_idx = i % num_bodies
                assert torch.allclose(force, friction_torch[env_idx, body_idx, j, :], atol=1e-5)
    elif mode == ContactTestMode.NON_CONTACT:
        assert torch.all(friction_torch == 0.0).item()


def _test_contact_position(shape: RigidObject, sensor: ContactSensor, mode: ContactTestMode) -> None:
    """Test for the contact positions (only implemented for sphere and flat terrain).

    Checks that the contact position is radius distance away from the root of the object.

    This helper is only called from skipped tests (requires ``track_contact_points``
    which is not supported in ovphysx v1).

    Args:
        shape: The contact prim used for the contact sensor test.
        sensor: The sensor reporting data to be verified.
        mode: The contact test mode.
    """
    if not sensor.cfg.track_contact_points:
        assert sensor._data.contact_pos_w is None
        return

    # check shape of the contact_pos_w tensor (wp.to_torch expands vec3f -> float32 trailing dim)
    num_bodies = sensor.num_bodies
    contact_pos_torch = sensor._data.contact_pos_w.torch
    assert contact_pos_torch.shape == (sensor.num_instances // num_bodies, num_bodies, 1, 3)
    # check contact positions
    if mode == ContactTestMode.IN_CONTACT:
        pos_w_torch = sensor._data.pos_w.torch
        contact_position = pos_w_torch + torch.tensor([[0.0, 0.0, -shape.cfg.spawn.radius]], device=pos_w_torch.device)
        assert torch.all(
            torch.abs(torch.linalg.norm(contact_pos_torch - contact_position.unsqueeze(1), ord=2, dim=-1)) < 1e-2
        ).item()
    elif mode == ContactTestMode.NON_CONTACT:
        assert torch.all(torch.isnan(contact_pos_torch)).item()


def _check_prim_contact_state_times(
    sensor: ContactSensor,
    expected_air_time: float,
    expected_contact_time: float,
    expected_last_air_time: float,
    expected_last_contact_time: float,
    dt: float,
):
    """Check contact sensor data matches expected values.

    Args:
        sensor: Instance of ContactSensor containing data to be tested.
        expected_air_time: Air time ground truth [s].
        expected_contact_time: Contact time ground truth [s].
        expected_last_air_time: Last air time ground truth [s].
        expected_last_contact_time: Last contact time ground truth [s].
        dt: Time since previous contact mode switch [s]. If the contact prim left contact 0.1 seconds ago,
            dt should be 0.1 + simulation dt seconds.
    """
    # store current state of the contact prim
    in_air = expected_air_time > 0.0
    in_contact = expected_contact_time > 0.0
    measured_contact_time = sensor.data.current_contact_time.torch
    measured_air_time = sensor.data.current_air_time.torch
    measured_last_contact_time = sensor.data.last_contact_time.torch
    measured_last_air_time = sensor.data.last_air_time.torch
    # check current contact state
    assert pytest.approx(measured_contact_time.item(), 0.01) == expected_contact_time
    assert pytest.approx(measured_air_time.item(), 0.01) == expected_air_time
    # check last contact state
    assert pytest.approx(measured_last_contact_time.item(), 0.01) == expected_last_contact_time
    assert pytest.approx(measured_last_air_time.item(), 0.01) == expected_last_air_time
    # check current contact mode
    assert sensor.compute_first_contact(dt=dt).torch.item() == in_contact
    assert sensor.compute_first_air(dt=dt).torch.item() == in_air


def _perform_sim_step(sim: SimulationContext, scene: InteractiveScene, sim_dt: float) -> None:
    """Update sensors and step the contact sensor test scene.

    Args:
        sim: The active simulation context.
        scene: The interactive scene.
        sim_dt: Simulation time-step [s].
    """
    # write data to simulation
    scene.write_data_to_sim()
    # simulate
    sim.step(render=False)
    # update buffers at sim dt
    scene.update(dt=sim_dt)
