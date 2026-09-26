# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Real-backend tests for the OVPhysX ContactSensor.

Run via ``uv run python -m pytest``; the ovphysx wheel is now invocable
through the standard Kit Python entrypoint, so the older kitless
``./scripts/run_ovphysx.sh`` wrapper is no longer required.

The OVPhysX runtime fixes device mode (CPU vs GPU) when the process creates
its first ``ovphysx.PhysX`` instance and cannot switch it without a process
restart. Full coverage therefore requires two separate pytest
invocations -- once with ``-k 'cpu'`` and once with ``-k 'cuda:0'``.  The
``_ovphysx_skip_other_device`` autouse fixture below preempts the manager's
:exc:`RuntimeError` by ``pytest.skip``-ing on the unlocked device so
single-device runs finish cleanly.

The PhysX friction-force, contact-point, and their config-validation tests have no
counterpart here: :class:`ContactSensor` raises :exc:`NotImplementedError` for
``track_friction_forces`` and ``track_contact_points`` (issue #5325).

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

# The OVPhysX runtime wheel is optional. Skip gracefully when it is not installed;
# CI jobs that need OVPhysX coverage install it explicitly.
pytest.importorskip("ovphysx.types", reason="ovphysx wheel not installed")

from isaaclab_ov.assets import RigidObject  # noqa: E402
from isaaclab_ov.cloner import ovphysx_replicate  # noqa: E402
from isaaclab_ov.physics import OvPhysxCfg  # noqa: E402
from isaaclab_ov.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg  # noqa: E402

from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.sim.schemas as schemas  # noqa: E402
from isaaclab import cloner  # noqa: E402
from isaaclab.assets import RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context  # noqa: E402
from isaaclab.sim.utils.stage import get_current_stage  # noqa: E402
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

wp.init()

pytestmark = pytest.mark.device_split

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
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(
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
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(
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
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(
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
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(
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
        rigid_props=PhysxRigidBodyCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(
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
def test_cube_contact_time(device):
    """Checks contact sensor values for contact time and air time for a cube collision primitive."""
    _run_contact_sensor_test(CUBE_CFG, _SIM_DT, device, _TERRAINS, _DURATIONS)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@flaky(max_runs=5, min_passes=1)
def test_sphere_contact_time(device):
    """Checks contact sensor values for contact time and air time for a sphere collision primitive."""
    _run_contact_sensor_test(SPHERE_CFG, _SIM_DT, device, _TERRAINS, _DURATIONS)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_first_transition_with_aged_clock(device):
    """Regression for #7283: transitions must still be reported once the sensor clock has aged.

    The sensor clock is a float32 accumulator whose rounding error grows with simulated time. On a
    transition step the contact (resp. air) timer is exactly one polling period, so the default
    tolerance of :meth:`ContactSensor.compute_first_contact` has to absorb that error. A fixed 1e-8
    tolerance is orders of magnitude too small after a few seconds of simulated time.
    """
    # The sensor keeps history, so it refreshes every physics step and is polled at that same rate.
    # The lazy (zero-history) cadence is covered by the kernel-level tolerance test: on GPU this
    # backend serves stale contact forces when buffers refresh only on data access, which is
    # unrelated to the tolerance under test here.
    # At 2.5 s the float32 clock error already breaks a fixed 1e-8 tolerance; not every age does
    # (10 s happens to round cleanly), so keep an age that fails without the adaptive tolerance.
    clock_age = 2.5
    history_length = 1
    decimation = 1
    poll_dt = decimation * _SIM_DT
    poll_steps = 16

    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=True) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        scene_cfg.shape = CUBE_CFG
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=CUBE_CFG.prim_path,
            track_pose=True,
            debug_vis=False,
            update_period=0.0,
            track_air_time=True,
            history_length=history_length,
            track_contact_points=False,
            track_friction_forces=False,
            filter_prim_paths_expr=[],
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        sensor: ContactSensor = scene["contact_sensor"]
        shape: RigidObject = scene["shape"]
        contact_pose = CUBE_CFG.contact_pose.to(device=shape.device).unsqueeze(0)
        non_contact_pose = CUBE_CFG.non_contact_pose.to(device=shape.device).unsqueeze(0)

        def _in_contact() -> bool:
            """Ground truth for the contact state, read through the public data accessor."""
            return torch.norm(sensor.data.net_normal_forces_w.torch, dim=-1).max().item() > 0.1

        def _hold(pose: torch.Tensor, num_steps: int) -> None:
            """Pin the cube to a pose for the given number of physics steps."""
            for _ in range(num_steps):
                shape.write_root_pose_to_sim_index(root_pose=pose)
                _perform_sim_step(sim, scene, _SIM_DT)

        # Settle the cube on the ground so the sensor starts in contact.
        _hold(contact_pose, 8)
        assert _in_contact(), "Cube should be in contact with the ground before the clock is aged."

        # Age the sensor clock without stepping physics: the resting contact state is unchanged, so
        # this isolates the float32 clock drift from any change in the contact forces.
        for tick in range(int(round(clock_age / _SIM_DT))):
            sensor.update(_SIM_DT)
        aged_clock = wp.to_torch(sensor._timestamp).max().item()
        assert aged_clock == pytest.approx(clock_age + 8 * _SIM_DT, abs=0.05)

        # Lift the cube off the ground for half the window, then set it back down.
        reported_air: list[int] = []
        reported_contact: list[int] = []
        expected_air: list[int] = []
        expected_contact: list[int] = []
        was_in_contact = True
        for step in range(poll_steps):
            _hold(non_contact_pose if step < poll_steps // 2 else contact_pose, decimation)
            # Poll before anything reads ``data`` this step: the query itself must refresh lazily
            # updated buffers, otherwise it reports the previous step's timers.
            first_contact = sensor.compute_first_contact(poll_dt).torch.any().item()
            first_air = sensor.compute_first_air(poll_dt).torch.any().item()
            in_contact = _in_contact()
            if in_contact and not was_in_contact:
                expected_contact.append(step)
            if not in_contact and was_in_contact:
                expected_air.append(step)
            was_in_contact = in_contact
            if first_contact:
                reported_contact.append(step)
            if first_air:
                reported_air.append(step)

        assert len(expected_air) == 1, f"Expected exactly one lift-off in the window; got {expected_air}."
        assert len(expected_contact) == 1, f"Expected exactly one touchdown in the window; got {expected_contact}."
        assert reported_air == expected_air, (
            f"compute_first_air missed or mis-reported the lift-off at clock {aged_clock:.3f}s: "
            f"reported {reported_air}, expected {expected_air}."
        )
        assert reported_contact == expected_contact, (
            f"compute_first_contact missed or mis-reported the touchdown at clock {aged_clock:.3f}s: "
            f"reported {reported_contact}, expected {expected_contact}."
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_cube_stack_contact_filtering(device):
    """Checks contact sensor reporting for filtering stacked cube prims."""
    num_envs = 6
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
            contact_sensor_2.data.normal_force_matrix_w.torch[:, :, 0],
            contact_sensor_2.data.net_normal_forces_w.torch,
        )
        # Check that forces are opposite and equal
        torch.testing.assert_close(
            contact_sensor_2.data.normal_force_matrix_w.torch[:, :, 0],
            -contact_sensor.data.normal_force_matrix_w.torch[:, :, 0],
        )
        # Check values are non-zero (contacts are happening and are getting reported)
        assert contact_sensor_2.data.net_normal_forces_w.torch.sum().item() > 0.0
        assert contact_sensor.data.net_normal_forces_w.torch.sum().item() > 0.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_multi_body_per_sensor_indexing(device):
    """Ground-truth body-index check for a single sensor that resolves to two bodies.

    OVPhysX :class:`ContactBinding` returns sensors in **pattern-major** order
    (``[env_0/body_0, env_1/body_0, …, env_0/body_1, env_1/body_1, …]``),
    whereas the inherited PhysX kernel formula assumes env-major
    (``[env_0/body_0, env_0/body_1, …, env_1/body_0, …]``).  Single-body
    sensors don't disambiguate the two layouts, so this test exercises the
    multi-body discovery path with one cube on the ground and one floating
    above it.  After the scene settles, only the bottom cube should report a
    non-zero net force.  An env-major bug would attribute that force to the
    wrong (env, body) slot — caught here. A single env would make both layouts
    coincide, so the test uses several.
    """
    num_envs = 3
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
            prim_path="{ENV_REGEX_NS}/Cube_[^/]*",
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
        net_forces = contact_sensor.data.net_normal_forces_w.torch
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
        # Without filter_prim_paths_expr there is no per-partner breakdown to report.
        assert contact_sensor.data.normal_force_matrix_w is None


def _author_nested_chain(prim_path: str) -> None:
    """Author a chain of kinematic rigid bodies whose link prims are nested under each other.

    Mirrors the layout produced by the URDF importer in Isaac Sim 6.0+, where each child
    link prim is authored under its parent link prim instead of as a flat sibling. The
    bodies are kinematic so their poses stay at the authored values without joints.
    """
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, prim_path)
    link_specs = [
        ("pelvis", (0.0, 0.0, 1.25)),
        ("pelvis/left_hip", (0.0, 0.0, -0.5)),
        ("pelvis/left_hip/left_knee", (0.0, 0.0, -0.5)),
    ]
    for rel_path, offset in link_specs:
        link_path = f"{prim_path}/{rel_path}"
        link = UsdGeom.Xform.Define(stage, link_path)
        link.AddTranslateOp().Set(Gf.Vec3d(*offset))
        body_api = UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        body_api.CreateKinematicEnabledAttr(True)
        geom = UsdGeom.Cube.Define(stage, f"{link_path}/geom")
        geom.GetSizeAttr().Set(0.5)
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    # add the contact-report schema to every nested link
    schemas.activate_contact_sensors(prim_path)


@pytest.mark.parametrize(
    "device, body_pattern, num_envs, body_names",
    [
        ("cpu", "[^/]*", 3, ["pelvis", "left_hip", "left_knee"]),
        ("cuda:0", "[^/]*", 3, ["pelvis", "left_hip", "left_knee"]),
        ("cpu", ".*/left_knee", 1, ["left_knee"]),
        ("cuda:0", ".*/left_knee", 3, ["left_knee"]),
    ],
)
def test_nested_rigid_body_hierarchy(device, body_pattern, num_envs, body_names):
    """Checks contact binding creation and body resolution on nested rigid-body hierarchies.

    Regression test for the sensor-pattern construction: patterns were built from the
    first matched body's parent plus leaf names, which cannot address bodies nested
    under other bodies, so the contact binding bound only the first-level links and
    initialization failed on URDF-importer-style assets.

    The source chain is authored under ``env_0`` and replicated through the OVPhysX
    clone path so the test covers both nested body resolution and multi-environment
    contact binding behavior. Mid-path wildcards must bind each leaf only once, even
    when several of its ancestors match.
    """
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=False) as sim:
        stage = get_current_stage()
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + body_pattern,
            track_pose=False,
            debug_vis=False,
            update_period=0.0,
        )
        clone_plan = cloner.clone_plan_from_env_0(cloner.CloneCfg(), (contact_sensor_cfg,), num_envs, 3.0)
        assert clone_plan.env_ids is not None and clone_plan.positions is not None
        env_positions = clone_plan.positions
        env_0 = UsdGeom.Xform.Define(stage, "/World/envs/env_0")
        env_0.AddTranslateOp().Set(Gf.Vec3d(*env_positions[0].tolist()))
        _author_nested_chain("/World/envs/env_0/Robot")

        ovphysx_replicate(
            stage,
            clone_plan.sources,
            clone_plan.destinations,
            clone_plan.env_ids,
            clone_plan.clone_mask,
            positions=clone_plan.positions,
        )
        contact_sensor = ContactSensor(contact_sensor_cfg)
        sim.reset()

        assert contact_sensor.num_sensors == len(body_names)
        assert contact_sensor.body_names == body_names

        # step to fill the sensor buffers; kinematic bodies generate no contact forces
        for _ in range(2):
            sim.step()
            contact_sensor.update(_SIM_DT, force_recompute=True)
        net_forces = contact_sensor.data.net_normal_forces_w.torch
        assert net_forces.shape == (num_envs, len(body_names), 3)


# Only USD authoring and the device-independent __str__ are checked, so one device covers them.
@pytest.mark.parametrize("device", ["cpu"])
def test_contact_sensor_threshold(device):
    """Test that the contact sensor USD threshold attribute is set to 0.0.

    Regression for #3498, where the spawner passed ``activate_contact_sensors`` (a bool) as the threshold.
    """
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
        assert "Contact sensor @" in str(contact_sensor)

        assert "PhysxContactReportAPI" in prim.GetAppliedSchemas()
        threshold_value = prim.GetAttribute("physxContactReport:threshold").Get()
        assert threshold_value == pytest.approx(0.0, abs=1e-6), (
            f"Expected USD threshold to be close to 0.0, but got {threshold_value}"
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lazy_sensor_reports_contact_loss(device):
    """Regression for issue #7613: a lazily read sensor must report the loss of contact.

    Mirrors the PhysX regression test: with ``history_length=0`` and ``lazy_sensor_update=True``
    the sensor is only refreshed when :attr:`ContactSensor.data` is accessed, which a policy-rate
    reader does once per four physics steps. The reported force must still drop to zero once the
    shape is held in the air.
    """
    with _ovphysx_sim_context(device=device, dt=_SIM_DT, add_lighting=False) as sim:
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=True)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
        scene_cfg.shape = CUBE_CFG
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path=CUBE_CFG.prim_path,
            track_pose=True,
            update_period=0.0,
            track_air_time=True,
            history_length=0,
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()

        sensor: ContactSensor = scene["contact_sensor"]
        shape: RigidObject = scene["shape"]
        contact_pose = CUBE_CFG.contact_pose.to(device=shape.device).unsqueeze(0)
        non_contact_pose = CUBE_CFG.non_contact_pose.to(device=shape.device).unsqueeze(0)

        # Mimic a policy stepping the environment with a decimation of 4: the sensor data is
        # read once per policy step and left untouched in between.
        decimation = 4
        num_policy_steps = 6

        def run_policy_step(root_pose: torch.Tensor) -> float:
            """Holds the shape at ``root_pose`` for one policy step and reads the sensor once."""
            for _ in range(decimation):
                shape.write_root_pose_to_sim_index(root_pose=root_pose)
                _perform_sim_step(sim, scene, _SIM_DT)
            return torch.linalg.norm(sensor.data.net_normal_forces_w.torch, dim=-1).max().item()

        contact_forces = [run_policy_step(contact_pose) for _ in range(num_policy_steps)]
        air_forces = [run_policy_step(non_contact_pose) for _ in range(num_policy_steps)]

        # Guard against a vacuous test: the sensor must have reported contact while on the ground.
        assert contact_forces[-1] > 0.1, f"Expected a contact force on the ground; got {contact_forces}"
        # The first read in the air may still carry the impulse of the last contact step, so only
        # the subsequent reads are required to be free of contact.
        assert max(air_forces[1:]) < 0.1, f"Stale contact force reported in the air: {air_forces}"


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
