# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify contact sensor functionality on rigid object prims."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from dataclasses import MISSING
from enum import Enum
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from flaky import flaky
from isaaclab_physx.sensors.contact_sensor import contact_sensor as contact_sensor_module
from isaaclab_physx.sensors.contact_sensor.contact_sensor import ContactSensor as PhysxContactSensor

from pxr import Gf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sensors.contact_sensor import BaseContactSensor
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.terrains import HfRandomUniformTerrainCfg, TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass

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


@pytest.fixture(scope="module")
def setup_simulation():
    """Fixture to set up simulation parameters."""
    sim_dt = 0.0025
    durations = [sim_dt, sim_dt * 2, sim_dt * 32, sim_dt * 128]
    terrains = [FLAT_TERRAIN_CFG, COBBLESTONE_TERRAIN_CFG]
    devices = ["cuda:0", "cpu"]
    settings = get_settings_manager()
    return sim_dt, durations, terrains, devices, settings


def _make_graph_test_sensor(device: str = "cuda:0") -> PhysxContactSensor:
    """Create a minimal contact sensor for graph unit tests without a USD scene."""
    sensor = PhysxContactSensor.__new__(PhysxContactSensor)
    sensor.cfg = SimpleNamespace(prim_path="/World/Sensor")
    sensor._device = device
    sensor._use_graph = True
    sensor._compute_graph = None
    sensor._initialize_handle = None
    sensor._invalidate_initialize_handle = None
    sensor._prim_deletion_handle = None
    sensor._resolve_indices_and_mask = lambda env_ids, mask: mask
    sensor._fetch_physx_buffers = lambda: None
    return sensor


def test_cuda_graph_capture_launches_first_update_without_eager_compute():
    """The initial graph capture should execute the update through one graph launch."""
    device = "cuda:0"
    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    source = wp.ones(1, dtype=wp.int32, device=device)
    destination = wp.zeros(1, dtype=wp.int32, device=device)
    sensor = _make_graph_test_sensor()

    compute_call_count = 0

    def compute() -> None:
        nonlocal compute_call_count
        compute_call_count += 1
        wp.copy(destination, source)

    sensor._compute = compute
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(device)

    assert compute_call_count == 1
    assert destination.numpy().tolist() == [1]

    source.fill_(2)
    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(device)

    assert compute_call_count == 1
    assert destination.numpy().tolist() == [2]


def test_contact_sensor_refuses_update_inside_outer_graph_capture():
    """Updating the sensor during an outer capture should raise before touching PhysX."""
    device = "cuda:0"
    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    source = wp.ones(1, dtype=wp.int32, device=device)
    destination = wp.zeros(1, dtype=wp.int32, device=device)
    sensor = _make_graph_test_sensor()

    fetch_call_count = 0

    def fetch() -> None:
        nonlocal fetch_call_count
        fetch_call_count += 1

    sensor._fetch_physx_buffers = fetch

    with wp.ScopedCapture(device=wp.get_device(device)):
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            sensor._update_buffers_impl(env_mask)
        # record something so the outer capture does not end empty
        wp.copy(destination, source)

    assert sensor._compute_graph is None
    assert fetch_call_count == 0


def test_contact_sensor_falls_back_when_graph_capture_fails(monkeypatch):
    """A capture failure should disable graphs and still produce data eagerly."""
    device = "cuda:0"
    env_mask = wp.ones(1, dtype=wp.bool, device=device)
    source = wp.ones(1, dtype=wp.int32, device=device)
    destination = wp.zeros(1, dtype=wp.int32, device=device)
    sensor = _make_graph_test_sensor()

    compute_call_count = 0

    def compute() -> None:
        nonlocal compute_call_count
        compute_call_count += 1
        wp.copy(destination, source)

    sensor._compute = compute

    class _FailingCapture:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            raise RuntimeError("capture unavailable")

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(contact_sensor_module.wp, "ScopedCapture", _FailingCapture)

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(device)

    assert sensor._use_graph is False
    assert sensor._compute_graph is None
    assert compute_call_count == 1
    assert destination.numpy().tolist() == [1]

    sensor._update_buffers_impl(env_mask)
    wp.synchronize_device(device)

    assert compute_call_count == 2


def test_contact_sensor_invalidation_drops_cached_graph_state(monkeypatch):
    """Invalidation should clear the captured graph and every cached warp view."""
    sensor = _make_graph_test_sensor()
    monkeypatch.setattr(BaseContactSensor, "_invalidate_initialize_callback", lambda self, event: None)

    cached_attrs = [
        "_body_physx_view",
        "_contact_view",
        "_compute_graph",
        "_net_forces_flat",
        "_force_matrix_flat",
        "_poses_flat",
        "_contact_points_flat",
        "_friction_forces_flat",
        "_friction_counts",
        "_friction_start_indices",
    ]
    for attr in cached_attrs:
        setattr(sensor, attr, object())

    sensor._invalidate_initialize_callback(None)

    for attr in cached_attrs:
        assert getattr(sensor, attr) is None, attr


@pytest.mark.parametrize("disable_contact_processing", [True, False])
@flaky(max_runs=5, min_passes=1)
def test_cube_contact_time(setup_simulation, disable_contact_processing):
    """Checks contact sensor values for contact time and air time for a cube collision primitive."""
    # check for both contact processing enabled and disabled
    # internally, the contact sensor should enable contact processing so it should always work.
    sim_dt, durations, terrains, devices, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", disable_contact_processing)
    _run_contact_sensor_test(CUBE_CFG, sim_dt, devices, terrains, settings, durations)


@pytest.mark.parametrize("disable_contact_processing", [True, False])
@flaky(max_runs=5, min_passes=1)
def test_sphere_contact_time(setup_simulation, disable_contact_processing):
    """Checks contact sensor values for contact time and air time for a sphere collision primitive."""
    # check for both contact processing enabled and disabled
    # internally, the contact sensor should enable contact processing so it should always work.
    sim_dt, durations, terrains, devices, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", disable_contact_processing)
    _run_contact_sensor_test(SPHERE_CFG, sim_dt, devices, terrains, settings, durations)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 6, 24])
def test_cube_stack_contact_filtering(setup_simulation, device, num_envs):
    """Checks contact sensor reporting for filtering stacked cube prims."""
    sim_dt, durations, terrains, devices, settings = setup_simulation
    with build_simulation_context(device=device, dt=sim_dt, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Instance new scene for the current terrain and contact prim.
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

        # Check that contact processing is enabled
        assert not settings.get("/physics/disableContactProcessing")

        # Set variables internally for reference
        sim.reset()

        contact_sensor = scene["contact_sensor"]
        contact_sensor_2 = scene["contact_sensor_2"]

        # Check that contact processing is enabled
        assert contact_sensor.contact_view.filter_count == 1
        assert contact_sensor_2.contact_view.filter_count == 1

        # Play the simulation
        scene.reset()
        for _ in range(500):
            _perform_sim_step(sim, scene, sim_dt)

        # Check values for cube 2 --> cube 1 is the only collision for cube 2
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


def _author_nested_chain(prim_path: str):
    """Author a chain of kinematic rigid bodies whose link prims are nested under each other.

    Mirrors the layout produced by the URDF importer in Isaac Sim 6.0+, where each child
    link prim is authored under its parent link prim instead of as a flat sibling. The
    bodies are kinematic so their poses stay at the authored values without joints.
    """
    stage = get_current_stage()
    UsdGeom.Xform.Define(stage, prim_path)
    # each link: an Xform with the rigid-body schema and a cube collision geometry below it
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


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("num_envs", [1, 6])
def test_nested_rigid_body_hierarchy(setup_simulation, device, num_envs):
    """Checks contact sensor creation and per-body data ordering on nested rigid-body hierarchies.

    Regression test for issue #5126: the sensor views were created from a single
    parent-level name alternation, which cannot address bodies nested under other
    bodies, so sensor initialization failed on URDF-importer-style assets.

    The chains are authored directly under each environment prim (no scene/cloner):
    the sensor path under test only depends on the prims existing on the stage.
    """
    sim_dt, durations, terrains, devices, settings = setup_simulation
    with build_simulation_context(device=device, dt=sim_dt, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None
        stage = get_current_stage()
        env_origins = [(3.0 * env_id, 0.0, 0.0) for env_id in range(num_envs)]
        for env_id, origin in enumerate(env_origins):
            env_xform = UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
            env_xform.AddTranslateOp().Set(Gf.Vec3d(*origin))
            _author_nested_chain(f"/World/envs/env_{env_id}/Robot")
        contact_sensor = ContactSensor(
            ContactSensorCfg(
                prim_path="/World/envs/env_.*/Robot/.*",
                track_pose=True,
                debug_vis=False,
                update_period=0.0,
            )
        )
        sim.reset()

        # all three nested bodies must be resolved into the views (pre-fix: init raised);
        # exact order guards the body-major view convention in body_names
        assert contact_sensor.num_sensors == 3
        assert contact_sensor.body_names == ["pelvis", "left_hip", "left_knee"]

        # step to fill the sensor buffers; kinematic bodies keep their authored poses
        expected_body_z = {"pelvis": 1.25, "left_hip": 0.75, "left_knee": 0.25}
        for _ in range(2):
            sim.step()
            contact_sensor.update(sim_dt, force_recompute=True)

        # each (env, body) cell must carry that body's authored world position, which
        # validates the body-major view ordering end-to-end through the data kernels
        pos_w = contact_sensor.data.pos_w.torch  # (num_envs, num_bodies, 3)
        for env_id, origin in enumerate(env_origins):
            for body_id, body_name in enumerate(contact_sensor.body_names):
                expected = torch.tensor([origin[0], origin[1], expected_body_z[body_name]], device=pos_w.device)
                torch.testing.assert_close(pos_w[env_id, body_id], expected, atol=1e-4, rtol=0.0)


def test_no_contact_reporting(setup_simulation):
    """Test that forcing the disable of contact processing results in no contact reporting.

    We borrow the test :func:`test_cube_stack_contact_filtering` to test this and force disable contact processing.
    """
    # TODO: This test only works on CPU. For GPU, it seems the contact processing is not disabled.
    sim_dt, durations, terrains, devices, settings = setup_simulation
    with build_simulation_context(device="cpu", dt=sim_dt, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Instance new scene for the current terrain and contact prim.
        scene_cfg = ContactSensorSceneCfg(num_envs=2, env_spacing=1.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG
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

        # Force disable contact processing
        settings.set_bool("/physics/disableContactProcessing", True)

        # Set variables internally for reference
        sim.reset()

        # Extract from scene for type hinting
        contact_sensor: ContactSensor = scene["contact_sensor"]
        contact_sensor_2: ContactSensor = scene["contact_sensor_2"]

        # Check buffers have the right size
        assert contact_sensor.contact_view.filter_count == 1
        assert contact_sensor_2.contact_view.filter_count == 1

        # Reset the contact sensors
        scene.reset()
        # Let the scene come to a rest
        for _ in range(500):
            _perform_sim_step(sim, scene, sim_dt)

        # check values are zero (contacts are happening but not reported)
        assert contact_sensor.data.net_forces_w.torch.sum().item() == 0.0
        assert contact_sensor.data.force_matrix_w.torch.sum().item() == 0.0
        assert contact_sensor_2.data.net_forces_w.torch.sum().item() == 0.0
        assert contact_sensor_2.data.force_matrix_w.torch.sum().item() == 0.0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_contact_sensor_no_stale_data_after_reset(setup_simulation, device):
    """Regression for issue #4970: ``scene.reset(env_ids)`` must not surface pre-reset contact data.

    Reproduces the manager-based RL flow where an environment terminates, ``_reset_idx`` writes
    new asset poses and calls :meth:`InteractiveScene.reset`, and the next observation read
    happens before any further physics step. The contact sensor's lazy refetch must not return
    PhysX's stale post-step buffer for the reset env — it must reflect the freshly reset state.
    """
    sim_dt, _, _, _, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", False)
    with build_simulation_context(device=device, dt=sim_dt, add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=False)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube")
        scene_cfg.shape.init_state.pos = (0.0, 0.0, 1.0)
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            update_period=0.0,
            history_length=10,
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        contact_sensor: ContactSensor = scene["contact_sensor"]
        shape: RigidObject = scene["shape"]

        # Drop the cube and step until it has settled on the ground.
        for _ in range(200):
            _perform_sim_step(sim, scene, sim_dt)

        # Sanity: cube is on the ground and reporting non-zero contact force.
        pre_reset_force_mag = torch.linalg.norm(contact_sensor.data.net_forces_w.torch, dim=-1).item()
        assert pre_reset_force_mag > 1.0, f"Expected non-zero contact force before reset; got {pre_reset_force_mag!r}"

        # Mimic ``ManagerBasedRLEnv._reset_idx``: write the post-reset asset pose, then
        # ``scene.reset(env_ids)`` (which resets all sensors). No further physics step runs.
        env_ids = torch.tensor([0], device=shape.device)
        new_root_pose = torch.tensor([[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0]], device=shape.device)
        new_root_vel = torch.zeros((1, 6), device=shape.device)
        shape.write_root_pose_to_sim_index(root_pose=new_root_pose, env_ids=env_ids)
        shape.write_root_velocity_to_sim_index(root_velocity=new_root_vel, env_ids=env_ids)
        scene.reset(env_ids=env_ids)

        # The sensor must not return the cached pre-reset PhysX contact value here.
        post_reset_force_mag = torch.linalg.norm(contact_sensor.data.net_forces_w.torch, dim=-1).item()
        assert post_reset_force_mag == 0.0, (
            "Contact sensor returned stale pre-reset data after scene.reset(): "
            f"got {post_reset_force_mag}, expected 0.0 (pre-reset value was {pre_reset_force_mag})."
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    ("history_length", "update_period_steps", "expected_fetches", "expected_last_update"),
    [(0, 1, 0, 0.0), (3, 1, 4, 0.01), (3, 2, 4, 0.0075)],
)
def test_contact_history_updates_at_sensor_period(
    setup_simulation, device, history_length, update_period_steps, expected_fetches, expected_last_update
):
    """History-bearing sensors update at their period even when the scene is lazy."""
    sim_dt, _, _, _, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", False)
    with build_simulation_context(device=device, dt=sim_dt, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=2.0, lazy_sensor_update=True)
        scene_cfg.terrain = FLAT_TERRAIN_CFG.replace(prim_path="/World/ground")
        scene_cfg.shape = CUBE_CFG.replace(prim_path="{ENV_REGEX_NS}/Cube")
        scene_cfg.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            update_period=update_period_steps * sim_dt,
            track_air_time=True,
            history_length=history_length,
        )
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        scene.reset()

        contact_sensor: ContactSensor = scene["contact_sensor"]
        fetch_count = 0
        original_fetch = contact_sensor._fetch_physx_buffers

        def count_fetches() -> None:
            nonlocal fetch_count
            fetch_count += 1
            original_fetch()

        contact_sensor._fetch_physx_buffers = count_fetches

        for _ in range(4):
            _perform_sim_step(sim, scene, sim_dt)

        assert fetch_count == expected_fetches
        last_update = wp.to_torch(contact_sensor._timestamp_last_update).item()
        assert last_update == pytest.approx(expected_last_update)

        _ = contact_sensor.data
        expected_fetches_after_read = 1 if history_length == 0 else expected_fetches
        assert fetch_count == expected_fetches_after_read
        air_time = contact_sensor.data.current_air_time.torch.item()
        expected_air_time = 4 * sim_dt if history_length == 0 else expected_last_update
        assert air_time == pytest.approx(expected_air_time)
        _ = contact_sensor.data
        assert fetch_count == expected_fetches_after_read


@pytest.mark.isaacsim_ci
def test_sensor_print(setup_simulation):
    """Test sensor print is working correctly."""
    sim_dt, durations, terrains, devices, settings = setup_simulation
    with build_simulation_context(device="cuda:0", dt=sim_dt, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None
        # Spawn things into stage
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
def test_contact_sensor_threshold(setup_simulation, device):
    """Test that the contact sensor USD threshold attribute is set to 0.0."""
    sim_dt, durations, terrains, devices, settings = setup_simulation
    with build_simulation_context(device=device, dt=sim_dt, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None
        # Spawn things into stage
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


# minor gravity force in -z to ensure object stays on ground plane
@pytest.mark.parametrize("grav_dir", [(-10.0, 0.0, -0.1), (0.0, -10.0, -0.1)])
@pytest.mark.isaacsim_ci
def test_friction_reporting(setup_simulation, grav_dir):
    """
    Test friction force reporting for contact sensors.

    This test places a contact sensor enabled cube onto a ground plane under different gravity directions.
    It then compares the normalized friction force dir with the direction of gravity to ensure they are aligned.
    """
    sim_dt, _, _, _, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", True)
    device = "cuda:0"
    sim_cfg = SimulationCfg(dt=sim_dt, device=device, gravity=grav_dir)
    with build_simulation_context(sim_cfg=sim_cfg, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None

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
        scene["shape"].write_root_pose_to_sim_index(
            root_pose=torch.tensor([0, 0.0, CUBE_CFG.spawn.size[2] / 2.0, 1, 0, 0, 0], device=device).unsqueeze(0)
        )

        # step sim once to compute friction forces
        _perform_sim_step(sim, scene, sim_dt)

        # check that forces are being reported match expected friction forces
        expected_friction, _, _, _ = scene["contact_sensor"].contact_view.get_friction_data(dt=sim_dt)
        expected_friction_torch = wp.to_torch(expected_friction)
        reported_friction = scene["contact_sensor"].data.friction_forces_w.torch[0, 0, :]

        torch.testing.assert_close(expected_friction_torch.sum(dim=0), reported_friction[0], atol=1e-6, rtol=1e-5)

        # check that friction force direction opposes gravity direction
        grav = torch.tensor(grav_dir, device=device)
        norm_reported_friction = reported_friction / reported_friction.norm()
        norm_gravity = grav / grav.norm()
        dot = torch.dot(norm_reported_friction[0], norm_gravity)

        torch.testing.assert_close(torch.abs(dot), torch.tensor(1.0, device=device), atol=1e-4, rtol=1e-3)


@pytest.mark.isaacsim_ci
def test_invalid_prim_paths_config(setup_simulation):
    sim_dt, _, _, _, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", True)
    device = "cuda:0"
    sim_cfg = SimulationCfg(dt=sim_dt, device=device)
    with build_simulation_context(sim_cfg=sim_cfg, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None

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


@pytest.mark.isaacsim_ci
def test_invalid_max_contact_points_config(setup_simulation):
    sim_dt, _, _, _, settings = setup_simulation
    settings.set_bool("/physics/disableContactProcessing", True)
    device = "cuda:0"
    sim_cfg = SimulationCfg(dt=sim_dt, device=device)
    with build_simulation_context(sim_cfg=sim_cfg, add_lighting=False) as sim:
        sim._app_control_on_stop_handle = None

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


"""
Internal helpers.
"""


def _run_contact_sensor_test(
    shape_cfg: ContactSensorRigidObjectCfg,
    sim_dt: float,
    devices: list[str],
    terrains: list[TerrainImporterCfg],
    settings,
    durations: list[float],
):
    """
    Runs a rigid body test for a given contact primitive configuration.

    This method iterates through each device and terrain combination in the simulation environment,
    running tests for contact sensors.
    """
    for device in devices:
        for terrain in terrains:
            for track_contact_data in [True, False]:
                with build_simulation_context(device=device, dt=sim_dt, add_lighting=True) as sim:
                    sim._app_control_on_stop_handle = None

                    scene_cfg = ContactSensorSceneCfg(num_envs=1, env_spacing=1.0, lazy_sensor_update=False)
                    scene_cfg.terrain = terrain
                    scene_cfg.shape = shape_cfg
                    test_contact_data = False
                    if (type(shape_cfg.spawn) is sim_utils.SphereCfg) and (terrain.terrain_type == "plane"):
                        test_contact_data = True
                    elif track_contact_data:
                        continue

                    if track_contact_data:
                        if terrain.terrain_type == "plane":
                            filter_prim_paths_expr = [terrain.prim_path + "/terrain/GroundPlane/CollisionPlane"]
                        elif terrain.terrain_type == "generator":
                            filter_prim_paths_expr = [terrain.prim_path + "/terrain/mesh"]
                    else:
                        filter_prim_paths_expr = []

                    scene_cfg.contact_sensor = ContactSensorCfg(
                        prim_path=shape_cfg.prim_path,
                        track_pose=True,
                        debug_vis=False,
                        update_period=0.0,
                        track_air_time=True,
                        history_length=3,
                        track_contact_points=track_contact_data,
                        track_friction_forces=track_contact_data,
                        filter_prim_paths_expr=filter_prim_paths_expr,
                    )
                    scene = InteractiveScene(scene_cfg)

                    # Play the simulation
                    sim.reset()

                    # Run contact time and air time tests.
                    _test_sensor_contact(
                        shape=scene["shape"],
                        sensor=scene["contact_sensor"],
                        mode=ContactTestMode.IN_CONTACT,
                        sim=sim,
                        scene=scene,
                        sim_dt=sim_dt,
                        durations=durations,
                        test_contact_data=test_contact_data,
                    )
                    _test_sensor_contact(
                        shape=scene["shape"],
                        sensor=scene["contact_sensor"],
                        mode=ContactTestMode.NON_CONTACT,
                        sim=sim,
                        scene=scene,
                        sim_dt=sim_dt,
                        durations=durations,
                        test_contact_data=test_contact_data,
                    )


def _test_sensor_contact(
    shape: RigidObject,
    sensor: ContactSensor,
    mode: ContactTestMode,
    sim: SimulationContext,
    scene: InteractiveScene,
    sim_dt: float,
    durations: list[float],
    test_contact_data: bool = False,
):
    """Test for the contact sensor.

    This test sets the contact prim to a pose either in contact or out of contact with the ground plane for
    a known duration. Once the contact duration has elapsed, the data stored inside the contact sensor
    associated with the contact prim is checked against the expected values.

    This process is repeated for all elements in :attr:`TestContactSensor.durations`, where each successive
    contact timing test is punctuated by setting the contact prim to the complement of the desired contact mode
    for 1 sim time-step.

    Args:
        shape: The contact prim used for the contact sensor test.
        sensor: The sensor reporting data to be verified by the contact sensor test.
        mode: The contact test mode: either contact with ground plane or air time.
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

        if test_contact_data:
            _test_contact_position(shape, sensor, mode)
            _test_friction_forces(shape, sensor, mode)

        # switch the contact mode for 1 dt step before the next contact test begins.
        shape.write_root_pose_to_sim_index(root_pose=torch.tensor(reset_pose, device=shape.device).unsqueeze(0))
        # perform simulation step
        _perform_sim_step(sim, scene, sim_dt)
        # set the last air time to 2 sim_dt steps, because last_air_time and last_contact_time
        # adds an additional sim_dt to the total time spent in the previous contact mode for uncertainty in
        # when the contact switch happened in between a dt step.
        expected_last_reset_contact_time = 2 * sim_dt


def _test_friction_forces(shape: RigidObject, sensor: ContactSensor, mode: ContactTestMode) -> None:
    if not sensor.cfg.track_friction_forces:
        assert sensor._data.friction_forces_w is None
        return

    # check shape of the friction_forces_w tensor (wp.to_torch expands vec3f -> float32 trailing dim)
    num_sensors = sensor.num_sensors
    friction_torch = sensor._data.friction_forces_w.torch
    assert friction_torch.shape == (sensor.num_instances // num_sensors, num_sensors, 1, 3)
    # compare friction forces
    if mode == ContactTestMode.IN_CONTACT:
        assert torch.any(torch.abs(friction_torch) > 1e-5).item()
        friction_forces, _, buffer_count, buffer_start_indices = sensor.contact_view.get_friction_data(
            dt=sensor._sim_physics_dt
        )
        friction_forces_t = wp.to_torch(friction_forces)
        buffer_count_t = wp.to_torch(buffer_count).to(torch.int32)
        buffer_start_t = wp.to_torch(buffer_start_indices).to(torch.int32)
        # the raw view buffers are body-major (one view pattern per body):
        # flat index = body_id * num_envs + env_id
        num_envs = sensor.num_instances // num_sensors
        for i in range(sensor.num_instances):
            for j in range(sensor.contact_view.filter_count):
                start_index_ij = buffer_start_t[i, j]
                count_ij = buffer_count_t[i, j]
                force = torch.sum(friction_forces_t[start_index_ij : (start_index_ij + count_ij), :], dim=0)
                env_idx = i % num_envs
                sensor_idx = i // num_envs
                assert torch.allclose(force, friction_torch[env_idx, sensor_idx, j, :], atol=1e-5)

    elif mode == ContactTestMode.NON_CONTACT:
        assert torch.all(friction_torch == 0.0).item()


def _test_contact_position(shape: RigidObject, sensor: ContactSensor, mode: ContactTestMode) -> None:
    """Test for the contact positions (only implemented for sphere and flat terrain)
    checks that the contact position is radius distance away from the root of the object
    Args:
        shape: The contact prim used for the contact sensor test.
        sensor: The sensor reporting data to be verified by the contact sensor test.
        mode: The contact test mode: either contact with ground plane or air time.
    """
    if not sensor.cfg.track_contact_points:
        assert sensor._data.contact_pos_w is None
        return

    # check shape of the contact_pos_w tensor (wp.to_torch expands vec3f -> float32 trailing dim)
    num_sensors = sensor.num_sensors
    contact_pos_torch = sensor._data.contact_pos_w.torch
    assert contact_pos_torch.shape == (sensor.num_instances // num_sensors, num_sensors, 1, 3)
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
    """Checks contact sensor data matches expected values.

    Args:
        sensor: Instance of ContactSensor containing data to be tested.
        expected_air_time: Air time ground truth.
        expected_contact_time: Contact time ground truth.
        expected_last_air_time: Last air time ground truth.
        expected_last_contact_time: Last contact time ground truth.
        dt: Time since previous contact mode switch. If the contact prim left contact 0.1 seconds ago,
            dt should be 0.1 + simulation dt seconds.
    """
    # store current state of the contact prim
    in_air = False
    in_contact = False
    if expected_air_time > 0.0:
        in_air = True
    if expected_contact_time > 0.0:
        in_contact = True
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


def _perform_sim_step(sim, scene, sim_dt):
    """Updates sensors and steps the contact sensor test scene."""
    # write data to simulation
    scene.write_data_to_sim()
    # simulate
    sim.step(render=False)
    # update buffers at sim dt
    scene.update(dt=sim_dt)
