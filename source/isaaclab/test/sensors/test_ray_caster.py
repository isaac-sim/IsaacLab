# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# if not AppLauncher.instance():
#     app_launcher = AppLauncher(headless=True, enable_cameras=True)
#     simulation_app = app_launcher.app
# elif AppLauncher.instance() and AppLauncher.instance()._enable_cameras is False:
#     # FIXME: workaround as AppLauncher instance can currently not be closed without terminating the test
#     raise ValueError("AppLauncher instance exists but enable_cameras is False")
# else:
#     app_launcher = AppLauncher.instance()
#     simulation_app = app_launcher.app

app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaacsim.core.utils.stage as stage_utils
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip

NUM_ENVS = 2
# sample camera poses
POSITION = [2.5, 2.5, 2.5]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]



BOX_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "boxes": terrain_gen.MeshBoxTerrainCfg(
            box_height_range=(1.0, 1.0)
        ),
    },
)
"""Box terrains configuration."""


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration."""

    # terrain - flat terrain plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BOX_TERRAIN_CFG,
        max_init_terrain_level=None,
    )

    # rigid objects - balls
    balls = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        spawn=sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
        ),
    )

    # articulations - robot
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    # sensors
    ray_caster_articulation: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.0, 1.0),
        ),
        attach_yaw_only=True,
    )
    ray_caster_rigid_object: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/ball",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.0, 1.0),
        ),
        attach_yaw_only=True,
    )
    ray_caster_xform: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot",
        mesh_prim_paths=["/World/ground"],
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(1.0, 1.0),
        ),
        attach_yaw_only=True,
    )

@pytest.fixture
def setup_sim():
    """Create a simulation context and scene."""
    # Create a new stage
    stage_utils.create_new_stage()
    # Load simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    # construct scene
    scene_cfg = MySceneCfg(num_envs=NUM_ENVS, env_spacing=5.0, lazy_sensor_update=False)
    yield sim, scene_cfg
    # Cleanup
    sim.clear_all_callbacks()
    sim.clear_instance()
    # Make sure that all mesh instances are resolved
    assert RayCaster.meshes == {}
    assert RayCaster._instance_count == 0

def test_ray_caster_init_articulation(setup_sim):
    sim, scene_cfg = setup_sim
    
    # pop other than the articulation raycaster
    scene_cfg.ray_caster_rigid_object = None
    scene_cfg.ray_caster_xform = None

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Check that raycasters are initialized
    assert scene.sensors["ray_caster_articulation"].is_initialized

    # check correct initialization of meshes, and instances
    assert len(RayCaster.meshes) == 1
    assert RayCaster._instance_count == 1

    # check correct prim paths
    assert scene.sensors["ray_caster_articulation"]._view.prim_paths[0] == "/World/envs/env_0/robot/base"

    # check that buffers exists and have the expected shapes
    assert scene.sensors["ray_caster_articulation"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_articulation"].data.quat_w.shape == (NUM_ENVS, 4)
    num_rays = (scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_articulation"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)

    # check placement equal to articulation
    torch.testing.assert_close(
        scene.articulations["robot"].data.root_pos_w,
        scene.sensors["ray_caster_articulation"].data.pos_w,
        atol=1e-5,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        scene.articulations["robot"].data.root_quat_w,
        scene.sensors["ray_caster_articulation"].data.quat_w,
        atol=1e-5,
        rtol=1e-3,
    )

def test_ray_caster_init_rigid_object(setup_sim):
    sim, scene_cfg = setup_sim

    # pop other than the rigid object raycaster
    scene_cfg.ray_caster_articulation = None
    scene_cfg.ray_caster_xform = None

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Check that raycasters are initialized
    assert scene.sensors["ray_caster_rigid_object"].is_initialized

    # check correct initialization of meshes, and instances
    assert len(RayCaster.meshes) == 1
    assert RayCaster._instance_count == 1

    # check correct prim paths
    assert scene.sensors["ray_caster_rigid_object"]._view.prim_paths[0] == "/World/envs/env_0/ball"

    # check that buffers exists and have the expected shapes
    assert scene.sensors["ray_caster_rigid_object"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_rigid_object"].data.quat_w.shape == (NUM_ENVS, 4)
    num_rays = (scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_rigid_object"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)

    # check placement equal to rigid object
    torch.testing.assert_close(
        scene.rigid_objects["balls"].data.root_pos_w,
        scene.sensors["ray_caster_rigid_object"].data.pos_w,
        atol=1e-5,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        scene.rigid_objects["balls"].data.root_quat_w,
        scene.sensors["ray_caster_rigid_object"].data.quat_w,
        atol=1e-5,
        rtol=1e-3,
    )    

def test_ray_caster_init_xform(setup_sim):
    sim, scene_cfg = setup_sim

    # pop other than the xform raycaster
    scene_cfg.ray_caster_rigid_object = None
    scene_cfg.ray_caster_articulation = None

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Check that raycasters are initialized
    assert scene.sensors["ray_caster_xform"].is_initialized

    # check correct initialization of meshes, and instances
    assert len(RayCaster.meshes) == 1
    assert RayCaster._instance_count == 1

    # check correct prim paths
    assert scene.sensors["ray_caster_xform"]._view.prim_paths[0] == "/World/envs/env_0/robot"

    # check that buffers exists and have the expected shapes
    assert scene.sensors["ray_caster_xform"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_xform"].data.quat_w.shape == (NUM_ENVS, 4)
    num_rays = (scene.sensors["ray_caster_xform"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_xform"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_xform"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_xform"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_xform"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)

def test_ray_caster_multi_init(setup_sim):
    sim, scene_cfg = setup_sim

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Check that raycasters are initialized
    assert scene.sensors["ray_caster_articulation"].is_initialized
    assert scene.sensors["ray_caster_rigid_object"].is_initialized
    assert scene.sensors["ray_caster_xform"].is_initialized

    # check correct initialization of meshes, and instances
    assert len(RayCaster.meshes) == 1
    assert RayCaster._instance_count == 3

    # check correct prim paths
    assert scene.sensors["ray_caster_articulation"]._view.prim_paths[0] == "/World/envs/env_0/robot/base"
    assert scene.sensors["ray_caster_rigid_object"]._view.prim_paths[0] == "/World/envs/env_0/ball"
    assert scene.sensors["ray_caster_xform"]._view.prim_paths[0] == "/World/envs/env_0/robot"

    # check that buffers exists and have the expected shapes
    assert scene.sensors["ray_caster_articulation"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_rigid_object"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_xform"].data.pos_w.shape == (NUM_ENVS, 3)
    assert scene.sensors["ray_caster_articulation"].data.quat_w.shape == (NUM_ENVS, 4)
    assert scene.sensors["ray_caster_rigid_object"].data.quat_w.shape == (NUM_ENVS, 4)
    assert scene.sensors["ray_caster_xform"].data.quat_w.shape == (NUM_ENVS, 4)
    num_rays = (scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_articulation"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_articulation"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)
    num_rays = (scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_rigid_object"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_rigid_object"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)
    num_rays = (scene.sensors["ray_caster_xform"].cfg.pattern_cfg.size[0] / scene.sensors["ray_caster_xform"].cfg.pattern_cfg.resolution + 1) * (scene.sensors["ray_caster_xform"].cfg.pattern_cfg.size[1] / scene.sensors["ray_caster_xform"].cfg.pattern_cfg.resolution + 1)
    assert scene.sensors["ray_caster_xform"].data.ray_hits_w.shape == (NUM_ENVS, num_rays, 3)

def test_ray_hits_w(setup_sim):
    sim, scene_cfg = setup_sim

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # get the ray hits
    ray_hits_w = scene.sensors["ray_caster_articulation"].data.ray_hits_w
    ray_hits_w_rigid_object = scene.sensors["ray_caster_rigid_object"].data.ray_hits_w
    ray_hits_w_xform = scene.sensors["ray_caster_xform"].data.ray_hits_w



def test_ray_caster_offset(setup_sim):
    sim, scene_cfg = setup_sim

    scene_cfg.ray_caster_articulation.offset = RayCasterCfg.OffsetCfg(pos=POSITION, rot=QUAT_WORLD)
    scene_cfg.ray_caster_rigid_object.offset = RayCasterCfg.OffsetCfg(pos=POSITION, rot=QUAT_WORLD)

    # create scene and reset sim
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # check offset is correctly applied to rays
    torch.testing.assert_close(
        scene.sensors["ray_caster_articulation"].ray_origins,
        scene.sensors["ray_caster_rigid_object"].ray_origins,
        atol=1e-5,
        rtol=1e-3,
    )
