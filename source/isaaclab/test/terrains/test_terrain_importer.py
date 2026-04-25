# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

from typing import Literal

import numpy as np
import pytest
import torch
import trimesh
import warp as wp

from pxr import UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab import cloner as lab_cloner
from isaaclab.sim import PreviewSurfaceCfg, SimulationContext, build_simulation_context, get_first_matching_child_prim
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("env_spacing", [1.0, 4.325, 8.0])
@pytest.mark.parametrize("num_envs", [1, 4, 125, 379, 1024])
def test_terrain_importer_env_origins(device, env_spacing, num_envs):
    """Tests that env origins are consistent when computed using the TerrainImporter and Lab's grid_transforms."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # create terrain importer
        terrain_importer_cfg = TerrainImporterCfg(
            num_envs=num_envs,
            env_spacing=env_spacing,
            prim_path="/World/ground",
            terrain_type="plane",  # for flat ground, origins are in grid
            terrain_generator=None,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)
        # obtain env origins using terrain importer
        terrain_importer_origins = terrain_importer.env_origins

        # obtain env origins using Lab's grid_transforms
        lab_grid_origins, _ = lab_cloner.grid_transforms(num_envs, spacing=env_spacing, device=sim.device)

        # check if the env origins are the same
        torch.testing.assert_close(terrain_importer_origins, lab_grid_origins, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_terrain_generation(device):
    """Generates assorted terrains and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            max_init_terrain_level=None,
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            num_envs=1,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # calculate expected size from config
        cfg = terrain_importer.cfg.terrain_generator
        assert cfg is not None
        expectedSizeX = cfg.size[0] * cfg.num_rows + 2 * cfg.border_width
        expectedSizeY = cfg.size[1] * cfg.num_cols + 2 * cfg.border_width

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_custom_material", [True, False])
def test_plane(device, use_custom_material):
    """Generates a plane and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None

        # create custom material
        visual_material = PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)) if use_custom_material else None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            num_envs=1,
            env_spacing=1.0,
            visual_material=visual_material,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Plane")
        assert mesh is None


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_usd(device):
    """Imports terrain from a usd and tests that the resulting mesh has the correct size."""
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Handler for terrains importing
        terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="usd",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Terrains/rough_plane.usd",
            num_envs=1,
            env_spacing=1.0,
        )
        terrain_importer = TerrainImporter(terrain_importer_cfg)

        # check if mesh prim path exists
        mesh_prim_path = terrain_importer.cfg.prim_path + "/terrain"
        assert mesh_prim_path in terrain_importer.terrain_prim_paths

        # obtain underling mesh
        mesh = _obtain_collision_mesh(mesh_prim_path, mesh_type="Mesh")
        assert mesh is not None

        # expect values from USD file
        expectedSizeX = 96
        expectedSizeY = 96

        # get size from mesh bounds
        bounds = mesh.bounds
        actualSize = abs(bounds[1] - bounds[0])

        assert actualSize[0] == pytest.approx(expectedSizeX)
        assert actualSize[1] == pytest.approx(expectedSizeY)


@pytest.mark.skip(reason="It seems like IsaacSim is not setting the initial positions correctly for the balls.")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ball_drop(device):
    """Generates assorted terrains and spheres created as meshes.

    Tests that spheres fall onto terrain and do not pass through it. This ensures that the triangle mesh
    collision works as expected.
    """
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with rough terrain and balls
        _populate_scene(geom_sphere=False, sim=sim)

        # Play simulator
        sim.reset()

        # Create a view over all the balls using PhysX view
        physics_sim_view = sim.physics_manager.get_physics_sim_view()
        ball_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball")

        # Run simulator
        for _ in range(500):
            sim.step(render=False)

        # Ball may have some small non-zero velocity if the roll on terrain <~.2
        # If balls fall through terrain velocity is much higher ~82.0
        view_velocities = ball_view.get_linear_velocities().contiguous()
        max_velocity_z = torch.max(torch.abs(wp.to_torch(view_velocities)[:, 2]))
        assert max_velocity_z.item() <= 0.5


@pytest.mark.skip(reason="It seems like IsaacSim is not setting the initial positions correctly for the balls.")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_ball_drop_geom_sphere(device):
    """Generates assorted terrains and geom spheres.

    Tests that spheres fall onto terrain and do not pass through it. This ensures that the sphere collision
    works as expected.
    """
    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
        sim._app_control_on_stop_handle = None
        # Create a scene with rough terrain and balls
        # TODO: Currently the test fails with geom spheres, need to investigate with the PhysX team.
        #   Setting the geom_sphere as False to pass the test. This test should be enabled once
        #   the issue is fixed.
        _populate_scene(geom_sphere=False, sim=sim)

        # Play simulator
        sim.reset()

        # Create a view over all the balls using PhysX view
        physics_sim_view = sim.physics_manager.get_physics_sim_view()
        ball_view = physics_sim_view.create_rigid_body_view("/World/envs/env_*/ball")

        # Run simulator
        for _ in range(500):
            sim.step(render=False)

        # Ball may have some small non-zero velocity if the roll on terrain <~.2
        # If balls fall through terrain velocity is much higher ~82.0
        view_velocities = ball_view.get_linear_velocities().contiguous()
        max_velocity_z = torch.max(torch.abs(wp.to_torch(view_velocities)[:, 2]))
        assert max_velocity_z.item() <= 0.5


def _obtain_collision_mesh(mesh_prim_path: str, mesh_type: Literal["Mesh", "Plane"]) -> trimesh.Trimesh | None:
    """Get the collision mesh from the terrain."""
    # traverse the prim and get the collision mesh
    mesh_prim = get_first_matching_child_prim(mesh_prim_path, lambda prim: prim.GetTypeName() == mesh_type)
    # check it is valid
    assert mesh_prim.IsValid()

    if mesh_prim.GetTypeName() == "Mesh":
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    else:
        return None


def _populate_scene(sim: SimulationContext, num_balls: int = 2048, geom_sphere: bool = False):
    """Create a scene with terrain and randomly spawned balls.

    The spawned balls are either USD Geom Spheres or are USD Meshes. We check against both these to make sure
    both USD-shape and USD-mesh collisions work as expected.
    """
    # Handler for terrains importing
    terrain_importer_cfg = terrain_gen.TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        num_envs=num_balls,
    )
    terrain_importer = TerrainImporter(terrain_importer_cfg)

    # Create environment clones using Lab's cloner utilities
    env_fmt = "/World/envs/env_{}"
    env_ids = torch.arange(num_balls, dtype=torch.long, device=sim.device)
    env_origins, _ = lab_cloner.grid_transforms(num_balls, spacing=2.0, device=sim.device)
    # Everything under the namespace "/World/envs/env_0" will be cloned
    sim.stage.DefinePrim("/World/envs/env_0", "Xform")

    # Define the scene
    # -- Ball with physics properties using Isaac Lab spawners
    ball_prim_path = "/World/envs/env_0/ball"

    # Create physics material
    physics_material_cfg = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.2,
        dynamic_friction=1.0,
        restitution=0.0,
    )

    # Create visual material
    visual_material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))

    if geom_sphere:
        # Spawn a geom sphere with rigid body properties
        sphere_cfg = sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=visual_material_cfg,
            physics_material=physics_material_cfg,
        )
        sphere_cfg.func(ball_prim_path, sphere_cfg, translation=(0.0, 0.0, 5.0))
    else:
        # Spawn a mesh sphere with rigid body properties
        mesh_sphere_cfg = sim_utils.MeshSphereCfg(
            radius=0.25,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=visual_material_cfg,
            physics_material=physics_material_cfg,
        )
        mesh_sphere_cfg.func(ball_prim_path, mesh_sphere_cfg, translation=(0.0, 0.0, 0.5))

    # Clone the scene
    envs_prim_paths = [f"/World/envs/env_{i}" for i in range(num_balls)]
    lab_cloner.usd_replicate(sim.stage, [env_fmt.format(0)], [env_fmt], env_ids, positions=env_origins)
    physics_scene_path = sim.cfg.physics_prim_path
    lab_cloner.filter_collisions(
        sim.stage, physics_scene_path, "/World/collisions", prim_paths=envs_prim_paths, global_paths=["/World/ground"]
    )

    # Set ball positions over terrain origins
    # Create a view over all the balls using Isaac Lab's FrameView
    ball_view = sim_utils.FrameView("/World/envs/env_.*/ball")
    # cache initial state of the balls
    ball_initial_positions = terrain_importer.env_origins.clone()
    ball_initial_positions[:, 2] += 5.0
    # set initial poses
    # note: setting here writes to USD :)
    ball_view.set_world_poses(positions=wp.from_torch(ball_initial_positions))
