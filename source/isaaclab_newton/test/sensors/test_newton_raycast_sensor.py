# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Newton BVH ray-cast sensor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sensors import (
    LegacyMultiMeshRayCaster,
    LegacyMultiMeshRayCasterCamera,
    LegacyRayCasterCamera,
    NewtonRaycastSensor,
    NewtonRaycastSensorCfg,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCamera, RayCasterCamera, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

SENSOR_HEIGHT = 2.0
RAY_OFFSET = 0.2
RAY_START_HEIGHT = SENSOR_HEIGHT - RAY_OFFSET


@configclass
class RaycastTestSceneCfg(InteractiveSceneCfg):
    """Scene with a ground plane, a floating sensor body, and a dynamic box."""

    env_spacing = 8.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    sensor_body = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SENSOR_HEIGHT)),
    )

    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.0, 0.0, 0.5)),
    )

    raycast = NewtonRaycastSensorCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        # Start the rays below the carrier cube so they do not hit it.
        offset=NewtonRaycastSensorCfg.OffsetCfg(pos=(0.0, 0.0, -RAY_OFFSET)),
        pattern_cfg=GridPatternCfg(resolution=0.2, size=(0.4, 0.4)),
        ray_alignment="yaw",
        max_distance=100.0,
    )


@configclass
class GenericRaycastTestSceneCfg(RaycastTestSceneCfg):
    """Scene using the backend-dispatching ray-caster configuration.

    Sets the Newton-only ``global_world_only`` field on the base
    :class:`~isaaclab.sensors.RayCasterCfg` to verify it is honored after backend dispatch.
    """

    raycast = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, -RAY_OFFSET)),
        pattern_cfg=GridPatternCfg(resolution=0.2, size=(0.4, 0.4)),
        ray_alignment="yaw",
        max_distance=100.0,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )


@pytest.fixture(params=[True, False], ids=["cuda_graph", "eager"])
def sim(request):
    """Newton simulation context, with and without CUDA graphs."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 100.0,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(),
            use_cuda_graph=request.param,
        ),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        yield sim


def _step_and_read(sim, scene) -> NewtonRaycastSensor:
    sim.step()
    scene.update(sim.get_physics_dt())
    return scene["raycast"]


@pytest.mark.parametrize("global_world_only", [False, True])
def test_rays_hit_ground_plane(sim, global_world_only):
    """All rays of a downward grid pattern hit the global-world ground at the sensor height."""
    scene_cfg = RaycastTestSceneCfg(num_envs=2)
    scene_cfg.raycast.global_world_only = global_world_only
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    sensor = _step_and_read(sim, scene)

    hits = sensor.data.ray_hits_w.torch
    distances = sensor.data.ray_distances.torch
    normals = sensor.data.ray_normals_w.torch
    assert hits.shape == (2, sensor.num_rays, 3)
    torch.testing.assert_close(hits[..., 2], torch.zeros_like(hits[..., 2]), atol=1e-3, rtol=0)
    torch.testing.assert_close(distances, torch.full_like(distances, RAY_START_HEIGHT), atol=1e-3, rtol=0)
    expected_normal = torch.tensor([0.0, 0.0, 1.0], device=normals.device).expand_as(normals)
    torch.testing.assert_close(normals, expected_normal, atol=1e-3, rtol=0)


def test_generic_ray_caster_uses_newton_scene_bvh(sim):
    """The backend-dispatching ray caster selects the Newton BVH implementation."""
    scene = InteractiveScene(GenericRaycastTestSceneCfg(num_envs=1))
    sim.reset()
    sensor = _step_and_read(sim, scene)

    assert isinstance(sensor, NewtonRaycastSensor)
    assert hasattr(sensor.data, "ray_distances")
    torch.testing.assert_close(
        sensor.data.ray_distances.torch,
        torch.full_like(sensor.data.ray_distances.torch, RAY_START_HEIGHT),
        atol=1e-3,
        rtol=0,
    )


def test_remaining_warp_mesh_factories_select_legacy_newton_adapters(sim):
    """Camera and multi-mesh factories retain their explicit legacy implementations."""
    assert RayCasterCamera.resolve_class() is LegacyRayCasterCamera
    assert MultiMeshRayCaster.resolve_class() is LegacyMultiMeshRayCaster
    assert MultiMeshRayCasterCamera.resolve_class() is LegacyMultiMeshRayCasterCamera


def test_bvh_refit_tracks_moving_geometry(sim):
    """Sliding a box under the sensor changes the hits, proving the BVH refits live."""
    scene = InteractiveScene(RaycastTestSceneCfg(num_envs=1))
    sim.reset()
    sensor = _step_and_read(sim, scene)
    torch.testing.assert_close(
        sensor.data.ray_distances.torch,
        torch.full_like(sensor.data.ray_distances.torch, RAY_START_HEIGHT),
        atol=1e-3,
        rtol=0,
    )

    # Move the box directly under the sensor: rays now hit its top face at z=1.
    box: RigidObject = scene["box"]
    pose = box.data.root_pose_w.torch.clone()
    pose[:, :2] = 0.0
    box.write_root_pose_to_sim_index(root_pose=pose)
    scene.write_data_to_sim()
    sensor = _step_and_read(sim, scene)

    distances = sensor.data.ray_distances.torch
    torch.testing.assert_close(distances, torch.full_like(distances, RAY_START_HEIGHT - 1.0), atol=1e-3, rtol=0)


@configclass
class RaycastCameraSceneCfg(RaycastTestSceneCfg):
    """Adds a downward-looking Newton tiled camera next to the ray-cast sensor."""

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/SensorBody/cam",
        width=64,
        height=48,
        data_types=["depth"],
        renderer_cfg=NewtonWarpRendererCfg(),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, -RAY_OFFSET), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg(),
    )


def test_renderer_and_raycast_share_newton_manager_graph(sim):
    """Tiled-camera and ray-cast updates share the Newton manager graph."""
    scene = InteractiveScene(RaycastCameraSceneCfg(num_envs=1))
    sim.reset()
    sim.step()
    scene.update(sim.get_physics_dt())

    depth = scene["camera"].data.output["depth"].torch
    distances = scene["raycast"].data.ray_distances.torch
    assert abs(depth[0, 24, 32].item() - RAY_START_HEIGHT) < 5e-3
    torch.testing.assert_close(distances, torch.full_like(distances, RAY_START_HEIGHT), atol=1e-3, rtol=0)

    task_names = sorted(NewtonManager._sensor_tasks)
    assert any(name.startswith("newton_raycast:") for name in task_names)
    assert any(name.startswith("newton_warp_render:") for name in task_names)
    assert (NewtonManager._sensor_graph is not None) == sim.cfg.physics.use_cuda_graph
