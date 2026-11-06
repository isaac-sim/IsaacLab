# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Integration tests for ray caster sensor view paths, env_mask, and intrinsics.

These tests require Isaac Sim (the Kit launcher). They cover the integration-level
items from ``TODO_ray_caster_kernel_tests.md``:

- ``_get_sensor_transforms_wp`` RigidBodyView path
- ``MultiMeshRayCaster`` env_mask behavior
- ``MultiMeshRayCasterCamera.set_intrinsic_matrices`` propagation
- ``_update_mesh_transforms`` non-identity orientation offset
- Depth clipping ordering for ``MultiMeshRayCasterCamera``
"""

from isaaclab.assets import AssetBaseCfg
from isaaclab.test.utils import launch_test_simulation, resolve_test_sim_device

launch_test_simulation(enable_cameras=True)

import copy
import math
from typing import Any, cast

import numpy as np
import pytest
import torch
import warp as wp

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.cloner import CloneCfg, ClonePlan, InclusionSet, PrototypeWorldTopology
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import (
    MultiMeshRayCaster,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    MultiMeshRayCasterCfg,
    RayCaster,
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import instantiate
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

pytestmark = pytest.mark.integration

_GROUND_PATH = "/World/Ground"
_DT = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim_and_ground():
    """Create a blank stage with a flat ground plane at z=0."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=_DT, device=resolve_test_sim_device()))
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh(_GROUND_PATH, mesh)
    return sim


def _single_downward_ray_cfg(prim_path: str) -> RayCasterCfg:
    """RayCasterCfg with a single downward ray, no offset, world alignment."""
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment="world",
    )


def _spawn_cube_part(part_path: str, translation: tuple[float, float, float]) -> None:
    """Create a small mesh-bearing part under an Xform target."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(part_path, "Xform", translation=translation, stage=stage)
    cube = cast(Any, UsdGeom.Cube.Define(stage, f"{part_path}/Mesh"))
    cube.CreateSizeAttr().Set(0.35)


@pytest.fixture
def sim_ground():
    sim = _make_sim_and_ground()
    yield sim
    sim.stop()
    sim.clear_instance()


# ---------------------------------------------------------------------------
# _get_sensor_transforms_wp: RigidBodyView path
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_rigid_body_view_path(sim_ground):
    """Mount a ray caster on a prim with RigidBodyAPI (no ArticulationRootAPI).

    Exercises the ``RigidBodyView.get_transforms()`` path in
    :meth:`_get_sensor_transforms_wp`.
    """
    sim = sim_ground
    expected_pos = (1.0, 2.0, 6.0)

    prim_path = "/World/RigidBody"
    sim_utils.create_prim(prim_path, "Xform", translation=expected_pos)
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = cast(Any, UsdPhysics.MassAPI.Apply(prim))
    mass_api.CreateMassAttr().Set(1.0)
    cube_path = f"{prim_path}/CollisionCube"
    cube_geom = cast(Any, UsdGeom.Cube.Define(stage, cube_path))
    cube_geom.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))

    sensor = RayCaster(_single_downward_ray_cfg(prim_path))
    sim.reset()
    sensor.update(_DT)

    pos_w = sensor.data.pos_w.torch[0].cpu().numpy()
    np.testing.assert_allclose(
        pos_w,
        expected_pos,
        atol=0.15,
        err_msg="RigidBodyView: sensor pos_w must match initial prim position",
    )

    hits = sensor.data.ray_hits_w.torch[0, 0].cpu().numpy()
    assert abs(hits[2]) < 0.5, f"RigidBodyView: downward ray should hit near z=0, got z={hits[2]}"


@pytest.mark.isaacsim_ci
def test_polygon_mesh_hits_outer_surface(sim_ground):
    """A six-quad box reports outer surfaces rather than artificial faces across the box interior."""
    stage = sim_utils.get_current_stage()
    mesh = UsdGeom.Mesh.Define(stage, "/World/QuadBox")
    mesh.CreatePointsAttr(
        [
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 2, 1, 4, 5, 6, 7, 0, 1, 5, 4, 1, 2, 6, 5, 2, 3, 7, 6, 3, 0, 4, 7])
    sim_utils.create_prim("/World/Scanner", "Xform", translation=(0.0, 0.0, 3.0))
    sensor = instantiate(
        RayCasterCfg(
            prim_path="/World/Scanner",
            mesh_prim_paths=["/World/QuadBox"],
            offset=RayCasterCfg.OffsetCfg(pos=(0.2, 0.0, 0.3)),
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(1.0, 3.0)),
            ray_alignment="world",
            max_distance=2.5,
        )
    )
    sim_ground.reset()
    sensor.update(_DT)
    data = sensor.data
    starts = sensor.ray_starts.torch
    inside = starts[:, :, 1].abs() < 1.0
    assert inside.sum() == 4
    expected_hits = starts[inside].clone()
    expected_hits[:, 2] = 1.0
    torch.testing.assert_close(data.ray_hits_w.torch[inside], expected_hits)
    assert torch.isinf(data.ray_hits_w.torch[~inside]).all()

    # Move origins beyond the configured range; misses must replace the previous hits.
    sensor.drift.warp.fill_(wp.vec3f(0.0, 0.0, 0.5))
    sensor.update(_DT)
    assert torch.isinf(sensor.data.ray_hits_w.torch).all()


# ---------------------------------------------------------------------------
# MultiMeshRayCasterCamera.set_intrinsic_matrices
# ---------------------------------------------------------------------------


@pytest.fixture
def sim_ground_camera():
    """Fixture providing sim + a base MultiMeshRayCasterCameraCfg."""
    sim = _make_sim_and_ground()

    camera_cfg = MultiMeshRayCasterCameraCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0,
        offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 5.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        ),
        data_types=["distance_to_camera"],
    )

    sim_utils.create_prim("/World/Camera", "Xform")

    yield sim, camera_cfg

    sim.stop()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_multi_mesh_camera_set_intrinsic_matrices(sim_ground_camera):
    """Depth output must change when intrinsics are updated on MultiMeshRayCasterCamera.

    The multi-mesh variant overrides ``_initialize_rays_impl`` without calling
    ``super()``, so the warp view refresh path may differ from RayCasterCamera.
    This test verifies that ``set_intrinsic_matrices`` actually takes effect.
    """
    sim, camera_cfg = sim_ground_camera

    camera = MultiMeshRayCasterCamera(cfg=camera_cfg)
    sim.reset()

    # Capture output with default intrinsics
    for _ in range(3):
        sim.step()
        camera.update(_DT)
    camera_output = cast(dict[str, Any], camera.data.output)
    output_before = camera_output["distance_to_camera"].torch.clone()

    # Change to a very different intrinsic matrix (different FOV)
    new_matrix = torch.tensor(
        [[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
        device=camera.device,
    ).unsqueeze(0)
    camera.set_intrinsic_matrices(new_matrix, focal_length=1.0)

    for _ in range(3):
        sim.step()
        camera.update(_DT)
    camera_output = cast(dict[str, Any], camera.data.output)
    output_after = camera_output["distance_to_camera"].torch.clone()

    assert not torch.allclose(output_before, output_after, atol=1e-3), (
        "MultiMeshRayCasterCamera: depth output must change after set_intrinsic_matrices; "
        "unchanged output indicates stale warp ray buffers."
    )
    assert not torch.any(torch.isnan(output_after)), "No NaN values expected after intrinsics update"


# ---------------------------------------------------------------------------
# Depth clipping ordering for MultiMeshRayCasterCamera
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_multi_mesh_camera_d2ip_and_d2c_independent(sim_ground_camera):
    """Requesting both d2ip and d2c simultaneously must produce correct independent results.

    The two outputs are clipped independently from the same raw ray-distance buffer.
    This test verifies the data types do not interfere with each other.
    """
    sim, base_cfg = sim_ground_camera

    joint_cfg = copy.deepcopy(base_cfg)
    joint_cfg.prim_path = "/World/CameraJoint"
    joint_cfg.data_types = ["distance_to_image_plane", "distance_to_camera"]
    joint_cfg.max_distance = 4.5  # camera is 5 m up, so some rays should be clipped
    joint_cfg.depth_clipping_behavior = "max"
    sim_utils.create_prim("/World/CameraJoint", "Xform")
    cam_joint = MultiMeshRayCasterCamera(joint_cfg)

    d2ip_cfg = copy.deepcopy(base_cfg)
    d2ip_cfg.prim_path = "/World/CameraD2IP"
    d2ip_cfg.data_types = ["distance_to_image_plane"]
    d2ip_cfg.max_distance = 4.5
    d2ip_cfg.depth_clipping_behavior = "max"
    sim_utils.create_prim("/World/CameraD2IP", "Xform")
    cam_d2ip = MultiMeshRayCasterCamera(d2ip_cfg)

    d2c_cfg = copy.deepcopy(base_cfg)
    d2c_cfg.prim_path = "/World/CameraD2C"
    d2c_cfg.data_types = ["distance_to_camera"]
    d2c_cfg.max_distance = 4.5
    d2c_cfg.depth_clipping_behavior = "max"
    sim_utils.create_prim("/World/CameraD2C", "Xform")
    cam_d2c = MultiMeshRayCasterCamera(d2c_cfg)

    sim.reset()

    cam_joint.update(_DT)
    cam_d2ip.update(_DT)
    cam_d2c.update(_DT)

    joint_output = cast(dict[str, Any], cam_joint.data.output)
    d2ip_output = cast(dict[str, Any], cam_d2ip.data.output)
    d2c_output = cast(dict[str, Any], cam_d2c.data.output)
    d2ip_joint = joint_output["distance_to_image_plane"].torch
    d2c_joint = joint_output["distance_to_camera"].torch
    d2ip_solo = d2ip_output["distance_to_image_plane"].torch
    d2c_solo = d2c_output["distance_to_camera"].torch

    # Joint camera must match solo cameras (clipping one must not corrupt the other)
    torch.testing.assert_close(d2ip_joint, d2ip_solo, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d2c_joint, d2c_solo, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MultiMeshRayCaster env_mask behavior
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_multi_mesh_tracks_heterogeneous_links_from_clone_sources(sim_ground):
    """Exact body bindings preserve reordered links, missing target slots and static ground geometry."""
    sim = sim_ground
    num_envs = 3
    stage = sim_utils.get_current_stage()
    link_x = {"base": 0.0, "link_a": 0.5, "link_b": 1.0}
    link_size = {"base": 0.1, "link_a": 0.1, "link_b": 0.2}

    sim_utils.create_prim("/World/envs", "Xform", stage=stage)
    for env_id, links in enumerate((("link_a", "link_b"), ("link_a",), ("link_b", "link_a"))):
        env_path = f"/World/envs/env_{env_id}"
        sim_utils.create_prim(env_path, "Xform", translation=(3.0 * env_id, 0.0, 0.0), stage=stage)
        sim_utils.create_prim(f"{env_path}/Sensor", "Xform", translation=(0.75, 0.0, 2.0), stage=stage)
        sim_utils.create_prim(f"{env_path}/Chain", "Xform", stage=stage)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"{env_path}/Chain"))
        for link in ("base", *links):
            link_path = f"{env_path}/Chain/{link}"
            sim_utils.create_prim(link_path, "Xform", translation=(link_x[link], 0.0, 1.0), stage=stage)
            UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(link_path))
            box = UsdGeom.Cube.Define(stage, f"{link_path}/Collision")
            box.CreateSizeAttr(link_size[link])
            UsdPhysics.CollisionAPI.Apply(box.GetPrim())
            # Env 2 has native bodies but takes its visual geometry from the env 0 prototype.
            if env_id != 2:
                UsdGeom.Cube.Define(stage, f"{link_path}/visuals").CreateSizeAttr(link_size[link])
            joint = UsdPhysics.FixedJoint.Define(stage, f"{env_path}/Chain/joint_{link}")
            if link == "base":
                joint.CreateLocalPos0Attr((3.0 * env_id, 0.0, 1.0))
            else:
                joint.CreateBody0Rel().SetTargets([f"{env_path}/Chain/base"])
                joint.CreateLocalPos0Attr((link_x[link], 0.0, 0.0))
            joint.CreateBody1Rel().SetTargets([link_path])

    plan = ClonePlan(
        PrototypeWorldTopology(
            num_asset_prototypes=2,
            world_prototypes=np.array([0, 1]),
            world_prototype_starts=np.array([0, 0, 1, 2]),
            world_prototype_layout=np.array([0, 1, 0]),
        ),
        asset_cfgs=(AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Chain"),) * 2,
    )
    sim.set_clone_plan(plan)

    cfg = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Sensor",
        mesh_prim_paths=[
            _GROUND_PATH,
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="{ENV_REGEX_NS}/Chain/link_[^/]*/visuals",
                track_mesh_transforms=True,
            ),
        ],
        update_period=0,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=(0.5, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment="world",
    )
    sensor = MultiMeshRayCaster(cfg)
    sim.reset()
    sensor.update(_DT, force_recompute=True)

    assert not stage.GetPrimAtPath("/World/envs/env_2/Chain/link_a/visuals")
    expected = torch.tensor(
        [[[3.0 * env_id + 0.5, 0.0, 1.05], [3.0 * env_id + 1.0, 0.0, 1.1]] for env_id in range(num_envs)]
    )
    expected[1, 1, 2] = 0.0
    torch.testing.assert_close(sensor.data.ray_hits_w.torch.cpu(), expected, atol=1e-3, rtol=0.0)


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("rigid_mount", [False, True], ids=["static_frames", "rigid_bodies"])
def test_ray_caster_rejects_mounts_missing_from_environments(sim_ground, rigid_mount):
    """Mounts from a subset of worlds cannot fill the sensor rows used by environment resets."""
    scene_cfg = InteractiveSceneCfg(
        num_envs=4,
        env_spacing=2.0,
        clone_cfg=CloneCfg(clone_combinations=[InclusionSet(assets=["frame_a", "frame_b"]), InclusionSet(assets=[])]),
    )
    for name in ("a", "b"):
        setattr(
            scene_cfg,
            f"frame_{name}",
            AssetBaseCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Frame{name.upper()}",
                spawn=sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    rigid_props=sim_utils.UsdPhysicsRigidBodyCfg() if rigid_mount else None,
                    mass_props=sim_utils.MassCfg(mass=1.0) if rigid_mount else None,
                    collision_props=sim_utils.UsdPhysicsCollisionCfg() if rigid_mount else None,
                ),
            ),
        )
    scene_cfg.scanner = _single_downward_ray_cfg("{ENV_REGEX_NS}/Frame[AB]")
    scene = instantiate(scene_cfg)
    with pytest.raises(RuntimeError, match="found 2 instances in 4 environments"):
        sim_ground.reset()
    assert not scene["scanner"].is_initialized


@pytest.mark.isaacsim_ci
def test_multi_mesh_env_mask_preserves_masked_buffers(sim_ground):
    """Masked environments must retain their pre-update buffer values.

    Creates a single-env MultiMeshRayCaster, captures output after one update,
    then calls ``_update_buffers_impl`` with the environment masked out and
    verifies the output buffers are unchanged.
    """
    sim = sim_ground

    prim_path = "/World/Sensor"
    sim_utils.create_prim(prim_path, "Xform", translation=(0.0, 0.0, 3.0))

    cfg = MultiMeshRayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment="world",
    )
    sensor = MultiMeshRayCaster(cfg)
    sim.reset()

    # First update: populate buffers with real values
    sensor.update(_DT)
    hits_before = sensor.data.ray_hits_w.torch.clone()

    # Second update with env masked out: buffers must not change
    mask_all_false = wp.array([False], dtype=wp.bool, device=sensor.device)
    sensor._update_buffers_impl(mask_all_false)

    hits_after = sensor.data.ray_hits_w.torch
    torch.testing.assert_close(
        hits_after,
        hits_before,
        atol=0.0,
        rtol=0.0,
        msg="Masked env: ray_hits_w must be unchanged after update with env masked out",
    )


# ---------------------------------------------------------------------------
# _update_mesh_transforms: non-identity orientation offset
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_update_mesh_transforms_non_identity_offset(sim_ground):
    """Tracked mesh geometry must account for body orientation when applying offset.

    Setup: a kinematic rigid body at (0, 0, 2) rotated 90 deg around Z, with a
    child mesh offset by (1, 0, 0) in the body's local frame.

    Correct world position of the baked geometry = body_pos + rotate(body_ori, local_offset)
        = (0, 0, 2) + rotate(90degZ, (1, 0, 0))
        = (0, 0, 2) + (0, 1, 0)
        = (0, 1, 2)

    The mesh pose table stores the owner-body pose; the offset is baked into the
    Warp mesh vertices so backend body poses remain the single runtime pose source.
    """
    sim = sim_ground

    from isaaclab.utils.math import quat_from_euler_xyz

    # 90 deg yaw quaternion in xyzw
    yaw90 = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi / 2]))
    yaw90_xyzw = tuple(yaw90[0].tolist())

    # Create a kinematic rigid body at (0, 0, 2) rotated 90 deg around Z
    body_path = "/World/DynamicBody"
    sim_utils.create_prim(body_path, "Xform", translation=(0.0, 0.0, 2.0), orientation=yaw90_xyzw)
    stage = sim_utils.get_current_stage()
    body_prim = cast(Any, stage.GetPrimAtPath(body_path))
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    mass_api = cast(Any, UsdPhysics.MassAPI.Apply(body_prim))
    mass_api.CreateMassAttr().Set(1.0)
    body_prim.GetAttribute("physics:kinematicEnabled").Set(True)

    # Create a child Xform offset by (1, 0, 0) in the body's local frame,
    # then place mesh geometry under it. The tracked target view resolves this
    # local offset and updates from the backend body pose.
    child_mesh_path = f"{body_path}/OffsetMesh"
    sim_utils.create_prim(child_mesh_path, "Xform", translation=(1.0, 0.0, 0.0))
    mesh_data = make_plane(size=(2, 2), height=0.0, center_zero=True)
    create_prim_from_mesh(f"{child_mesh_path}/Plane", mesh_data)
    # Add collision so PhysX tracks the body
    col_path = f"{body_path}/CollisionCube"
    cube_geom = cast(Any, UsdGeom.Cube.Define(stage, col_path))
    cube_geom.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(col_path))

    # Create a sensor prim to mount the MultiMeshRayCaster on
    sensor_path = "/World/SensorMount"
    sim_utils.create_prim(sensor_path, "Xform", translation=(0.0, 1.0, 5.0))

    # Configure MultiMeshRayCaster to track the child mesh
    cfg = MultiMeshRayCasterCfg(
        prim_path=sensor_path,
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr=child_mesh_path,
                track_mesh_transforms=True,
            ),
        ],
        update_period=0,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment="world",
    )
    sensor = MultiMeshRayCaster(cfg)
    sim.reset()
    sensor.update(_DT, force_recompute=True)

    # The runtime pose buffer stores the body pose; the child offset is in the mesh vertices.
    mesh_pos = wp.to_torch(sensor._mesh_positions_w).clone()
    np.testing.assert_allclose(
        mesh_pos[0, 0].cpu().numpy(),
        [0.0, 0.0, 2.0],
        atol=0.15,
        err_msg="Tracked mesh pose should be the owner body pose.",
    )
    np.testing.assert_allclose(
        sensor.data.ray_hits_w.torch[0, 0].cpu().numpy(),
        [0.0, 1.0, 2.0],
        atol=0.15,
        err_msg="Ray hit should include the child mesh offset baked into the Warp geometry.",
    )


# ---------------------------------------------------------------------------
# Articulation child links after state writes
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_ray_casters_read_link_pose_after_joint_write(sim_ground):
    """Ray casters report an articulation child link at its written pose before the next step.

    Environments reset inside ``step()`` and read observations before the next physics step, while the
    GPU pipeline defers the child-link pose update of articulation writes. ``Arm`` is a 2 m bar on a
    revolute joint about the Y axis, with the pivot fixed at (0, 0, 1).
    """
    sim = sim_ground
    robot = Articulation(
        ArticulationCfg(
            prim_path="/World/Robot",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd"
            ),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
            actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        )
    )
    # The standard and the multi-mesh ray casters read their mount pose through separate paths.
    ray_caster = RayCaster(_single_downward_ray_cfg("/World/Robot/Arm"))
    multi_mesh_ray_caster = MultiMeshRayCaster(
        MultiMeshRayCasterCfg(
            prim_path="/World/Robot/Arm",
            mesh_prim_paths=[_GROUND_PATH],
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
            ray_alignment="world",
        )
    )
    # A static scanner tracks the arm mesh with downward rays at x = -1 m and x = 1 m.
    sim_utils.create_prim("/World/Scanner", "Xform", translation=(0.0, 0.0, 3.0))
    scanner = MultiMeshRayCaster(
        MultiMeshRayCasterCfg(
            prim_path="/World/Scanner",
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/Robot/Arm", track_mesh_transforms=True)
            ],
            pattern_cfg=patterns.GridPatternCfg(resolution=2.0, size=(2.0, 0.0), direction=(0.0, 0.0, -1.0)),
            ray_alignment="world",
        )
    )
    sim.reset()
    arm = robot.find_bodies("Arm")[0][0]

    def step_and_write_joint_position(position: float) -> None:
        sim.step(render=False)
        robot.update(_DT)
        for sensor in (ray_caster, multi_mesh_ray_caster, scanner):
            sensor.update(_DT)
        robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, position))

    # Read one sensor first after each write: an asset body getter would update the kinematics itself.
    for sensor, position in ((ray_caster, 0.5), (multi_mesh_ray_caster, -0.5)):
        step_and_write_joint_position(position)
        pose_w = torch.cat([sensor.data.pos_w.torch, sensor.data.quat_w.torch], dim=-1)
        torch.testing.assert_close(pose_w, robot.data.body_link_pose_w.torch[:, arm])

    # Pointing the arm along -x puts its top face (z = 1.05 m) under the first ray only.
    step_and_write_joint_position(math.pi)
    hits = scanner.data.ray_hits_w.torch[0]
    torch.testing.assert_close(hits[0], torch.tensor([-1.0, 0.0, 1.05], device=hits.device), atol=1e-3, rtol=0.0)
    assert torch.isinf(hits[1]).all()
