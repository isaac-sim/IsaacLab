# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Integration tests for ray caster sensor view paths, env_mask, and intrinsics.

These tests require Isaac Sim (AppLauncher). They cover the integration-level
items from ``TODO_ray_caster_kernel_tests.md``:

- ``_get_view_transforms_wp`` ArticulationView and RigidBodyView paths
- ``MultiMeshRayCaster`` env_mask behavior
- ``MultiMeshRayCasterCamera.set_intrinsic_matrices`` propagation
- ``_update_mesh_transforms`` non-identity orientation offset (known bug, xfail)
- Depth clipping ordering for ``MultiMeshRayCasterCamera``
"""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import copy

import numpy as np
import pytest
import torch
import warp as wp

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
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

_GROUND_PATH = "/World/Ground"
_DT = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sim_and_ground():
    """Create a blank stage with a flat ground plane at z=0."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=_DT))
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh(_GROUND_PATH, mesh)
    sim_utils.update_stage()
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


@pytest.fixture
def sim_ground():
    sim = _make_sim_and_ground()
    yield sim
    sim.stop()
    sim.clear_instance()


# ---------------------------------------------------------------------------
# _get_view_transforms_wp: ArticulationView path
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_articulation_view_path(sim_ground):
    """Mount a ray caster on a prim with ArticulationRootAPI.

    Verifies that sensor pos_w matches the prim's initial position and that
    the downward ray hits the ground plane.  This exercises the
    ``ArticulationView.get_root_transforms()`` quaternion-convention path in
    :meth:`_get_view_transforms_wp`.
    """
    sim = sim_ground
    expected_pos = (3.0, 4.0, 5.0)

    prim_path = "/World/ArticulatedBody"
    sim_utils.create_prim(prim_path, "Xform", translation=expected_pos)
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    # Mass is needed for physics; collision is needed for PhysX to track the body.
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(1.0)
    # Create a small collision cube so PhysX treats this as a real body.
    cube_path = f"{prim_path}/CollisionCube"
    cube_geom = UsdGeom.Cube.Define(stage, cube_path)
    cube_geom.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    sim_utils.update_stage()

    sensor = RayCaster(_single_downward_ray_cfg(prim_path))
    sim.reset()
    sensor.update(_DT)

    pos_w = sensor.data.pos_w.torch[0].cpu().numpy()
    np.testing.assert_allclose(
        pos_w,
        expected_pos,
        atol=0.15,
        err_msg="ArticulationView: sensor pos_w must match initial prim position",
    )

    hits = sensor.data.ray_hits_w.torch[0, 0].cpu().numpy()
    assert abs(hits[2]) < 0.5, f"ArticulationView: downward ray should hit near z=0, got z={hits[2]}"


# ---------------------------------------------------------------------------
# _get_view_transforms_wp: RigidBodyView path
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_rigid_body_view_path(sim_ground):
    """Mount a ray caster on a prim with RigidBodyAPI (no ArticulationRootAPI).

    Exercises the ``RigidBodyView.get_transforms()`` path in
    :meth:`_get_view_transforms_wp`.
    """
    sim = sim_ground
    expected_pos = (1.0, 2.0, 6.0)

    prim_path = "/World/RigidBody"
    sim_utils.create_prim(prim_path, "Xform", translation=expected_pos)
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(1.0)
    cube_path = f"{prim_path}/CollisionCube"
    cube_geom = UsdGeom.Cube.Define(stage, cube_path)
    cube_geom.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    sim_utils.update_stage()

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
    output_before = camera.data.output["distance_to_camera"].clone()

    # Change to a very different intrinsic matrix (different FOV)
    new_matrix = torch.tensor(
        [[200.0, 0.0, 320.0], [0.0, 200.0, 240.0], [0.0, 0.0, 1.0]],
        device=camera.device,
    ).unsqueeze(0)
    camera.set_intrinsic_matrices(new_matrix, focal_length=1.0)

    for _ in range(3):
        sim.step()
        camera.update(_DT)
    output_after = camera.data.output["distance_to_camera"].clone()

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

    The ``distance_to_image_plane`` computation reads ``_ray_distance`` before
    ``distance_to_camera`` clips it in-place.  This test verifies the two data
    types do not interfere with each other.
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

    d2ip_joint = cam_joint.data.output["distance_to_image_plane"]
    d2c_joint = cam_joint.data.output["distance_to_camera"]
    d2ip_solo = cam_d2ip.data.output["distance_to_image_plane"]
    d2c_solo = cam_d2c.data.output["distance_to_camera"]

    # Joint camera must match solo cameras (clipping one must not corrupt the other)
    torch.testing.assert_close(d2ip_joint, d2ip_solo, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(d2c_joint, d2c_solo, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# MultiMeshRayCaster env_mask behavior
# ---------------------------------------------------------------------------


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
    """Tracked mesh position must account for body orientation when applying offset.

    Setup: a kinematic rigid body at (0, 0, 2) rotated 90 deg around Z, with a
    child mesh offset by (1, 0, 0) in the body's local frame.

    Correct world position of mesh = body_pos + rotate(body_ori, local_offset)
        = (0, 0, 2) + rotate(90degZ, (1, 0, 0))
        = (0, 0, 2) + (0, 1, 0)
        = (0, 1, 2)

    Naive subtraction (the old bug) would give: body_pos - offset = (-1, 0, 2).
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
    body_prim = stage.GetPrimAtPath(body_path)
    UsdPhysics.RigidBodyAPI.Apply(body_prim)
    mass_api = UsdPhysics.MassAPI.Apply(body_prim)
    mass_api.CreateMassAttr().Set(1.0)
    body_prim.GetAttribute("physics:kinematicEnabled").Set(True)

    # Create a child Xform offset by (1, 0, 0) in the body's local frame,
    # then place mesh geometry under it. The Xform translation is the offset
    # that _obtain_trackable_prim_view / resolve_prim_pose will discover.
    child_mesh_path = f"{body_path}/OffsetMesh"
    sim_utils.create_prim(child_mesh_path, "Xform", translation=(1.0, 0.0, 0.0))
    mesh_data = make_plane(size=(2, 2), height=0.0, center_zero=True)
    create_prim_from_mesh(f"{child_mesh_path}/Plane", mesh_data)
    # Add collision so PhysX tracks the body
    col_path = f"{body_path}/CollisionCube"
    cube_geom = UsdGeom.Cube.Define(stage, col_path)
    cube_geom.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(col_path))
    sim_utils.update_stage()

    # Create a sensor prim to mount the MultiMeshRayCaster on
    sensor_path = "/World/SensorMount"
    sim_utils.create_prim(sensor_path, "Xform", translation=(0.0, 0.0, 5.0))

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
    sensor.update(_DT)

    # Verify mesh position: body at (0,0,2) rotated 90deg Z, child offset (1,0,0) local
    # Expected: (0, 0, 2) + rotate(90degZ, (1,0,0)) = (0, 0, 2) + (0, 1, 0) = (0, 1, 2)
    mesh_pos = sensor._mesh_positions_w_torch.clone()
    np.testing.assert_allclose(
        mesh_pos[0, 0].cpu().numpy(),
        [0.0, 1.0, 2.0],
        atol=0.15,
        err_msg=(
            "Mesh position should be (0, 1, 2) via proper frame decomposition: "
            "body_pos + rotate(body_ori, local_offset). "
            "If this fails, the offset is not being rotated by the body orientation."
        ),
    )
