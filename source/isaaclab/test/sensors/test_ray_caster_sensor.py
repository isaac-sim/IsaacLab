# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for RayCaster sensor behavior: alignment modes, reset, dynamic-parent tracking."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import numpy as np
import pytest
import torch
import warp as wp

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner import TemplateCloneCfg, clone_from_template, sequential
from isaaclab.sensors.ray_caster import (
    MultiMeshRayCaster,
    MultiMeshRayCasterCamera,
    MultiMeshRayCasterCameraCfg,
    MultiMeshRayCasterCfg,
    RayCaster,
    RayCasterCamera,
    RayCasterCameraCfg,
    RayCasterCfg,
    patterns,
)
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.math import quat_from_euler_xyz

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

_GROUND_PATH = "/World/Ground"


def _make_sim_and_ground():
    """Create a blank stage with a flat ground plane at z=0 and return the SimulationContext."""
    sim_utils.create_new_stage()
    dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh(_GROUND_PATH, mesh)
    sim_utils.update_stage()
    return sim


def _ray_caster_cfg(prim_path: str, alignment: str) -> RayCasterCfg:
    """Single downward ray, no offset from prim."""
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment=alignment,
    )


@pytest.fixture
def sim_ground():
    sim = _make_sim_and_ground()
    yield sim
    sim.stop()
    sim.clear_instance()


# -------------------------------------------------------------------
# Alignment mode tests
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_world_alignment_ignores_sensor_pitch(sim_ground):
    """In 'world' alignment, ray direction is always (0,0,-1) regardless of sensor pitch.

    Two sensors at the same location: one upright (identity), one pitched 30°.
    World-mode sensors must produce the same hit position (straight below at z=0).
    """
    sim = sim_ground

    # Upright sensor: identity orientation
    sim_utils.create_prim("/World/SensorUpright", "Xform", translation=(0.0, 0.0, 2.0))
    # Pitched 30° sensor — orientation=(x,y,z,w) per Isaac Lab convention
    pitch_quat = quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([np.pi / 6]), torch.tensor([0.0])
    )  # shape (1, 4), xyzw
    sim_utils.create_prim(
        "/World/SensorPitched",
        "Xform",
        translation=(0.0, 0.0, 2.0),
        orientation=tuple(pitch_quat[0].tolist()),  # xyzw
    )

    sensor_upright = RayCaster(_ray_caster_cfg("/World/SensorUpright", "world"))
    sensor_pitched = RayCaster(_ray_caster_cfg("/World/SensorPitched", "world"))
    sim.reset()

    dt = 0.01
    sensor_upright.update(dt)
    sensor_pitched.update(dt)

    # ray_hits_w returns a ProxyArray; use .torch for tensor indexing.
    hits_upright = sensor_upright.data.ray_hits_w.torch  # (1, 1, 3)
    hits_pitched = sensor_pitched.data.ray_hits_w.torch

    # Both must hit z=0 (straight down, world frame direction)
    assert abs(hits_upright[0, 0, 2].item()) < 0.02, (
        f"Upright world sensor must hit z≈0, got {hits_upright[0, 0, 2].item()}"
    )
    assert abs(hits_pitched[0, 0, 2].item()) < 0.02, (
        f"Pitched world sensor must hit z≈0, got {hits_pitched[0, 0, 2].item()}"
    )
    # Lateral positions must agree (same start at [0,0,2] + same direction [0,0,-1])
    torch.testing.assert_close(hits_upright, hits_pitched, atol=0.02, rtol=0)


@pytest.mark.isaacsim_ci
def test_base_alignment_rotates_ray_direction(sim_ground):
    """In 'base' alignment, ray direction follows the full sensor orientation.

    A sensor pitched +30° around Y (quat_from_euler_xyz(pitch=pi/6)):
    - Rotates (0,0,-1) to (-sin(30°), 0, -cos(30°)) = (-0.5, 0, -0.866)
    - world mode → ray still goes straight down, hits x≈0, z≈0
    - base mode  → ray tilts, hits at x ≈ -2*tan(30°) ≈ -1.155
    """
    sim = sim_ground

    pitch_quat = quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([np.pi / 6]), torch.tensor([0.0])
    )  # shape (1, 4), xyzw
    orientation = tuple(pitch_quat[0].tolist())

    sim_utils.create_prim("/World/SensorWorld", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorBase", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)

    sensor_world = RayCaster(_ray_caster_cfg("/World/SensorWorld", "world"))
    sensor_base = RayCaster(_ray_caster_cfg("/World/SensorBase", "base"))
    sim.reset()

    dt = 0.01
    sensor_world.update(dt)
    sensor_base.update(dt)

    hits_world = sensor_world.data.ray_hits_w.torch  # (1, 1, 3)
    hits_base = sensor_base.data.ray_hits_w.torch

    # World mode: ray still hits directly below (x≈0, y≈0, z≈0)
    assert abs(hits_world[0, 0, 0].item()) < 0.05, f"World mode hit x must be near 0, got {hits_world[0, 0, 0].item()}"
    assert abs(hits_world[0, 0, 2].item()) < 0.05, f"World mode must hit z≈0, got {hits_world[0, 0, 2].item()}"

    # Base mode: pitch +30° around Y rotates (0,0,-1) to (-0.5, 0, -0.866).
    # From height 2, the ray hits x = -2 * tan(30°) ≈ -1.155.
    expected_x = -2.0 * np.tan(np.pi / 6)
    assert abs(hits_base[0, 0, 0].item() - expected_x) < 0.05, (
        f"Base mode hit x should be ≈{expected_x:.3f}, got {hits_base[0, 0, 0].item():.3f}"
    )
    assert abs(hits_base[0, 0, 2].item()) < 0.05, f"Base mode must hit ground (z≈0), got {hits_base[0, 0, 2].item()}"


@pytest.mark.isaacsim_ci
def test_yaw_alignment_direction_unchanged(sim_ground):
    """In 'yaw' alignment, ray directions stay world-down despite pitch+roll.

    Setup: sensor at (0,0,2), pitched 30° and yawed 45°; pattern has a single ray
    at local offset (+1, 0, 0).

    - world mode: start = sensor_pos + (1,0,0) (no rotation applied to offset)
    - yaw  mode:  start = sensor_pos + yaw_rot(45°) @ (1,0,0) = (cos45°, sin45°, 0)

    Both modes fire the ray straight down (direction unchanged), so both hit z=0.
    The hit x-coordinate differs between modes, confirming the yaw-only rotation of
    start positions is applied in 'yaw' mode but not in 'world' mode.
    """
    sim = sim_ground

    combined_quat = quat_from_euler_xyz(
        torch.tensor([0.0]),
        torch.tensor([np.pi / 6]),  # 30° pitch
        torch.tensor([np.pi / 4]),  # 45° yaw
    )  # shape (1, 4), xyzw
    orientation = tuple(combined_quat[0].tolist())

    sim_utils.create_prim("/World/SensorWorldY", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorYaw", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)

    # Use a single ray at local offset (+1, 0, 0), still pointing down
    def _cfg_with_offset(prim_path, alignment):
        return RayCasterCfg(
            prim_path=prim_path,
            mesh_prim_paths=[_GROUND_PATH],
            update_period=0,
            offset=RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
            debug_vis=False,
            pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
            ray_alignment=alignment,
        )

    sensor_world = RayCaster(_cfg_with_offset("/World/SensorWorldY", "world"))
    sensor_yaw = RayCaster(_cfg_with_offset("/World/SensorYaw", "yaw"))
    sim.reset()

    dt = 0.01
    sensor_world.update(dt)
    sensor_yaw.update(dt)

    hits_world = sensor_world.data.ray_hits_w.torch  # (1, 1, 3)
    hits_yaw = sensor_yaw.data.ray_hits_w.torch

    # Both modes must hit the ground (direction unchanged = straight down in both modes)
    assert abs(hits_world[0, 0, 2].item()) < 0.05, "World mode must hit z≈0"
    assert abs(hits_yaw[0, 0, 2].item()) < 0.05, "Yaw mode must hit z≈0 (direction straight down)"

    # world mode: offset (1,0,0) not rotated → ray starts at sensor_pos+(1,0,0) → hits x≈1
    assert abs(hits_world[0, 0, 0].item() - 1.0) < 0.05, (
        f"World mode: hit x should be ≈1.0 (unrotated offset), got {hits_world[0, 0, 0].item():.3f}"
    )

    # yaw mode: offset (1,0,0) rotated by 45° yaw → starts at sensor_pos+(cos45°, sin45°, 0) → hits x≈cos45°
    expected_x_yaw = np.cos(np.pi / 4)  # ≈ 0.707
    assert abs(hits_yaw[0, 0, 0].item() - expected_x_yaw) < 0.05, (
        f"Yaw mode: hit x should be ≈{expected_x_yaw:.3f} (yaw-rotated offset), got {hits_yaw[0, 0, 0].item():.3f}"
    )
    # Confirm they differ — if they were the same, the test would not cover the yaw rotation
    assert not torch.allclose(hits_world, hits_yaw, atol=0.1), (
        "Yaw and world modes must produce different hit positions for non-zero lateral offset"
    )


# -------------------------------------------------------------------
# Reset / drift test
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_ray_caster_reset_resamples_drift(sim_ground):
    """reset() resamples drift values within the configured drift_range."""
    sim = sim_ground

    sim_utils.create_prim("/World/Sensor", "Xform", translation=(0.0, 0.0, 2.0))
    cfg = _ray_caster_cfg("/World/Sensor", "world")
    cfg.drift_range = (0.01, 0.05)  # force non-zero drift
    sensor = RayCaster(cfg)
    sim.reset()
    # sim.reset() initializes the sensor with zero drift; call sensor.reset() to resample
    # from the configured drift_range before we capture the baseline.
    sensor.reset()

    dt = 0.01
    sensor.update(dt)
    drift_before = sensor.drift.clone()  # (1, 3) torch tensor

    lo, hi = cfg.drift_range

    # After sensor.reset(), drift should be within the configured range
    assert drift_before.shape == (1, 3), f"Drift shape should be (1, 3), got {drift_before.shape}"
    assert (drift_before >= lo - 1e-6).all() and (drift_before <= hi + 1e-6).all(), (
        f"Initial drift must be in [{lo}, {hi}], got [{drift_before.min():.4f}, {drift_before.max():.4f}]"
    )

    # reset() resamples drift; values should remain within the configured range
    # Call reset() multiple times until we get a different sample (probability of same is near zero
    # for continuous uniform distribution, but we retry to avoid flakiness).
    for _ in range(5):
        sensor.reset()
        drift_after = sensor.drift.clone()
        if not torch.allclose(drift_after, drift_before):
            break
    assert drift_after.shape == drift_before.shape, "Drift shape must be preserved after reset"
    assert (drift_after >= lo - 1e-6).all() and (drift_after <= hi + 1e-6).all(), (
        f"Drift after reset must be in [{lo}, {hi}], got [{drift_after.min():.4f}, {drift_after.max():.4f}]"
    )
    assert not torch.allclose(drift_after, drift_before), (
        "reset() must resample drift; values must change from initial sample"
    )


# -------------------------------------------------------------------
# Dynamic-parent regression (PR #5476; FrameView/Fabric staleness from #5179)
# -------------------------------------------------------------------
# Sensor under a rigid body must follow the body through physics integration.
# Pre-fix: the sensor stayed at its spawn pose forever as the body fell.

_FALLING_INITIAL_Z = 5.0
_FALLING_PARENT_PATH = "/World/PhysicsParent"
_FALLING_SENSOR_PATH = f"{_FALLING_PARENT_PATH}/sensor"


def _build_falling_parent_with_sensor() -> None:
    """Spawn a RigidBody+ArticulationRoot parent at z=5 with a child Xform sensor mount."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(_FALLING_PARENT_PATH, "Xform", translation=(0.0, 0.0, _FALLING_INITIAL_Z), stage=stage)
    parent_prim = stage.GetPrimAtPath(_FALLING_PARENT_PATH)
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
    UsdPhysics.MassAPI.Apply(parent_prim).CreateMassAttr().Set(1.0)
    cube_path = f"{_FALLING_PARENT_PATH}/CollisionCube"
    UsdGeom.Cube.Define(stage, cube_path).CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    sim_utils.create_prim(_FALLING_SENSOR_PATH, "Xform", translation=(0.0, 0.0, 0.0), stage=stage)
    sim_utils.update_stage()


_GRID_KW = dict(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0))
_PINHOLE_KW = dict(width=4, height=4, focal_length=24.0)
_DYNAMIC_PARENT_VARIANTS = {
    "raycaster": (RayCaster, RayCasterCfg, patterns.GridPatternCfg(**_GRID_KW), False),
    "camera": (RayCasterCamera, RayCasterCameraCfg, patterns.PinholeCameraPatternCfg(**_PINHOLE_KW), True),
    "multi_mesh_raycaster": (
        MultiMeshRayCaster,
        MultiMeshRayCasterCfg,
        patterns.GridPatternCfg(**_GRID_KW),
        False,
    ),
    "multi_mesh_camera": (
        MultiMeshRayCasterCamera,
        MultiMeshRayCasterCameraCfg,
        patterns.PinholeCameraPatternCfg(**_PINHOLE_KW),
        True,
    ),
}


def _build_dynamic_parent_sensor(sensor_kind: str):
    """Construct the requested ray-caster / camera variant on the spawned sensor mount."""
    sensor_cls, cfg_cls, pattern, is_camera = _DYNAMIC_PARENT_VARIANTS[sensor_kind]
    cfg_kwargs = dict(
        prim_path=_FALLING_SENSOR_PATH,
        mesh_prim_paths=[_GROUND_PATH],
        update_period=0,
        spawn=None,  # use the Xform we already spawned; don't redirect through a body
        debug_vis=False,
        pattern_cfg=pattern,
    )
    cfg_kwargs.update({"data_types": ["distance_to_image_plane"]} if is_camera else {"ray_alignment": "base"})
    return sensor_cls(cfg_cls(**cfg_kwargs))


def _sensor_z(sensor) -> float:
    """Z-component of env-0's sensor world pose for either RayCaster or RayCasterCamera."""
    pos = sensor.data.pos_w
    return (pos.torch if hasattr(pos, "torch") else pos)[0, 2].item()


def _parent_body_z_ground_truth(sensor) -> float | None:
    """Parent body z from PhysX ``_body_view``, or ``None`` if not the PhysX rigid-body
    branch (Newton routes via ``SensorFrameTransform``; PhysX static branch has no view)."""
    if sensor.__backend_name__ != "physx" or sensor._body_view is None:
        return None
    return wp.to_torch(sensor._body_view.get_transforms()).view(-1, 7)[0, 2].item()


_DYNAMIC_PARENT_PARAMS = ["raycaster", "camera", "multi_mesh_raycaster", "multi_mesh_camera"]


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("sensor_kind", _DYNAMIC_PARENT_PARAMS)
def test_sensor_pose_tracks_falling_rigid_parent(sim_ground, sensor_kind):
    """Sensor world pose must follow a rigid-body parent's free fall.

    Two layered assertions: (1) loose change-detection (drop > 1 cm after ~150 ms)
    catches the pre-fix ``drop = 0 m exactly`` signature on any backend; (2) PhysX
    rigid-body branch additionally requires sensor z == parent-body z to ~µm
    (sensor Xform has identity offset relative to body in this test).
    """
    sim = sim_ground

    _build_falling_parent_with_sensor()
    sensor = _build_dynamic_parent_sensor(sensor_kind)
    sim.reset()

    dt, num_steps = 0.01, 15
    z_before = _sensor_z(sensor)
    for _ in range(num_steps):
        sim.step(render=False)
        sensor.update(dt)
    z_after = _sensor_z(sensor)
    drop = z_before - z_after

    assert drop > 0.01, (
        f"{sensor_kind} sensor pose did not change after {num_steps} sim steps under gravity."
        f" z_before={z_before:.6f} m, z_after={z_after:.6f} m, drop={drop:.6f} m."
        " The backend body tracker is returning a stale pose."
    )

    body_z = _parent_body_z_ground_truth(sensor)
    if body_z is not None:
        assert abs(z_after - body_z) < 1e-4, (
            f"{sensor_kind} sensor z={z_after:.6f} m does not match parent body z={body_z:.6f} m"
            f" (diff={abs(z_after - body_z):.6f} m). Body tracker offset composition is wrong."
        )


# Camera offset-composition regression: per-step compose must NOT bake into
# ``_offset_pos_wp`` / ``_offset_quat_wp`` — those are zero-copy torch aliases
# (``_offset_pos`` / ``_offset_quat``) carrying ``cfg.offset`` for ``reset()`` /
# ``set_world_poses()``. Mutating the warp side stomps the alias.


_CAMERA_OFFSET_PARENT_PATH = "/World/CameraOffsetParent"
_CAMERA_OFFSET_SENSOR_PATH = f"{_CAMERA_OFFSET_PARENT_PATH}/sensor"


def _build_static_rigid_parent_with_offset_sensor() -> None:
    """Spawn a kinematic rigid-body parent at z=5 with a sensor mount at non-identity USD-local."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(_CAMERA_OFFSET_PARENT_PATH, "Xform", translation=(0.0, 0.0, 5.0), stage=stage)
    parent_prim = stage.GetPrimAtPath(_CAMERA_OFFSET_PARENT_PATH)
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
    UsdPhysics.MassAPI.Apply(parent_prim).CreateMassAttr().Set(1.0)
    # Kinematic so gravity doesn't drop the body during the test window.
    UsdPhysics.RigidBodyAPI(parent_prim).CreateKinematicEnabledAttr().Set(True)
    cube_path = f"{_CAMERA_OFFSET_PARENT_PATH}/CollisionCube"
    UsdGeom.Cube.Define(stage, cube_path).CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    # Sensor xform at (0, 0.1, 0) under the body — non-identity body-to-Xform translation.
    sim_utils.create_prim(_CAMERA_OFFSET_SENSOR_PATH, "Xform", translation=(0.0, 0.1, 0.0), stage=stage)
    sim_utils.update_stage()


@pytest.mark.isaacsim_ci
def test_camera_offset_buffer_survives_body_tracker_init(sim_ground):
    """``_offset_pos`` must equal ``cfg.offset`` after init — backends compose
    body→Xform per step inside :meth:`_get_sensor_transforms_wp`, not by baking
    into the warp side (which would stomp the torch alias)."""
    sim = sim_ground
    _build_static_rigid_parent_with_offset_sensor()

    cfg_offset_pos = (0.5, 0.0, 0.0)
    sensor = RayCasterCamera(
        RayCasterCameraCfg(
            prim_path=_CAMERA_OFFSET_SENSOR_PATH,
            mesh_prim_paths=[_GROUND_PATH],
            update_period=0,
            spawn=None,
            debug_vis=False,
            offset=RayCasterCameraCfg.OffsetCfg(pos=cfg_offset_pos, rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
            pattern_cfg=patterns.PinholeCameraPatternCfg(width=4, height=4, focal_length=24.0),
            data_types=["distance_to_image_plane"],
        )
    )
    sim.reset()

    actual = sensor._offset_pos[0].cpu()
    expected = torch.tensor(cfg_offset_pos, dtype=actual.dtype)
    assert torch.allclose(actual, expected, atol=1e-5), (
        f"Camera _offset_pos was corrupted by the body-tracker init."
        f" expected cfg.offset={expected.tolist()}, got={actual.tolist()}."
    )


@pytest.mark.isaacsim_ci
def test_camera_set_world_poses_under_rigid_parent(sim_ground):
    """``set_world_poses`` must land the camera at the requested world pose under a
    rigid parent. With the warp-side bake bug, the alias write drops the body→Xform
    component and the camera lands at ``body_pose * T_xform_camera`` instead."""
    sim = sim_ground
    _build_static_rigid_parent_with_offset_sensor()

    sensor = RayCasterCamera(
        RayCasterCameraCfg(
            prim_path=_CAMERA_OFFSET_SENSOR_PATH,
            mesh_prim_paths=[_GROUND_PATH],
            update_period=0,
            spawn=None,
            debug_vis=False,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
            pattern_cfg=patterns.PinholeCameraPatternCfg(width=4, height=4, focal_length=24.0),
            data_types=["distance_to_image_plane"],
        )
    )
    sim.reset()
    target = torch.tensor([[1.0, 2.0, 6.0]], device=sensor._device)
    sensor.set_world_poses(positions=target, convention="world")
    sim.step(render=False)
    sensor.update(0.01)

    # CameraData.pos_w is a plain torch tensor (no ProxyArray wrapper).
    actual = sensor.data.pos_w[0]
    torch.testing.assert_close(actual, target[0], atol=1e-2, rtol=0)


# -------------------------------------------------------------------
# Tracked-target regression
# -------------------------------------------------------------------
# ``track_mesh_transforms=True`` targets must follow their rigid body through
# physics. Pre-fix: same FrameView/Fabric staleness as the sensor side.
# Fix: PhysX ``RigidObjectView`` + per-step compose; Newton ``cl_register_site``
# + ``SensorFrameTransform``.

_TARGET_PARENT_PATH = "/World/MovingTargetParent"
_TARGET_MESH_PATH = f"{_TARGET_PARENT_PATH}/Cube"


def _build_falling_target_cube() -> None:
    """Spawn a falling rigid-body cube whose mesh acts as the raycast target."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(_TARGET_PARENT_PATH, "Xform", translation=(0.0, 0.0, 1.5), stage=stage)
    parent_prim = stage.GetPrimAtPath(_TARGET_PARENT_PATH)
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    UsdPhysics.MassAPI.Apply(parent_prim).CreateMassAttr().Set(1.0)
    UsdGeom.Cube.Define(stage, _TARGET_MESH_PATH).CreateSizeAttr().Set(0.4)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(_TARGET_MESH_PATH))
    sim_utils.update_stage()


_TRACKED_TARGET_VARIANTS = {
    "multi_mesh_raycaster": (
        MultiMeshRayCaster,
        MultiMeshRayCasterCfg,
        patterns.GridPatternCfg(**_GRID_KW),
        False,
    ),
    "multi_mesh_camera": (
        MultiMeshRayCasterCamera,
        MultiMeshRayCasterCameraCfg,
        patterns.PinholeCameraPatternCfg(**_PINHOLE_KW),
        True,
    ),
}


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("sensor_kind", list(_TRACKED_TARGET_VARIANTS.keys()))
def test_multi_mesh_target_tracks_falling_rigid_body(sim_ground, sensor_kind):
    """Tracked target mesh poses must follow their parent rigid body through physics.

    Setup: stationary sensor at z=4 with a tracked target — a falling rigid-body
    cube starting at z=1.5; after 20 sim steps it has dropped under gravity, and
    ``_mesh_positions_w`` (which the raycast kernel reads) must reflect the new
    world pose. Parameterized over both multi-mesh variants. We probe
    ``_mesh_positions_w`` directly — ray geometry differs across raycaster vs
    camera, but both write this same buffer through the tracker.
    """
    sim = sim_ground
    sensor_cls, cfg_cls, pattern, is_camera = _TRACKED_TARGET_VARIANTS[sensor_kind]

    sim_utils.create_prim("/World/StaticSensor", "Xform", translation=(0.0, 0.0, 4.0))
    _build_falling_target_cube()

    cfg_kwargs = dict(
        prim_path="/World/StaticSensor",
        mesh_prim_paths=[
            cfg_cls.RaycastTargetCfg(prim_expr=_TARGET_PARENT_PATH, track_mesh_transforms=True),
        ],
        update_period=0,
        spawn=None,
        debug_vis=False,
        pattern_cfg=pattern,
    )
    cfg_kwargs.update({"data_types": ["distance_to_camera"]} if is_camera else {"ray_alignment": "world"})
    sensor = sensor_cls(cfg_cls(**cfg_kwargs))
    sim.reset()

    dt = 0.01
    _ = sensor.data  # trigger _update_buffers_impl → writes per-step poses into _mesh_positions_w
    target_z_before = wp.to_torch(sensor._mesh_positions_w)[0, 0, 2].item()

    for _ in range(20):
        sim.step(render=False)
        sensor.update(dt, force_recompute=True)
    target_z_after = wp.to_torch(sensor._mesh_positions_w)[0, 0, 2].item()
    drop = target_z_before - target_z_after

    assert drop > 0.05, (
        f"{sensor_kind}: tracked target world-z in _mesh_positions_w did not change after 20 sim"
        f" steps under gravity. before={target_z_before:.6f}, after={target_z_after:.6f},"
        f" drop={drop:.6f}. The target-mesh tracker is returning a stale pose."
    )

    # PhysX ground-truth cross-check: parent body's world z = raycast-kernel input.
    if sensor.__backend_name__ == "physx":
        body_z = wp.to_torch(sensor._target_view.get_transforms()).view(-1, 7)[0, 2].item()
        assert abs(target_z_after - body_z) < 1e-4, (
            f"_mesh_positions_w z={target_z_after:.6f} does not match parent body z={body_z:.6f}"
            f" (diff={abs(target_z_after - body_z):.6f})."
        )


# -------------------------------------------------------------------
# Heterogeneous-prototype regression
# -------------------------------------------------------------------
# Each env must see ITS OWN prototype's body→mesh offset and mesh geometry. Pre-fix
# env-0 walk silently used proto_0's offset for every env. Two axes verified:
#   * offset:  per-prototype body→mesh translation (target tracker's per-proto walk)
#   * geometry: per-prototype cube size (base ``_initialize_warp_meshes`` per-env mesh)
# Different sizes ⇒ different vertex buffers per env ⇒ raycast hit z = body_z +
# half_size catches any mesh-ID mis-routing on top of any offset mis-routing.
_NUM_HETERO_ENVS = 4
_PROTO_OFFSETS_X = [0.3, -0.4]
_PROTO_CUBE_SIZES = [0.2, 0.6]


def _build_heterogeneous_template_scene() -> dict:
    """Two prototypes (different body→mesh offset AND cube size) cloned across 4 envs
    sequentially → env_{0,2} use proto_0 and env_{1,3} use proto_1."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/template", "Xform", stage=stage)
    sim_utils.create_prim("/World/template/Object", "Xform", stage=stage)

    for proto_idx, (offset_x, cube_size) in enumerate(zip(_PROTO_OFFSETS_X, _PROTO_CUBE_SIZES)):
        # Body is the Object root prim; Cube is a child at the proto-specific offset.
        proto_path = f"/World/template/Object/proto_asset_{proto_idx}"
        sim_utils.create_prim(proto_path, "Xform", stage=stage, translation=(0.0, 0.0, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(proto_path))
        UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(proto_path)).CreateMassAttr().Set(1.0)
        cube_path = f"{proto_path}/Cube"
        sim_utils.create_prim(cube_path, "Xform", stage=stage, translation=(offset_x, 0.0, 0.0))
        UsdGeom.Cube.Define(stage, f"{cube_path}/Geom").CreateSizeAttr().Set(cube_size)
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"{cube_path}/Geom"))

    sim_utils.create_prim("/World/envs", "Xform", stage=stage)
    env_origins = [(2.0 * i, 0.0, 1.0) for i in range(_NUM_HETERO_ENVS)]
    for i, origin in enumerate(env_origins):
        sim_utils.create_prim(f"/World/envs/env_{i}", "Xform", stage=stage, translation=origin)
    sim_utils.update_stage()

    # Sequential strategy: env i → proto (i % num_protos). Publish plans manually
    # since this hand-authored scene bypasses InteractiveScene.
    cfg = TemplateCloneCfg(device="cpu", clone_strategy=sequential, clone_physics=False)
    plans = clone_from_template(stage, num_clones=_NUM_HETERO_ENVS, template_clone_cfg=cfg)
    sim_utils.SimulationContext.instance().set_clone_plans(plans)
    sim_utils.update_stage()
    return {"plans": plans, "env_origins": env_origins}


@pytest.mark.isaacsim_ci
def test_multi_mesh_target_heterogeneous_prototypes(sim_ground):
    """Each env must see its own prototype's body→mesh offset *and* mesh geometry.

    Setup: 4 envs cloned via sequential strategy from two prototypes whose Cube
    children differ in BOTH the body→mesh offset (handled by the target tracker)
    AND the cube size (handled by the base mesh-build). One sensor per env mounted
    above each cube fires a single ray straight down. Three checks:

    1. **Per-env tracker offset** (mesh-pose buffer x): each env's tracked Cube
       sits at ``env_origin[i].x + proto_offsets[i % 2]``. The pre-fix env-0
       walk gave every env proto_0's offset; this catches that.
    2. **Per-env mesh geometry** (raycast hit z): each env's hit-z is
       ``body_z + half_size[i % 2]``. Different cube sizes per prototype mean
       different vertex buffers per env — verifies the mesh-ID routing isn't
       silently treating envs as homogeneous.
    3. **Dynamic correctness**: after physics steps, hit-z drops per env while
       the per-env x offset and size signature both stay consistent.
    """
    sim = sim_ground
    scene = _build_heterogeneous_template_scene()
    plan = scene["plans"]["/World/envs/env_{}/Object"]
    env_origins = scene["env_origins"]

    # One sensor per env, mounted above its own cube column.
    for i, origin in enumerate(env_origins):
        sim_utils.create_prim(
            f"/World/envs/env_{i}/SensorMount",
            "Xform",
            translation=(_PROTO_OFFSETS_X[i % 2], 0.0, 5.0),
        )
    cfg = MultiMeshRayCasterCfg(
        prim_path="/World/envs/env_.*/SensorMount",
        mesh_prim_paths=[
            MultiMeshRayCasterCfg.RaycastTargetCfg(
                prim_expr="/World/envs/env_.*/Object/Cube",
                track_mesh_transforms=True,
            ),
        ],
        update_period=0,
        spawn=None,
        debug_vis=False,
        pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0)),
        ray_alignment="world",
    )
    sensor = MultiMeshRayCaster(cfg)
    sim.reset()

    # Sequential strategy: env i → proto (i % num_protos). Sanity-check the fixture.
    proto_idx_per_env = plan.clone_mask.to(torch.int).argmax(dim=0).cpu().tolist()
    assert proto_idx_per_env == [0, 1, 0, 1], (
        f"Sequential strategy expected [0, 1, 0, 1] across 4 envs, got {proto_idx_per_env}."
    )

    _ = sensor.data  # trigger _update_buffers_impl so buffers reflect step-0 poses
    mesh_pos_initial = wp.to_torch(sensor._mesh_positions_w)[:, 0].cpu()  # (num_envs, 3)
    hits_initial = sensor.data.ray_hits_w.torch[:, 0].cpu()  # (num_envs, 3)

    # 1. Per-env tracker offset: mesh world x = env_origin.x + this prototype's body→mesh dx.
    expected_x = torch.tensor([env_origins[i][0] + _PROTO_OFFSETS_X[i % 2] for i in range(_NUM_HETERO_ENVS)])
    torch.testing.assert_close(mesh_pos_initial[:, 0], expected_x, atol=1e-3, rtol=0)

    # 2. Per-env mesh geometry: ray hits cube top at ``body_z + half_size``.
    expected_half_size = torch.tensor([_PROTO_CUBE_SIZES[i % 2] / 2.0 for i in range(_NUM_HETERO_ENVS)])
    body_z_per_env = mesh_pos_initial[:, 2]  # mesh-z == body-z (cube child's z-offset is 0)
    expected_hit_z = body_z_per_env + expected_half_size
    torch.testing.assert_close(hits_initial[:, 2], expected_hit_z, atol=1e-3, rtol=0)

    # 3. Dynamic correctness under gravity: hit-z drops, per-env x stable, size signature holds.
    for _ in range(15):
        sim.step(render=False)
        sensor.update(0.01, force_recompute=True)
    mesh_pos_after = wp.to_torch(sensor._mesh_positions_w)[:, 0].cpu()
    hits_after = sensor.data.ray_hits_w.torch[:, 0].cpu()

    torch.testing.assert_close(mesh_pos_after[:, 0], expected_x, atol=1e-2, rtol=0)
    drop = hits_initial[:, 2] - hits_after[:, 2]
    assert (drop > 0.01).all(), f"All envs must show hit-z drop under gravity; drops={drop.tolist()}"
    expected_hit_z_after = mesh_pos_after[:, 2] + expected_half_size
    torch.testing.assert_close(hits_after[:, 2], expected_hit_z_after, atol=1e-3, rtol=0)
