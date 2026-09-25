# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for RayCaster sensor behavior: alignment modes, offsets, reset, and parent tracking."""

from typing import Literal

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.math import quat_from_euler_xyz

pytestmark = pytest.mark.integration

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


def _ray_caster_cfg(prim_path: str, alignment: Literal["base", "yaw", "world"]) -> RayCasterCfg:
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
def test_ray_alignment_modes(sim_ground):
    """Each alignment mode rotates ray starts and directions by its share of the sensor orientation.

    Sensors at height 2 above the ground plane z=0:

    - Pitched +30° around Y (quat_from_euler_xyz(pitch=pi/6)), single ray at the sensor origin:
      world mode ignores the pitch and hits straight below, matching an upright world sensor; base mode
      rotates (0,0,-1) to (-0.5, 0, -0.866) and hits at x ≈ -2*tan(30°) ≈ -1.155.
    - Pitched 30° and yawed 45°, single ray at local offset (+1, 0, 0): world mode does not rotate the
      offset (hit x ≈ 1); yaw mode rotates it by the yaw only (hit x ≈ cos45°). Both fire straight down.
    """
    sim = sim_ground

    pitch_quat = quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([np.pi / 6]), torch.tensor([0.0])
    )  # shape (1, 4), xyzw
    orientation = tuple(pitch_quat[0].tolist())
    combined_quat = quat_from_euler_xyz(
        torch.tensor([0.0]),
        torch.tensor([np.pi / 6]),  # 30° pitch
        torch.tensor([np.pi / 4]),  # 45° yaw
    )  # shape (1, 4), xyzw
    combined_orientation = tuple(combined_quat[0].tolist())

    sim_utils.create_prim("/World/SensorUpright", "Xform", translation=(0.0, 0.0, 2.0))
    sim_utils.create_prim("/World/SensorWorld", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorBase", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorWorldY", "Xform", translation=(0.0, 0.0, 2.0), orientation=combined_orientation)
    sim_utils.create_prim("/World/SensorYaw", "Xform", translation=(0.0, 0.0, 2.0), orientation=combined_orientation)

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

    sensor_upright = RayCaster(_ray_caster_cfg("/World/SensorUpright", "world"))
    sensor_world = RayCaster(_ray_caster_cfg("/World/SensorWorld", "world"))
    sensor_base = RayCaster(_ray_caster_cfg("/World/SensorBase", "base"))
    sensor_world_offset = RayCaster(_cfg_with_offset("/World/SensorWorldY", "world"))
    sensor_yaw_offset = RayCaster(_cfg_with_offset("/World/SensorYaw", "yaw"))
    sim.reset()

    dt = 0.01
    for sensor in (sensor_upright, sensor_world, sensor_base, sensor_world_offset, sensor_yaw_offset):
        sensor.update(dt)

    # ray_hits_w returns a ProxyArray; use .torch for tensor indexing.
    hits_upright = sensor_upright.data.ray_hits_w.torch  # (1, 1, 3)
    hits_world = sensor_world.data.ray_hits_w.torch
    hits_base = sensor_base.data.ray_hits_w.torch

    # World mode: ray still hits directly below (x≈0, y≈0, z≈0), same as an upright sensor
    assert abs(hits_upright[0, 0, 2].item()) < 0.02, (
        f"Upright world sensor must hit z≈0, got {hits_upright[0, 0, 2].item()}"
    )
    assert abs(hits_world[0, 0, 0].item()) < 0.05, f"World mode hit x must be near 0, got {hits_world[0, 0, 0].item()}"
    assert abs(hits_world[0, 0, 2].item()) < 0.02, f"World mode must hit z≈0, got {hits_world[0, 0, 2].item()}"
    # Lateral positions must agree (same start at [0,0,2] + same direction [0,0,-1])
    torch.testing.assert_close(hits_upright, hits_world, atol=0.02, rtol=0)

    # Base mode: pitch +30° around Y rotates (0,0,-1) to (-0.5, 0, -0.866).
    # From height 2, the ray hits x = -2 * tan(30°) ≈ -1.155.
    expected_x = -2.0 * np.tan(np.pi / 6)
    assert abs(hits_base[0, 0, 0].item() - expected_x) < 0.05, (
        f"Base mode hit x should be ≈{expected_x:.3f}, got {hits_base[0, 0, 0].item():.3f}"
    )
    assert abs(hits_base[0, 0, 2].item()) < 0.05, f"Base mode must hit ground (z≈0), got {hits_base[0, 0, 2].item()}"

    hits_world_offset = sensor_world_offset.data.ray_hits_w.torch
    hits_yaw_offset = sensor_yaw_offset.data.ray_hits_w.torch

    # Both modes must hit the ground (direction unchanged = straight down in both modes)
    assert abs(hits_world_offset[0, 0, 2].item()) < 0.05, "World mode must hit z≈0"
    assert abs(hits_yaw_offset[0, 0, 2].item()) < 0.05, "Yaw mode must hit z≈0 (direction straight down)"

    # world mode: offset (1,0,0) not rotated → ray starts at sensor_pos+(1,0,0) → hits x≈1
    assert abs(hits_world_offset[0, 0, 0].item() - 1.0) < 0.05, (
        f"World mode: hit x should be ≈1.0 (unrotated offset), got {hits_world_offset[0, 0, 0].item():.3f}"
    )

    # yaw mode: offset (1,0,0) rotated by 45° yaw → starts at sensor_pos+(cos45°, sin45°, 0) → hits x≈cos45°
    expected_x_yaw = np.cos(np.pi / 4)  # ≈ 0.707
    assert abs(hits_yaw_offset[0, 0, 0].item() - expected_x_yaw) < 0.05, (
        f"Yaw mode: hit x should be ≈{expected_x_yaw:.3f} (yaw-rotated offset),"
        f" got {hits_yaw_offset[0, 0, 0].item():.3f}"
    )
    # Confirm they differ — if they were the same, the test would not cover the yaw rotation
    assert not torch.allclose(hits_world_offset, hits_yaw_offset, atol=0.1), (
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
    drift_after: torch.Tensor = drift_before.clone()
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


@pytest.mark.isaacsim_ci
def test_offset_does_not_affect_pos_w(sim_ground):
    """Verify that cfg.offset.pos shifts ray starts but NOT data.pos_w.

    data.pos_w must reflect the parent body position so that downstream
    observations like height_scan (pos_w_z - hit_z - 0.5) produce values
    relative to the body, not relative to the offset sensor frame.

    Regression test: previously the offset was baked into the FrameView's
    Xform local transform, causing data.pos_w to include the 20m offset
    and breaking height-scan observations during training.
    """
    sim = sim_ground

    # parent body at known position
    body_pos = (0.0, 0.0, 0.6)
    sim_utils.create_prim("/World/Robot", "Xform", translation=body_pos)

    # large z-offset to make the regression obvious
    offset_z = 20.0
    cfg = RayCasterCfg(
        prim_path="/World/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, offset_z)),
        mesh_prim_paths=[_GROUND_PATH],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=[1.0, 1.0]),
        ray_alignment="yaw",
    )

    dt = 0.01
    sensor = RayCaster(cfg)
    sim.reset()
    sensor.update(dt)

    # data.pos_w / data.ray_hits_w return ProxyArray wrappers; use .torch for tensor indexing.
    pos_w = sensor.data.pos_w.torch[0].cpu()

    # pos_w.z should be near the body height, NOT body_height + offset
    assert abs(pos_w[2].item() - body_pos[2]) < 1.0, (
        f"data.pos_w.z = {pos_w[2].item():.2f}, expected near body height {body_pos[2]}."
        f" If pos_w.z ≈ {body_pos[2] + offset_z}, the offset was incorrectly baked into the FrameView."
    )

    # ray_hits should be near z=0 (ground plane)
    hits_z = sensor.data.ray_hits_w.torch[0, :, 2].cpu()
    valid = hits_z[~torch.isinf(hits_z)]
    assert valid.numel() > 0, "Expected the offset rays to hit the ground plane"
    assert valid.abs().max().item() < 2.0, (
        f"Ray hits z range [{valid.min().item():.2f}, {valid.max().item():.2f}] — expected near ground (z≈0)."
    )

    # height_scan observation: pos_w_z - hit_z - 0.5 should be small, not ~20
    height_obs = pos_w[2].item() - valid.mean().item() - 0.5
    assert abs(height_obs) < 5.0, (
        f"height_scan observation = {height_obs:.2f}, expected near 0."
        f" If ≈{offset_z}, the offset leaked into data.pos_w."
    )


@pytest.mark.isaacsim_ci
def test_ray_caster_tracks_physics_body_parent_motion(sim_ground):
    """RayCaster pose must follow its physics-body parent after simulation steps."""
    from pxr import UsdGeom, UsdPhysics  # noqa: PLC0415

    sim = sim_ground
    dt = 0.01
    parent_path = "/World/PhysicsParent"

    expected_pos = (3.0, 4.0, 5.0)

    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(parent_path, "Xform", translation=expected_pos, stage=stage)
    parent_prim = stage.GetPrimAtPath(parent_path)
    UsdPhysics.RigidBodyAPI.Apply(parent_prim)
    UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
    mass_api = UsdPhysics.MassAPI.Apply(parent_prim)
    if mass_api is None:
        raise RuntimeError(f"Failed to apply MassAPI to {parent_path}.")
    mass_api.CreateMassAttr().Set(1.0)

    cube_path = f"{parent_path}/CollisionCube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    if cube is None:
        raise RuntimeError(f"Failed to create collision cube at {cube_path}.")
    cube.CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(cube_path))
    sim_utils.update_stage()

    sensor = RayCaster(_ray_caster_cfg(parent_path, "world"))
    sim.reset()
    sensor.update(dt, force_recompute=True)
    pos_before = sensor.data.pos_w.torch[0].clone()
    np.testing.assert_allclose(
        pos_before.cpu().numpy(), expected_pos, atol=0.15, err_msg="sensor pos_w must match initial parent position"
    )
    hit_z = sensor.data.ray_hits_w.torch[0, 0, 2].item()
    assert abs(hit_z) < 0.5, f"downward ray should hit near z=0, got z={hit_z}"

    for _ in range(100):
        sim.step(render=False)
        sensor.update(dt)

    sensor.update(dt, force_recompute=True)
    pos_after = sensor.data.pos_w.torch[0]
    drift_z = (pos_before[2] - pos_after[2]).item()

    assert drift_z > 0.5, (
        f"RayCaster pose did not follow its physics body parent. "
        f"z before={pos_before[2].item():.4f} z after={pos_after[2].item():.4f} drift={drift_z:.4f}m."
    )
