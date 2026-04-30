# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Tests for RayCaster sensor behavior: alignment modes and reset."""

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
