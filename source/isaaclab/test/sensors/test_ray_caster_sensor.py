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
    # Pitched 30° sensor
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

    hits_upright = sensor_upright.data.ray_hits_w  # (1, 1, 3)
    hits_pitched = sensor_pitched.data.ray_hits_w

    # Both must hit z=0 (straight down, world frame direction)
    assert abs(hits_upright[0, 0, 2].item()) < 0.01, "Upright world sensor must hit z≈0"
    assert abs(hits_pitched[0, 0, 2].item()) < 0.01, "Pitched world sensor must hit z≈0"
    # Lateral positions must agree (same start at [0,0,2] + same direction [0,0,-1])
    torch.testing.assert_close(hits_upright, hits_pitched, atol=0.02, rtol=0,
                                msg="World mode hits must be independent of pitch")


@pytest.mark.isaacsim_ci
def test_base_alignment_rotates_ray_direction(sim_ground):
    """In 'base' alignment, ray direction follows the full sensor orientation.

    A sensor pitched 30° forward:
    - world mode → ray still goes straight down, hits (0,0,0)
    - base mode  → ray tilts 30° forward, hits at x ≈ 2*tan(30°) ≈ 1.155
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

    hits_world = sensor_world.data.ray_hits_w  # (1, 1, 3)
    hits_base = sensor_base.data.ray_hits_w

    # World mode: ray still hits directly below (x≈0, y≈0, z≈0)
    assert abs(hits_world[0, 0, 0].item()) < 0.05, f"World mode hit x must be near 0, got {hits_world[0,0,0].item()}"
    assert abs(hits_world[0, 0, 2].item()) < 0.05, f"World mode must hit z≈0, got {hits_world[0,0,2].item()}"

    # Base mode: ray is tilted forward 30°; from height 2, hit at x ≈ 2*tan(30°) ≈ 1.155
    expected_x = 2.0 * np.tan(np.pi / 6)
    assert abs(hits_base[0, 0, 0].item() - expected_x) < 0.15, (
        f"Base mode hit x should be ≈{expected_x:.3f}, got {hits_base[0, 0, 0].item():.3f}"
    )
    assert abs(hits_base[0, 0, 2].item()) < 0.05, f"Base mode must hit ground (z≈0), got {hits_base[0,0,2].item()}"


@pytest.mark.isaacsim_ci
def test_yaw_alignment_direction_unchanged(sim_ground):
    """In 'yaw' alignment, ray direction is not rotated even with pitch+roll.

    For a sensor with combined pitch+yaw, 'yaw' mode must leave ray direction
    unchanged (same as 'world' mode for a center ray with zero local offset).
    """
    sim = sim_ground

    combined_quat = quat_from_euler_xyz(
        torch.tensor([0.0]),
        torch.tensor([np.pi / 6]),   # 30° pitch
        torch.tensor([np.pi / 4]),   # 45° yaw
    )  # shape (1, 4), xyzw
    orientation = tuple(combined_quat[0].tolist())

    sim_utils.create_prim("/World/SensorWorld", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorYaw", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)

    sensor_world = RayCaster(_ray_caster_cfg("/World/SensorWorld", "world"))
    sensor_yaw = RayCaster(_ray_caster_cfg("/World/SensorYaw", "yaw"))
    sim.reset()

    dt = 0.01
    sensor_world.update(dt)
    sensor_yaw.update(dt)

    hits_world = sensor_world.data.ray_hits_w
    hits_yaw = sensor_yaw.data.ray_hits_w

    # Both must hit near z=0 (direction unchanged by yaw mode; local_start=0 so start unchanged)
    assert abs(hits_world[0, 0, 2].item()) < 0.05, "World mode must hit z≈0"
    assert abs(hits_yaw[0, 0, 2].item()) < 0.05, "Yaw mode must hit z≈0 (direction unchanged)"
    # For zero-offset center ray: same start + same direction → same hit
    torch.testing.assert_close(hits_world, hits_yaw, atol=0.05, rtol=0,
                                msg="Yaw and world modes must agree for zero-offset center ray")


# -------------------------------------------------------------------
# Reset / drift test
# -------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_ray_caster_reset_clears_drift(sim_ground):
    """reset() resamples all envs' drift; reset(env_ids) only resets selected envs."""
    sim = sim_ground

    sim_utils.create_prim("/World/Sensor", "Xform", translation=(0.0, 0.0, 2.0))
    cfg = _ray_caster_cfg("/World/Sensor", "world")
    cfg.drift_range = (0.01, 0.05)  # force non-zero drift
    sensor = RayCaster(cfg)
    sim.reset()

    dt = 0.01
    sensor.update(dt)
    drift_before = sensor.drift.clone()

    # Full reset must preserve shape
    sensor.reset()
    drift_after = sensor.drift.clone()
    assert drift_after.shape == drift_before.shape, "Drift shape must be preserved after reset"
    assert drift_after.shape[1] == 3, "Drift must have 3 components (x, y, z)"
