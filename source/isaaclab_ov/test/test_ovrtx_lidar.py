# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lidar pose composition and return-layout contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_ov.renderers import OVRTXLidarCfg
from isaaclab_ov.renderers.ovrtx_lidar_sensor import OVRTXLidar

from isaaclab.utils.warp import ProxyArray


def _lidar_cfg(**overrides) -> OVRTXLidarCfg:
    """Build a one-ray lidar config. ``overrides`` replace the valid defaults.

    ``max_returns`` is 1 so a slot-count case measures the firing pattern. The schema default is 2.
    """
    values = {
        "prim_path": "/World/envs/env_.*/Lidar",
        "azimuth_deg": [0.0],
        "elevation_deg": [0.0],
        "fire_time_ns": [0],
        "channel_id": [0],
        "max_returns": 1,
    }
    values.update(overrides)
    return OVRTXLidarCfg(**values)


def test_lidar_cfg_rejects_a_non_positive_azimuth_step():
    """A rotary step of zero cannot divide the azimuth window."""
    with pytest.raises(ValueError, match="rot_pattern_firing_res_deg"):
        _lidar_cfg(rot_pattern_firing_res_deg=0.0)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"scan_type": "solidState"}, 1),
        (
            {
                "scan_type": "solidState",
                "azimuth_deg": [0.0, 1.0],
                "elevation_deg": [0.0, 0.0],
                "fire_time_ns": [0, 0],
                "channel_id": [0, 1],
                "max_returns": 2,
            },
            4,
        ),
        (
            {
                "scan_type": "solidState",
                "azimuth_deg": [0.0, 1.0, 2.0],
                "elevation_deg": [0.0, 0.0, 0.0],
                "fire_time_ns": [0, 0, 0],
                "channel_id": [0, 1, 2],
                "tiled_subsampling": 2,
            },
            1,
        ),
        ({"scan_type": "rotary", "rot_pattern_firing_res_deg": 90.0}, 4),
        (
            {
                "scan_type": "rotary",
                "rot_pattern_firing_res_deg": 10.0,
                "valid_start_azimuth_deg": 350.0,
                "valid_end_azimuth_deg": 10.0,
            },
            2,
        ),
    ],
)
def test_lidar_scan_slot_count(overrides, expected):
    """A solid-state scan keeps one slot per pattern ray, and a rotary scan repeats that pattern."""
    assert _lidar_cfg(**overrides).num_rays() == expected


def test_lidar_world_pose_composes_a_rotated_parent():
    """A parent yaw moves the offset off the world X axis instead of adding it to the origin."""
    lidar = OVRTXLidar.__new__(OVRTXLidar)
    lidar._initialize_handle = None
    lidar._invalidate_initialize_handle = None
    lidar._prim_deletion_handle = None
    lidar._debug_vis_handle = None
    lidar._renderer = None
    lidar._render_data = None
    lidar._num_envs = 1
    lidar._local_pos = torch.tensor([1.0, 0.0, 0.0])
    lidar._local_quat = torch.tensor([0.0, 0.0, 0.0, 1.0])
    lidar._pos_w = ProxyArray(wp.zeros(1, dtype=wp.vec3f, device="cpu"))
    lidar._quat_w = ProxyArray(wp.zeros(1, dtype=wp.quatf, device="cpu"))
    half = math.sqrt(0.5)
    parent_pos = torch.tensor([[2.0, 3.0, 4.0]])
    parent_quat = torch.tensor([[0.0, 0.0, half, half]])
    lidar._parent_view = SimpleNamespace(
        get_world_poses=lambda: (SimpleNamespace(torch=parent_pos), SimpleNamespace(torch=parent_quat)),
        close=lambda: None,
    )

    lidar._write_world_poses()

    torch.testing.assert_close(lidar._pos_w.torch, torch.tensor([[2.0, 4.0, 4.0]]))
    torch.testing.assert_close(lidar._quat_w.torch, parent_quat)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.ov
@pytest.mark.skipif(not wp.is_cuda_available(), reason="Tiled lidar capture requires CUDA.")
def test_tiled_lidar_returns_are_local_to_each_environment():
    """Identical targets give sensor-local hits, and resetting one tile preserves the other."""
    from isaaclab_ov.physics import OvPhysxCfg

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass, instantiate

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        target = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Target",
            spawn=sim_utils.CuboidCfg(
                size=(4.0, 4.0, 1.0),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
            ),
        )
        lidar = _lidar_cfg(
            prim_path="{ENV_REGEX_NS}/Lidar",
            scan_type="solidState",
            azimuth_deg=[0.0, 1.0],
            elevation_deg=[0.0, 0.0],
            fire_time_ns=[0, 0],
            channel_id=[0, 1],
            offset=OVRTXLidarCfg.OffsetCfg(pos=(0.0, 0.0, 3.0)),
        )

    with sim_utils.build_simulation_context(sim_cfg=sim_utils.SimulationCfg(physics=OvPhysxCfg())) as sim:
        scene = instantiate(SceneCfg(num_envs=2, env_spacing=8.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        sim.step()
        scene.update(sim.get_physics_dt())
        lidar = scene["lidar"]
        data = lidar.data
        points, valid = data.points.torch, data.valid.torch
        assert points.shape == (2, 2, 3)
        assert valid.all()
        assert torch.isfinite(points).all()
        torch.testing.assert_close(points[0], points[1], atol=1e-3, rtol=0.0)

        kept = points[1].clone()
        lidar.reset(env_ids=[0])
        assert not lidar.data.valid.torch[0].any()
        assert not torch.isfinite(lidar.data.points.torch[0]).any()
        assert lidar.data.valid.torch[1].all()
        torch.testing.assert_close(lidar.data.points.torch[1], kept)
