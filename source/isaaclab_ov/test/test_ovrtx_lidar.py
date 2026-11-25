# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Lidar pose composition, return-layout, and reset contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_ov.renderers import OVRTXLidarCfg
from isaaclab_ov.renderers.ovrtx_lidar_sensor import OVRTXLidar

from isaaclab.utils import validate
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


def test_lidar_cfg_rejects_dropped_misses():
    """Compacted returns cannot be placed in their original ray slots."""
    with pytest.raises(ValueError, match="include_invalid_points=True"):
        validate(_lidar_cfg(include_invalid_points=False))


def test_lidar_cfg_rejects_a_non_positive_azimuth_step():
    """A rotary step of zero cannot divide the azimuth window."""
    with pytest.raises(ValueError, match="rot_pattern_firing_res_deg"):
        _lidar_cfg(rot_pattern_firing_res_deg=0.0)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
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


def _lidar_without_runtime(*, num_envs: int, is_global: bool = False) -> OVRTXLidar:
    """Build a lidar without spawning a prim or renderer."""
    lidar = OVRTXLidar.__new__(OVRTXLidar)
    lidar._initialize_handle = None
    lidar._invalidate_initialize_handle = None
    lidar._prim_deletion_handle = None
    lidar._debug_vis_handle = None
    lidar._renderer = None
    lidar._render_data = None
    lidar._parent_view = None
    lidar._is_global = is_global
    lidar._num_envs = num_envs
    lidar._device = "cpu"
    lidar._data_dirty = False
    lidar._ALL_ENV_MASK = wp.ones(num_envs, dtype=wp.bool, device="cpu")
    lidar._reset_mask = wp.zeros(num_envs, dtype=wp.bool, device="cpu")
    lidar._create_timing_buffers()
    return lidar


def test_lidar_world_pose_composes_a_rotated_parent():
    """A parent yaw moves the offset off the world X axis instead of adding it to the origin."""
    lidar = _lidar_without_runtime(num_envs=1)
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


def test_cpu_mapped_spherical_cloud_preserves_returns():
    """Decode a CPU fallback map containing an oblique hit and a missed slot."""
    from isaaclab_ov.renderers.ovrtx_lidar import copy_tiled_point_cloud

    from isaaclab.sensors import LidarData

    class MappedCloud(dict):
        params = {"coordsType": np.array([1]), "tileStridePoints": np.array([2])}

    cloud = MappedCloud(
        Coordinates=np.array([[90.0, -30.0, 2.0], [10.0, 20.0, 3.0]], dtype=np.float32),
        Flags=np.array([0x40, 0], dtype=np.int32),
        Intensity=np.array([0.75, 1.0], dtype=np.float32),
    )
    data = LidarData(1, 2, "cpu")
    copy_tiled_point_cloud(cloud, data.points.torch, data.intensity.torch, data.valid.torch)
    torch.testing.assert_close(data.points.torch[0, 0], torch.tensor([0.0, math.sqrt(3.0), -1.0]))
    assert torch.isinf(data.points.torch[0, 1]).all()
    torch.testing.assert_close(data.valid.torch, torch.tensor([[True, False]]))
    torch.testing.assert_close(data.intensity.torch, torch.tensor([[0.75, 0.0]]))


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
    from isaaclab.utils import configclass, instantiate, replace

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
            azimuth_deg=[-8.0, 12.0],
            elevation_deg=[-5.0, 10.0],
            fire_time_ns=[0, 0],
            channel_id=[0, 1],
            offset=OVRTXLidarCfg.OffsetCfg(pos=(0.2, 0.1, 3.0), rot=(0.0, 0.0, 0.5, math.sqrt(0.75))),
        )

        spherical = replace(lidar, prim_path="{ENV_REGEX_NS}/Spherical", elements_coords_type="SPHERICAL")
        world = replace(lidar, prim_path="{ENV_REGEX_NS}/WorldLidar", output_frame_of_reference="WORLD")
        world_spherical = replace(world, prim_path="{ENV_REGEX_NS}/WorldSpherical", elements_coords_type="SPHERICAL")

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

        for native, converted in ((lidar, scene["spherical"]), (scene["world"], scene["world_spherical"])):
            expected, actual = native.data, converted.data
            assert expected.valid.torch.all()
            torch.testing.assert_close(actual.valid.torch, expected.valid.torch)
            torch.testing.assert_close(actual.points.torch, expected.points.torch, atol=1e-4, rtol=1e-5)

        kept = points[1].clone()
        lidar.reset(env_ids=[0])
        assert not lidar.data.valid.torch[0].any()
        assert not torch.isfinite(lidar.data.points.torch[0]).any()
        assert lidar.data.valid.torch[1].all()
        torch.testing.assert_close(lidar.data.points.torch[1], kept)


@pytest.mark.integration
@pytest.mark.rendering
@pytest.mark.ov
@pytest.mark.skipif(not wp.is_cuda_available(), reason="Tiled lidar capture requires CUDA.")
def test_referenced_lidar_follows_authored_mount():
    """A referenced lidar scans from its translated and tilted mount as the parent moves."""
    from isaaclab_ov.physics import OvPhysxCfg
    from isaaclab_ov.renderers.ovrtx_lidar_sensor import spawn_lidar

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils import configclass, instantiate

    def spawn_platform(path, cfg, **kwargs):
        platform = sim_utils.spawn_cuboid(path, cfg, **kwargs)
        mount = OVRTXLidarCfg.OffsetCfg(
            pos=(0.3, 0.0, 3.0), rot=(0.0, math.sin(math.pi / 12), 0.0, math.cos(math.pi / 12))
        )
        spawn_lidar(
            str(platform.GetPath()) + "/Lidar",
            _lidar_cfg(scan_type="solidState", offset=mount, output_frame_of_reference="WORLD"),
        )
        return platform

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        platform = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Platform",
            spawn=sim_utils.CuboidCfg(
                func=spawn_platform,
                size=(4.0, 4.0, 1.0),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
            ),
        )
        lidar = _lidar_cfg(
            prim_path="{ENV_REGEX_NS}/Platform/Lidar",
            spawn=None,
            scan_type="solidState",
            output_frame_of_reference="WORLD",
        )

    with sim_utils.build_simulation_context(sim_cfg=sim_utils.SimulationCfg(physics=OvPhysxCfg())) as sim:
        scene = instantiate(SceneCfg(num_envs=2, env_spacing=8.0))
        sim.register_interactive_scene(scene)
        sim.reset()
        platform = scene["platform"]
        pose = platform.data.root_link_pose_w.torch.clone()
        # A beam tilted 30 degrees from vertical intersects the top face 2.5 m below its mount.
        local_hit = pose.new_tensor([0.3 - 2.5 * math.tan(math.pi / 6), 0.0, 0.5])
        for displacement in (0.0, 0.4):
            pose[1, 0] += displacement
            platform.write_root_pose_to_sim_index(root_pose=pose)
            sim.step()
            scene.update(sim.get_physics_dt())
            data = scene["lidar"].data
            assert data.valid.torch.all()
            torch.testing.assert_close(data.points.torch[:, 0], pose[:, :3] + local_hit, atol=1e-3, rtol=0.0)
