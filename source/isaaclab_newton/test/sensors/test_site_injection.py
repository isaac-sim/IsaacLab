# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Site injection, sensor index validation, and hard-reset lifecycle tests."""

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonManager
from isaaclab_newton.sensors.frame_transformer.frame_transformer import FrameTransformer
from newton import ModelBuilder

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, ImuCfg, PvaCfg, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns import GridPatternCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass


@pytest.mark.parametrize("replicated", [False, True])
def test_sites_bind_once_to_clone_sources_or_explicit_builder(replicated):
    main = ModelBuilder()
    source = ModelBuilder() if replicated else main
    for leg in ("FL", "FR", "RL", "RR"):
        source.add_body(label=f"Robot/{leg}_foot")
    manager = NewtonManager(NewtonCfg())
    xform = wp.transform((1.0, 2.0, 3.0), wp.quat_identity())
    global_label = manager.register_site(None, xform)
    local_label = manager.register_site("Robot/.*_foot", xform)
    world_label = manager.register_site(None, xform, per_world=True)
    assert manager.register_site(None, xform, per_world=True) == world_label

    sources = {"Robot": source} if replicated else {}
    global_sites, body_sites, world_sites = manager.inject_sites(main, sources)
    assert main.shape_body[global_sites[global_label]] == -1
    indices = body_sites[id(source)][local_label]
    assert [source.shape_body[index] for index in indices] == list(range(4))
    assert [source.shape_label[index] for index in indices] == [f"{name}/{local_label}" for name in source.body_label]
    assert world_sites == {world_label: xform}

    # Requests persist across hard resets, but resolved sites are never added to the retained builder again.
    manager._site_index_map.update((label, (index, None)) for label, index in global_sites.items())
    manager._site_index_map[local_label] = (None, [indices])
    manager._site_index_map[world_label] = (None, [[0]])
    shape_count = main.shape_count
    assert manager.inject_sites(main, sources) == ({}, {}, {})
    assert main.shape_count == shape_count
    assert manager.register_site("Robot/.*_foot", xform) == local_label

    manager.register_site("Robot/nonexistent", xform)
    with pytest.raises(ValueError, match="matched no builder bodies"):
        manager.inject_sites(main, sources)


# ---------------------------------------------------------------------------
# Hard resets
# ---------------------------------------------------------------------------


def _cube_cfg(name: str, x: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, 0.0, 1.0)),
    )


@configclass
class SiteSensorSceneCfg(InteractiveSceneCfg):
    """Two falling cubes; the ray caster registers its site before the IMU, PVA and frame transformer.

    The frame transformer tracks a wildcard target that matches both cubes, then a single named target.
    """

    env_spacing = 4.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    cube_a = _cube_cfg("CubeA", -0.5)
    cube_b = _cube_cfg("CubeB", 0.5)
    ray_caster = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/CubeA",
        mesh_prim_paths=["/World/ground"],
        pattern_cfg=GridPatternCfg(resolution=1.0, size=(0.0, 0.0)),
    )
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/CubeB")
    pva = PvaCfg(prim_path="{ENV_REGEX_NS}/CubeB")
    frame_transformer = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/CubeB",
        target_frames=[
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Cube.*"),
            FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/CubeA", name="cube_a"),
        ],
    )


@pytest.mark.parametrize("num_envs", [1, 2], ids=["one_env", "two_envs"])
@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.integration
def test_hard_resets_keep_sensor_sites(num_envs, device):
    """Site sensors stay on their bodies through hard resets, which leave one native IMU sensor.

    Sensors registered their sites again on ``STOP`` and the kept builder received them a second time: the second
    ``sim.reset()`` of two environments failed, and in one environment the ray caster, which registers its site
    once, followed the IMU's cube. When the frame transformer initialized again, its single target also took the
    name of the wildcard target's second match.
    """
    sim_cfg = SimulationCfg(device=device, dt=0.005, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(SiteSensorSceneCfg(num_envs=num_envs))
        spin = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]] * num_envs, device=sim.device)
        first_pass = None
        for _ in range(3):
            sim.reset()
            scene["cube_b"].write_root_velocity_to_sim_index(root_velocity=spin)
            for _ in range(2):
                sim.step(render=False)
                scene.update(sim.get_physics_dt())

            assert scene["frame_transformer"].data.target_frame_names == ["CubeA", "CubeB", "cube_a"]
            ray_pos_w = scene["ray_caster"].data.pos_w.torch
            torch.testing.assert_close(ray_pos_w, scene["cube_a"].data.root_link_pos_w.torch)
            readings = {
                "imu_ang_vel_b": scene["imu"].data.ang_vel_b.torch.clone(),
                "pva_pos_w": scene["pva"].data.pos_w.torch.clone(),
                "target_pos_source": scene["frame_transformer"].data.target_pos_source.torch.clone(),
            }
            if first_pass is None:
                first_pass = readings
            for name, reading in readings.items():
                torch.testing.assert_close(reading, first_pass[name], msg=lambda message: f"{name}: {message}")
            assert len(sim.backend_manager.backend.imu_sensors) == 1


# ---------------------------------------------------------------------------
# FrameTransformer._validate_site_map
# ---------------------------------------------------------------------------


def _make_site_map(
    source_per_world: list[list[int]],
    target_per_worlds: list[list[list[int]]],
    world_origin_idx: int = 0,
) -> dict:
    m = {
        "world_origin": (world_origin_idx, None),
        "source": (None, source_per_world),
    }
    for i, pw in enumerate(target_per_worlds):
        m[f"target_{i}"] = (None, pw)
    return m


class TestSourceValidation:
    def test_source_wrong_env_count_raises(self):
        # site map has 1 world entry but num_envs=2
        site_map = _make_site_map([[10]], [])
        with pytest.raises(ValueError, match="1 world entries.*expected 2"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_zero_in_env_raises(self):
        site_map = _make_site_map([[], [20]], [])
        with pytest.raises(ValueError, match="matched 0 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)

    def test_source_two_in_env_raises(self):
        site_map = _make_site_map([[10, 11], [20]], [])
        with pytest.raises(ValueError, match="matched 2 bodies in env 0"):
            FrameTransformer._validate_site_map("source", "/Robot/base", [], [], site_map, num_envs=2)


class TestTargetValidation:
    def test_target_zero_bodies_raises(self):
        site_map = _make_site_map([[10], [20]], [[[], []]])
        with pytest.raises(ValueError, match="matched no bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )

    def test_target_non_uniform_raises(self):
        site_map = _make_site_map([[10], [20]], [[[30, 31], [40]]])
        with pytest.raises(ValueError, match="different numbers of bodies"):
            FrameTransformer._validate_site_map(
                "source", "/Robot/base", ["target_0"], ["/Robot/foot.*"], site_map, num_envs=2
            )


# ---------------------------------------------------------------------------
# FrameTransformer._build_sensor_index_lists
# ---------------------------------------------------------------------------


def _call(source_indices, target_per_world, target_frame_body_names, shape_labels, world_origin_idx, num_envs):
    return FrameTransformer._build_sensor_index_lists(
        source_indices, target_per_world, target_frame_body_names, shape_labels, world_origin_idx, num_envs
    )


class TestZeroTargets:
    def test_zero_targets_shapes_refs(self):
        """0 targets: shapes/refs contain only source entries."""
        names, tgt_per_tgt, shapes, refs = _call(
            source_indices=[10, 11],
            target_per_world=[],
            target_frame_body_names=[],
            shape_labels={},
            world_origin_idx=0,
            num_envs=2,
        )
        assert shapes == [10, 11]
        assert refs == [0, 0]
        assert names == []
        assert tgt_per_tgt == []
