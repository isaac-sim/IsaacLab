# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Simulation tests for the :class:`RayCaster` and :class:`MultiMeshRayCaster` sensors.

Covers ray alignment modes, drift resampling, pose tracking of physics-body parents, and the multi-mesh
target resolution through the clone plan and tracked body views.
"""

from typing import Literal

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import numpy as np
import pytest
import torch
import warp as wp

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.cloner.clone_plan import ClonePlan
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.math import quat_from_euler_xyz

pytestmark = pytest.mark.integration

GROUND_PATH = "/World/Ground"
DT = 0.01
SINGLE_DOWN_RAY = patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=(0.0, 0.0, -1.0))


@pytest.fixture
def sim():
    """A blank stage with a flat ground plane at z=0."""
    sim_utils.create_new_stage()
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT))
    create_prim_from_mesh(GROUND_PATH, make_plane(size=(100, 100), height=0.0, center_zero=True))
    sim_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear_instance()


def _ray_caster_cfg(prim_path: str, alignment: Literal["base", "yaw", "world"] = "world", **kwargs) -> RayCasterCfg:
    """A single downward ray mounted on ``prim_path``."""
    return RayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=[GROUND_PATH],
        update_period=0,
        pattern_cfg=SINGLE_DOWN_RAY,
        ray_alignment=alignment,
        **kwargs,
    )


def _euler_xyzw(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    quat = quat_from_euler_xyz(torch.tensor([roll]), torch.tensor([pitch]), torch.tensor([yaw]))
    return tuple(quat[0].tolist())


def _create_rigid_body(path: str, translation, orientation=None, articulation: bool = False, kinematic: bool = False):
    """Create a rigid body (optionally an articulation root) with a small collision cube so PhysX tracks it."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim(path, "Xform", translation=translation, orientation=orientation, stage=stage)
    prim = stage.GetPrimAtPath(path)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    if articulation:
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)
    if kinematic:
        prim.GetAttribute("physics:kinematicEnabled").Set(True)
    UsdGeom.Cube.Define(stage, f"{path}/CollisionCube").CreateSizeAttr().Set(0.1)
    UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(f"{path}/CollisionCube"))


"""
Alignment modes
"""


def test_alignment_modes(sim):
    """World alignment ignores the sensor orientation, base alignment rotates the ray direction, and yaw
    alignment rotates only the ray start positions."""
    orientation = _euler_xyzw(0.0, np.pi / 6, np.pi / 4)  # 30 deg pitch, 45 deg yaw
    sensors = {}
    for alignment in ("world", "base", "yaw"):
        path = f"/World/Sensor_{alignment}"
        sim_utils.create_prim(path, "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
        # the ray starts 1 m ahead of the sensor so the yaw rotation of the start position is observable
        offset = RayCasterCfg.OffsetCfg(pos=(1.0, 0.0, 0.0))
        sensors[alignment] = RayCaster(_ray_caster_cfg(path, alignment, offset=offset))
    sim.reset()
    hits = {}
    for alignment, sensor in sensors.items():
        sensor.update(DT)
        hits[alignment] = sensor.data.ray_hits_w.torch[0, 0].cpu().double()

    # all modes hit the ground plane
    for alignment, hit in hits.items():
        assert abs(hit[2].item()) < 0.05, f"{alignment} mode must hit z=0, got {hit[2].item():.3f}"
    # world: the offset is not rotated and the ray goes straight down from x=1
    torch.testing.assert_close(hits["world"][:2], torch.tensor([1.0, 0.0]).double(), atol=0.05, rtol=0)
    # yaw: the offset is rotated by the 45 deg yaw only; the ray still goes straight down
    torch.testing.assert_close(hits["yaw"][:2], torch.tensor([np.cos(np.pi / 4), np.sin(np.pi / 4)]), atol=0.05, rtol=0)
    # base: the full orientation applies to the offset and the direction, so the ray lands further away
    torch.testing.assert_close(hits["base"][:2], torch.tensor([0.0, 0.0]).double(), atol=0.05, rtol=0)


def test_base_alignment_pitch_tilts_ray(sim):
    """A sensor pitched by 30 deg in base alignment lands ``-2 * tan(30 deg)`` ahead from a height of 2 m."""
    orientation = _euler_xyzw(0.0, np.pi / 6, 0.0)
    sim_utils.create_prim("/World/SensorWorld", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sim_utils.create_prim("/World/SensorBase", "Xform", translation=(0.0, 0.0, 2.0), orientation=orientation)
    sensor_world = RayCaster(_ray_caster_cfg("/World/SensorWorld", "world"))
    sensor_base = RayCaster(_ray_caster_cfg("/World/SensorBase", "base"))
    sim.reset()
    sensor_world.update(DT)
    sensor_base.update(DT)

    hit_world = sensor_world.data.ray_hits_w.torch[0, 0].cpu().double()
    hit_base = sensor_base.data.ray_hits_w.torch[0, 0].cpu().double()
    torch.testing.assert_close(hit_world, torch.tensor([0.0, 0.0, 0.0]).double(), atol=0.05, rtol=0)
    torch.testing.assert_close(hit_base, torch.tensor([-2.0 * np.tan(np.pi / 6), 0.0, 0.0]), atol=0.05, rtol=0)


"""
Offsets, drift, and pose tracking
"""


def test_offset_shifts_rays_but_not_pos_w(sim):
    """``cfg.offset.pos`` shifts the ray starts but ``data.pos_w`` keeps reporting the parent body pose.

    Regression test: baking the offset into the view transform made height-scan observations
    (``pos_w.z - hit.z``) include the offset.
    """
    body_pos = (0.0, 0.0, 0.6)
    sim_utils.create_prim("/World/Robot", "Xform", translation=body_pos)
    cfg = RayCasterCfg(
        prim_path="/World/Robot",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        mesh_prim_paths=[GROUND_PATH],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.5, size=(1.0, 1.0)),
        ray_alignment="yaw",
    )
    sensor = RayCaster(cfg)
    sim.reset()
    sensor.update(DT)

    torch.testing.assert_close(sensor.data.pos_w.torch[0].cpu(), torch.tensor(body_pos), atol=1e-3, rtol=0)
    hits_z = sensor.data.ray_hits_w.torch[0, :, 2]
    assert torch.isfinite(hits_z).all()
    torch.testing.assert_close(hits_z, torch.zeros_like(hits_z), atol=1e-3, rtol=0)


def test_reset_resamples_drift(sim):
    """``reset()`` resamples the drift within ``drift_range``."""
    sim_utils.create_prim("/World/Sensor", "Xform", translation=(0.0, 0.0, 2.0))
    lo, hi = 0.01, 0.05
    sensor = RayCaster(_ray_caster_cfg("/World/Sensor", drift_range=(lo, hi)))
    sim.reset()

    samples = []
    for _ in range(3):
        sensor.reset()
        drift = sensor.drift.torch.clone()
        assert drift.shape == (1, 3)
        assert (drift >= lo - 1e-6).all() and (drift <= hi + 1e-6).all()
        samples.append(drift)
    assert not all(torch.equal(samples[0], sample) for sample in samples[1:]), "reset() must resample the drift"


@pytest.mark.parametrize("articulation", [False, True], ids=["rigid_body", "articulation"])
def test_tracks_physics_body_parent(sim, articulation):
    """The sensor pose follows its physics-body parent, both for rigid bodies and articulation roots."""
    body_path = "/World/Body"
    initial_pos = (3.0, 4.0, 5.0)
    _create_rigid_body(body_path, initial_pos, articulation=articulation)
    sim_utils.update_stage()
    sensor = RayCaster(_ray_caster_cfg(body_path))
    sim.reset()
    sensor.update(DT, force_recompute=True)

    torch.testing.assert_close(sensor.data.pos_w.torch[0].cpu(), torch.tensor(initial_pos), atol=0.15, rtol=0)
    assert abs(sensor.data.ray_hits_w.torch[0, 0, 2].item()) < 0.5, "downward ray should hit the ground"

    # the body falls under gravity and the sensor pose must follow
    for _ in range(50):
        sim.step(render=False)
    sensor.update(DT, force_recompute=True)
    assert sensor.data.pos_w.torch[0, 2].item() < initial_pos[2] - 0.05


"""
Multi-mesh ray caster
"""


def _multi_mesh_cfg(prim_path: str, mesh_prim_paths: list) -> MultiMeshRayCasterCfg:
    return MultiMeshRayCasterCfg(
        prim_path=prim_path,
        mesh_prim_paths=mesh_prim_paths,
        update_period=0,
        pattern_cfg=SINGLE_DOWN_RAY,
        ray_alignment="world",
    )


def test_multi_mesh_env_mask_preserves_masked_buffers(sim):
    """Environments masked out of an update keep their previous ray hits."""
    sim_utils.create_prim("/World/Sensor", "Xform", translation=(0.0, 0.0, 3.0))
    sensor = MultiMeshRayCaster(_multi_mesh_cfg("/World/Sensor", [GROUND_PATH]))
    sim.reset()
    sensor.update(DT)
    hits_before = sensor.data.ray_hits_w.torch.clone()
    assert torch.isfinite(hits_before).all()

    sensor._update_buffers_impl(wp.array([False], dtype=wp.bool, device=sensor.device))
    torch.testing.assert_close(sensor.data.ray_hits_w.torch, hits_before, atol=0.0, rtol=0.0)


def test_multi_mesh_uses_clone_plan_geometry_and_backend_pose(sim):
    """The clone plan selects the source geometry per env while the physics body view supplies live poses."""
    num_envs = 3
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/envs", "Xform", stage=stage)
    for env_id in range(num_envs):
        sim_utils.create_prim(f"/World/envs/env_{env_id}", "Xform", translation=(3.0 * env_id, 0.0, 0.0), stage=stage)
        sim_utils.create_prim(f"/World/envs/env_{env_id}/Sensor", "Xform", translation=(0.0, 0.0, 3.0), stage=stage)
        _create_rigid_body(f"/World/envs/env_{env_id}/Object", (0.0, 0.0, 0.0), kinematic=True)
    # only the source envs carry authored mesh parts; env_2 relies on the clone plan for its geometry
    for env_id in range(2):
        part_path = f"/World/envs/env_{env_id}/Object/part_0"
        sim_utils.create_prim(part_path, "Xform", stage=stage)
        UsdGeom.Cube.Define(stage, f"{part_path}/Mesh").CreateSizeAttr().Set(0.35)
    sim.set_clone_plan(
        ClonePlan(
            sources=("/World/envs/env_0/Object", "/World/envs/env_1/Object"),
            destinations=("/World/envs/env_{}/Object", "/World/envs/env_{}/Object"),
            clone_mask=np.asarray([[True, False, True], [False, True, False]], dtype=np.bool_),
            env_ids=np.arange(num_envs, dtype=np.int64),
            positions=None,
            cfg_rows={},
        )
    )
    sim_utils.update_stage()

    target = MultiMeshRayCasterCfg.RaycastTargetCfg(
        prim_expr="{ENV_REGEX_NS}/Object/part_[^/]*", track_mesh_transforms=True
    )
    cfg = _multi_mesh_cfg("{ENV_REGEX_NS}/Sensor", [target])
    cfg.pattern_cfg = patterns.GridPatternCfg(resolution=0.5, size=(1.0, 0.0), direction=(0.0, 0.0, -1.0))
    sensor = MultiMeshRayCaster(cfg)
    sim.reset()
    sensor.update(DT, force_recompute=True)

    assert not stage.GetPrimAtPath("/World/envs/env_2/Object/part_0").IsValid()
    mesh_ids = wp.to_torch(sensor._mesh_ids_wp).cpu()
    assert mesh_ids.shape == (num_envs, 1)
    assert mesh_ids[2, 0] == mesh_ids[0, 0], "env_2 must reuse the env_0 source geometry"
    mesh_positions = wp.to_torch(sensor._mesh_positions_w).cpu()
    torch.testing.assert_close(mesh_positions[:, 0, 0], torch.tensor([0.0, 3.0, 6.0]), atol=0.15, rtol=0.0)
    assert torch.isfinite(sensor.data.ray_hits_w.torch).any(dim=(1, 2)).all(), "every env must hit its object"


def test_multi_mesh_tracked_target_bakes_child_offset(sim):
    """Tracked geometry below a rotated body is stored relative to the body; the pose table holds the body pose."""
    yaw90 = _euler_xyzw(0.0, 0.0, np.pi / 2)
    body_path = "/World/DynamicBody"
    _create_rigid_body(body_path, (0.0, 0.0, 2.0), orientation=yaw90, kinematic=True)
    # a child plane offset by (1, 0, 0) in the body frame ends up at world (0, 1, 2) after the 90 deg yaw
    child_path = f"{body_path}/OffsetMesh"
    sim_utils.create_prim(child_path, "Xform", translation=(1.0, 0.0, 0.0))
    create_prim_from_mesh(f"{child_path}/Plane", make_plane(size=(2, 2), height=0.0, center_zero=True))
    sim_utils.update_stage()
    sim_utils.create_prim("/World/SensorMount", "Xform", translation=(0.0, 1.0, 5.0))

    target = MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=child_path, track_mesh_transforms=True)
    sensor = MultiMeshRayCaster(_multi_mesh_cfg("/World/SensorMount", [target]))
    sim.reset()
    sensor.update(DT, force_recompute=True)

    mesh_pos = wp.to_torch(sensor._mesh_positions_w)[0, 0].cpu()
    torch.testing.assert_close(mesh_pos, torch.tensor([0.0, 0.0, 2.0]), atol=0.15, rtol=0.0)
    hit = sensor.data.ray_hits_w.torch[0, 0].cpu()
    torch.testing.assert_close(hit, torch.tensor([0.0, 1.0, 2.0]), atol=0.15, rtol=0.0)
