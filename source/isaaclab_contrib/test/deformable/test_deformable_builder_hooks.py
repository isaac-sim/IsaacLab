# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.cloner.replicate import NewtonReplicateContext
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.materials import NewtonDeformableMaterialCfg

from isaaclab.assets.deformable_object.base_deformable_object import BaseDeformableObject
from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
from isaaclab.cloner.usd import UsdReplicateContext

from isaaclab_contrib.deformable import DeformableObject, VBDSolverCfg
from isaaclab_contrib.deformable.deformable_object import (
    DeformableRegistryEntry,
    add_deformable_entry_to_builder,
    setup_registered_deformable_fabric_sync,
)


class _FakeBuilder:
    def __init__(self):
        self.particle_count = 0
        self.cloth_meshes = []

    def add_cloth_mesh(self, **kwargs) -> None:
        self.cloth_meshes.append(kwargs)
        self.particle_count += len(kwargs["vertices"])


class _FakePath:
    def __init__(self, path: str):
        self.pathString = path


class _FakePrim:
    def __init__(self, path: str, *, valid: bool = True):
        self._path = path
        self._valid = valid

    def IsValid(self) -> bool:
        return self._valid

    def GetPath(self) -> _FakePath:
        return _FakePath(self._path)


class _FakeStage:
    def __init__(self, prims: dict[str, _FakePrim]):
        self._prims = prims

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self._prims.get(path, _FakePrim(path, valid=False))


def _make_surface_entry() -> DeformableRegistryEntry:
    half_sqrt = math.sqrt(0.5)
    return DeformableRegistryEntry(
        prim_path="/World/envs/env_.*/cloth",
        sim_mesh_prim_path="/World/envs/env_.*/cloth/mesh",
        vis_mesh_prim_path="/World/envs/env_.*/cloth/mesh",
        vertices=[
            wp.vec3(0.0, 0.0, 0.0),
            wp.vec3(1.0, 0.0, 0.0),
            wp.vec3(0.0, 1.0, 0.0),
        ],
        indices=[0, 1, 2],
        init_pos=(1.0, 0.0, 0.0),
        init_rot=(0.0, 0.0, half_sqrt, half_sqrt),
        deformable_type="surface",
    )


def _vec3_as_tuple(value) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def test_deformable_package_exports_public_symbols():
    """Test that deformable symbols are exported from the package root."""
    assert DeformableObject.__name__ == "DeformableObject"
    assert VBDSolverCfg.__name__ == "VBDSolverCfg"


def test_newton_material_defaults_match_registry_defaults():
    """Test that Newton material cfg defaults match the deformable registry defaults."""
    material_cfg = NewtonDeformableMaterialCfg()

    assert material_cfg.density == DeformableRegistryEntry.density
    assert material_cfg.particle_radius == DeformableRegistryEntry.particle_radius


def test_builder_hook_applies_env_quaternion_to_deformable_entry():
    """Test that deformable builder placement honors the environment quaternion."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    half_sqrt = math.sqrt(0.5)

    add_deformable_entry_to_builder(
        builder,
        entry,
        env_idx=0,
        env_position=[10.0, 20.0, 30.0],
        env_rotation=[0.0, 0.0, half_sqrt, half_sqrt],
    )

    mesh = builder.cloth_meshes[0]
    rotated_x_axis = wp.quat_rotate(mesh["rot"], wp.vec3(1.0, 0.0, 0.0))

    assert _vec3_as_tuple(mesh["pos"]) == pytest.approx((10.0, 21.0, 30.0))
    assert _vec3_as_tuple(rotated_x_axis) == pytest.approx((-1.0, 0.0, 0.0), abs=1e-6)
    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_builder_hook_resets_entry_offsets_on_first_environment():
    """Test that repeated model rebuilds do not accumulate stale particle offsets."""
    entry = _make_surface_entry()
    builder = _FakeBuilder()
    identity = [0.0, 0.0, 0.0, 1.0]

    add_deformable_entry_to_builder(builder, entry, 0, [0.0, 0.0, 0.0], identity)
    add_deformable_entry_to_builder(builder, entry, 1, [1.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0, 3]

    rebuilt_builder = _FakeBuilder()
    add_deformable_entry_to_builder(rebuilt_builder, entry, 0, [0.0, 0.0, 0.0], identity)

    assert entry.particle_offsets == [0]
    assert entry.particles_per_body == 3


def test_newton_deformable_queues_usd_and_newton_replication(monkeypatch):
    """Test that Newton deformables participate in both clone products."""
    cfg = SimpleNamespace()

    def fake_base_init(self, cfg):
        self.cfg = cfg
        self._DTYPE_TO_TORCH_TRAILING_DIMS = {}
        self._initialize_handle = None
        self._invalidate_initialize_handle = None
        self._prim_deletion_handle = None
        self._debug_vis_handle = None
        self._physics_ready_handle = None

    monkeypatch.setattr(BaseDeformableObject, "__init__", fake_base_init)
    monkeypatch.setattr(DeformableObject, "_register_deformable", lambda self: object())
    REPLICATION_QUEUE.clear()

    try:
        DeformableObject(cfg)
        queued_contexts = [ctx_cls for queued_cfg, ctx_cls in REPLICATION_QUEUE if queued_cfg is cfg]
    finally:
        REPLICATION_QUEUE.clear()

    assert queued_contexts == [UsdReplicateContext, NewtonReplicateContext]


def test_fabric_particle_sync_skips_missing_fabric_prim(monkeypatch):
    """Test that missing Fabric prims are skipped before attributes are authored."""
    entry = _make_surface_entry()
    entry.particle_offsets = [7]
    entry.particles_per_body = 3
    resolved_path = "/World/envs/env_0/cloth/mesh"

    class _FakeManager:
        _clone_physics_only = False
        _deformable_registry = [entry]
        marked = False
        synced = False

        @classmethod
        def _mark_particles_dirty(cls):
            cls.marked = True

        @classmethod
        def sync_particles_to_usd(cls):
            cls.synced = True

    usd_stage = _FakeStage({resolved_path: _FakePrim(resolved_path)})
    fabric_stage = _FakeStage({})

    monkeypatch.setattr(
        "isaaclab.sim.utils.stage.get_current_stage", lambda fabric=False: fabric_stage if fabric else usd_stage
    )
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", fabric_stage)
    monkeypatch.setitem(
        sys.modules, "usdrt", SimpleNamespace(Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(UInt=object())))
    )

    setup_registered_deformable_fabric_sync(_FakeManager)

    assert not _FakeManager.marked
    assert not _FakeManager.synced
