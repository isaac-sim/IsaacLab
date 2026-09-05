# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
from types import SimpleNamespace
from unittest import mock

import newton
import numpy as np
import pytest
import warp as wp
from isaaclab_newton.cloner.newton_clone_utils import replicate_builder_mapping
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.materials import NewtonDeformableMaterialCfg

from isaaclab_contrib.deformable import DeformableObject
from isaaclab_contrib.deformable.deformable_object import (
    DeformableRegistryEntry,
    add_deformable_entry_to_builder,
    add_registered_deformables_to_builder,
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
        prim_path="{ENV_REGEX_NS}/cloth",
        sim_mesh_prim_path="{ENV_REGEX_NS}/cloth/mesh",
        vis_mesh_prim_path="{ENV_REGEX_NS}/cloth/mesh",
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


def _replicate_deformable_worlds(builder, source, positions, quaternions):
    """Replicate ``source`` into one world per row of ``positions``, running the deformable hook."""
    positions = np.asarray(positions, dtype=np.float32)
    return replicate_builder_mapping(
        builder,
        ("/World/envs/env_0",),
        np.ones((1, len(positions)), dtype=bool),
        positions,
        np.asarray(quaternions, dtype=np.float32),
        {"/World/envs/env_0": source},
        ("/World/envs/env_{}",),
        np.arange(len(positions)),
        per_world_builder_hooks=(add_registered_deformables_to_builder,),
    )


def test_homogeneous_deformables_are_added_once_and_replicated(monkeypatch):
    """Test that translation-only worlds replicate each source deformable mesh once."""
    entries = [_make_surface_entry(), _make_surface_entry()]
    entries[1].init_pos = (0.0, 2.0, 0.0)
    monkeypatch.setattr(NewtonManager, "_deformable_registry", entries)
    source = newton.ModelBuilder()
    builder = newton.ModelBuilder()
    builder.add_particle(pos=(-1.0, -2.0, -3.0), vel=(0.0, 0.0, 0.0), mass=1.0, radius=0.1)
    positions = np.asarray([[2.0, 3.0, 0.0], [5.0, 3.0, 0.0], [2.0, 7.0, 0.0]], dtype=np.float32)

    with (
        mock.patch.object(source, "add_cloth_mesh", wraps=source.add_cloth_mesh) as add_cloth_mesh,
        mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate,
    ):
        _replicate_deformable_worlds(builder, source, positions, [[0.0, 0.0, 0.0, 1.0]] * 3)

    assert add_cloth_mesh.call_count == len(entries)
    replicate.assert_called_once()
    assert builder.world_count == 3
    assert [entry.particle_offsets for entry in entries] == [[1, 7, 13], [4, 10, 16]]
    assert [entry.particles_per_body for entry in entries] == [3, 3]
    np.testing.assert_array_equal(np.asarray(builder.tri_indices), np.arange(1, 19).reshape(-1, 3))
    # Both entries carry the same mesh rotated 90 deg about z, the second offset 2 m along y.
    local_particles = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 3.0, 0.0], [-1.0, 2.0, 0.0]]
    )
    expected_particles = np.concatenate(
        ([[-1.0, -2.0, -3.0]], (positions[:, None, :] + local_particles).reshape(-1, 3))
    )
    np.testing.assert_allclose(np.asarray(builder.particle_q), expected_particles, atol=1.0e-6)


def test_rotated_deformable_worlds_keep_per_world_fallback(monkeypatch):
    """Test that worlds with differing rotations fall back to per-world construction."""
    entry = _make_surface_entry()
    monkeypatch.setattr(NewtonManager, "_deformable_registry", [entry])
    source = newton.ModelBuilder()
    builder = newton.ModelBuilder()
    half_sqrt = math.sqrt(0.5)

    with mock.patch.object(builder, "replicate", wraps=builder.replicate) as replicate:
        _replicate_deformable_worlds(
            builder,
            source,
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, half_sqrt, half_sqrt]],
        )

    replicate.assert_not_called()
    assert source.particle_count == 0
    assert builder.world_count == 2
    assert entry.particle_offsets == [0, 3]
    assert entry.particles_per_body == 3
    np.testing.assert_allclose(
        np.asarray(builder.particle_q)[3:],
        [[4.0, 1.0, 0.0], [3.0, 1.0, 0.0], [4.0, 0.0, 0.0]],
        atol=1.0e-6,
    )


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
