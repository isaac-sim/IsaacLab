# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sim.spawners.materials import NewtonDeformableMaterialCfg

from isaaclab.cloner import ClonePlan

from isaaclab_contrib.deformable import DeformableObject
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


class _FakePrim:
    def __init__(self, *, valid: bool = True):
        self._valid = valid
        self.attributes = {}

    def IsValid(self) -> bool:
        return self._valid

    def CreateAttribute(self, name, *_args):
        return SimpleNamespace(IsValid=lambda: True, Set=lambda value: self.attributes.__setitem__(name, value))


class _FakeStage:
    def __init__(self, prims: dict[str, _FakePrim]):
        self._prims = prims

    def GetPrimAtPath(self, path: str) -> _FakePrim:
        return self._prims.get(path, _FakePrim(valid=False))


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


@pytest.mark.parametrize("available", [False, True])
def test_fabric_particle_sync_uses_planned_paths_and_skips_missing_sinks(monkeypatch, available):
    """Fabric bindings use actual plan IDs, even when no corresponding USD clone exists."""
    entry = _make_surface_entry()
    entry.prim_path = "/Scene/copy_[^/]+/cloth"
    entry.vis_mesh_prim_path = entry.prim_path + "/mesh"
    entry.particle_offsets = [7, 19]
    entry.particles_per_body = 3
    plan = ClonePlan(
        sources=("/Source",),
        destinations=("/Scene/copy_{}",),
        clone_mask=np.array([[True, False, True]]),
        env_ids=np.array([7, 12, 42]),
    )
    paths = [f"/Scene/copy_{env_id}/cloth/mesh" for env_id in (7, 42)]

    class _FakeManager(NewtonManager):
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

    fabric_stage = _FakeStage({path: _FakePrim() for path in paths} if available else {})
    monkeypatch.setattr("isaaclab.sim.SimulationContext.instance", lambda: SimpleNamespace(get_clone_plan=lambda: plan))
    monkeypatch.setattr(NewtonManager, "_usdrt_stage", fabric_stage)
    monkeypatch.setitem(
        sys.modules, "usdrt", SimpleNamespace(Sdf=SimpleNamespace(ValueTypeNames=SimpleNamespace(UInt=object())))
    )

    setup_registered_deformable_fabric_sync(_FakeManager)

    assert _FakeManager.marked is available
    assert _FakeManager.synced is available
    if available:
        for path, offset in zip(paths, entry.particle_offsets, strict=True):
            assert fabric_stage.GetPrimAtPath(path).attributes == {
                NewtonManager._newton_particle_offset_attr: offset,
                NewtonManager._newton_particle_count_attr: entry.particles_per_body,
            }
