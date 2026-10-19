# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils import configclass


class _FakeBinding:
    shape = (3, 7)
    prim_paths = [f"/World/envs/env_{i}/Robot/base" for i in range(3)]

    def read(self, dst):
        pass

    def destroy(self):
        pass


class _FakePhysx:
    def __init__(self):
        self.calls = []

    def create_tensor_binding(self, *, pattern, tensor_type):
        self.calls.append((pattern, tensor_type))
        return _FakeBinding()


class _DummyRayCaster(ray_caster_module._OvPhysxRayCasterMixin):
    def __init__(self):
        self.cfg = SimpleNamespace(prim_path="/World/envs/env_[^/]+/Robot/base/ray")
        self._device = "cpu"
        self._resolved = (
            "/World/envs/env_[^/]+/Robot/base",
            (0.1, 0.2, 0.3),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _resolve_rigid_body_ancestor_expr(self):
        return self._resolved

    def _initialize_static_pose_tracking(self, prims):
        raise AssertionError("dynamic clone-plan sources should not fall back to static USD pose tracking")


def test_initialize_pose_tracking_binds_body_glob_and_replicates_offset(monkeypatch):
    """The resolved rigid-body expression is bound as a glob, and its fixed offset is replicated per frame."""
    fake_physx = _FakePhysx()

    # No destination USD prims exist: resolution must go through the shared clone-plan resolver.
    monkeypatch.setattr(ray_caster_module.sim_utils, "find_matching_prims", lambda _path: [])
    sensor = _DummyRayCaster()
    sensor._backend_manager = SimpleNamespace(get_physx_instance=lambda: fake_physx)

    sensor._initialize_pose_tracking()

    assert fake_physx.calls == [("/World/envs/env_*/Robot/base", TT.RIGID_BODY_POSE)]
    assert sensor.count == 3
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_pos_wp),
        torch.tensor([[0.1, 0.2, 0.3]] * 3, dtype=torch.float32),
    )
    torch.testing.assert_close(
        wp.to_torch(sensor._offset_quat_wp),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32),
    )


@configclass
class _VariantSceneCfg(InteractiveSceneCfg):
    """A ray caster on a crate next to an object with two variants, so the environments are cloned in batches."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")
    crate = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 0.1)),
    )
    # both variants have their top face at z = 0.2 m
    variant = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Variant",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2)), sim_utils.CuboidCfg(size=(0.3, 0.1, 0.2))],
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 0.1)),
    )
    # one ray straight down onto the variant, a tracked target
    scanner = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Crate",
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.0, 1.0, 1.0)),
        mesh_prim_paths=["/World/ground", MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Variant")],
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(0.0, 0.0)),
        ray_alignment="world",
    )


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_ray_caster_follows_environment_order_with_variants(device):
    """Ray-caster rows and their tracked target belong to their own environment in a scene with variants.

    On GPU, OVPhysX discovers the bodies of each variant batch in clone order, not in environment order.
    """
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_VariantSceneCfg(num_envs=4))
        sim.reset()

        variant_pos_w = scene["variant"].data.root_pos_w.torch
        hits = scene["scanner"].data.ray_hits_w.torch[:, 0]
        torch.testing.assert_close(hits, variant_pos_w + torch.tensor([0.0, 0.0, 0.1], device=device))
