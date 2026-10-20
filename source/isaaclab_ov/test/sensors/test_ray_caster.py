# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass, instantiate


class _FakeBinding:
    shape = (3, 7)

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


def test_multi_mesh_tracks_multiple_body_types():
    """Distinct targets follow the native body table even when its order differs from scene declarations."""

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        left = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Left",
            spawn=sim_utils.CuboidCfg(
                size=(0.5, 0.5, 0.5),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassCfg(mass=1.0),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.75, 0.0, 0.0)),
        )
        right = left.replace(
            prim_path="{ENV_REGEX_NS}/Right", init_state=RigidObjectCfg.InitialStateCfg(pos=(0.75, 0.0, 0.0))
        )
        scanner = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Left",
            offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(0.75, 0.0, 2.0)),
            mesh_prim_paths=[
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr=f"{{ENV_REGEX_NS}}/{body}")
                for body in ("Left", "Right")
            ],
            pattern_cfg=patterns.GridPatternCfg(resolution=1.5, size=(1.5, 0.0)),
            ray_alignment="world",
        )

    cfg = sim_utils.SimulationCfg(physics=OvPhysxCfg(), gravity=(0.0, 0.0, 0.0))
    with sim_utils.build_simulation_context(sim_cfg=cfg) as sim:
        scene = instantiate(SceneCfg(num_envs=2, env_spacing=4.0))
        sim.reset()
        sensor = scene["scanner"]
        expected = torch.tensor([[[-0.75, 0.0, 0.25], [0.75, 0.0, 0.25]]], device=sim.device)
        expected = expected + scene.env_origins[:, None, :]
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)

        pose = scene["right"].data.root_link_pose_w.torch.clone()
        pose[:, 2] += torch.tensor([0.25, 0.5], device=sim.device)
        scene["right"].write_root_link_pose_to_sim_index(root_pose=pose)
        sensor.reset()
        expected[:, 1, 2] += torch.tensor([0.25, 0.5], device=sim.device)
        torch.testing.assert_close(sensor.data.ray_hits_w.torch, expected)
