# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX RayCaster backend glue."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_ov import tensor_types as TT
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.sensors.ray_caster import ray_caster as ray_caster_module

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sensors.ray_caster import MultiMeshRayCaster, MultiMeshRayCasterCfg, RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


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
    monkeypatch.setattr(ray_caster_module.OvPhysxManager, "get_physx_instance", staticmethod(lambda: fake_physx))

    sensor = _DummyRayCaster()

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


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_ray_casters_read_link_pose_after_joint_write(device):
    """Ray casters report an articulation child link at its written pose before the next step.

    Environments reset inside ``step()`` and read observations before the next physics step, while the
    GPU pipeline defers the child-link pose update of articulation writes. ``Arm`` is a 2 m bar on a
    revolute joint about the Y axis, with the pivot fixed at (0, 0, 1).
    """
    dt = 0.01
    sim_cfg = SimulationCfg(physics=OvPhysxCfg(), device=device, dt=dt)
    with build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        create_prim_from_mesh("/World/Ground", make_plane(size=(100, 100), height=0.0, center_zero=True))
        robot = Articulation(
            ArticulationCfg(
                prim_path="/World/Robot",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd"
                ),
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
                actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
            )
        )
        down = (0.0, 0.0, -1.0)
        ray_caster = RayCaster(
            RayCasterCfg(
                prim_path="/World/Robot/Arm",
                mesh_prim_paths=["/World/Ground"],
                pattern_cfg=patterns.GridPatternCfg(resolution=1.0, size=(0.0, 0.0), direction=down),
                ray_alignment="world",
            )
        )
        # A static scanner tracks the arm mesh with downward rays at x = -1 m and x = 1 m.
        sim_utils.create_prim("/World/Scanner", "Xform", translation=(0.0, 0.0, 3.0))
        scanner = MultiMeshRayCaster(
            MultiMeshRayCasterCfg(
                prim_path="/World/Scanner",
                mesh_prim_paths=[
                    MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="/World/Robot/Arm", track_mesh_transforms=True)
                ],
                pattern_cfg=patterns.GridPatternCfg(resolution=2.0, size=(2.0, 0.0), direction=down),
                ray_alignment="world",
            )
        )
        sim.reset()
        arm = robot.find_bodies("Arm")[0][0]

        def step_and_write_joint_position(position: float) -> None:
            sim.step(render=False)
            robot.update(dt)
            for sensor in (ray_caster, scanner):
                sensor.update(dt)
            robot.write_joint_position_to_sim_index(position=torch.full_like(robot.data.joint_pos.torch, position))

        # Read one sensor first after each write: an asset body getter would update the kinematics itself.
        step_and_write_joint_position(0.5)
        pose_w = torch.cat([ray_caster.data.pos_w.torch, ray_caster.data.quat_w.torch], dim=-1)
        torch.testing.assert_close(pose_w, robot.data.body_link_pose_w.torch[:, arm])

        # Pointing the arm along -x puts its top face (z = 1.05 m) under the first ray only.
        step_and_write_joint_position(math.pi)
        hits = scanner.data.ray_hits_w.torch[0]
        torch.testing.assert_close(hits[0], torch.tensor([-1.0, 0.0, 1.05], device=hits.device), atol=1e-3, rtol=0.0)
        assert torch.isinf(hits[1]).all()
