# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body synchronization."""

from __future__ import annotations

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager, XPBDSolverCfg

from usdrt import Rt

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils.configclass import configclass


@configclass
class _RenderSceneCfg(InteractiveSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


def _fabric_position(body_path: str) -> torch.Tensor:
    """Read the world position consumed by Kit/RTX from the real Fabric stage."""
    stage = sim_utils.get_current_stage(fabric=True)
    assert stage is not None, "The rendering-side Fabric stage is unavailable"

    prim = stage.GetPrimAtPath(body_path)
    assert prim.IsValid(), f"Fabric body prim does not exist: {body_path}"
    world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    assert world_matrix is not None, f"Fabric body prim has no world matrix: {body_path}"
    translation = world_matrix.ExtractTranslation()
    return torch.tensor([float(translation[i]) for i in range(3)])


class _FakeAttribute:
    def __init__(self, value_type, custom):
        self.value_type = value_type
        self.custom = custom
        self.value = None

    def Set(self, value):
        self.value = value


class _FakePrim:
    def __init__(self, valid=True):
        self.valid = valid
        self.attributes = {}
        self.applied_schemas = []
        self.created_world_matrix_attrs = 0
        self.set_world_xform_from_usd = 0

    def IsValid(self):
        return self.valid

    def CreateAttribute(self, name, value_type, custom=False):
        self.attributes[name] = _FakeAttribute(value_type, custom)
        return self.attributes[name]

    def GetAttribute(self, name):
        return self.attributes[name]

    def AddAppliedSchema(self, schema):
        self.applied_schemas.append(schema)


class _FakeStage:
    def __init__(self, prims=None):
        self.prims = prims or {}
        self.defined_prims = []

    def GetPrimAtPath(self, path):
        return self.prims.get(path, _FakePrim(valid=False))

    def DefinePrim(self, path, prim_type):
        prim = _FakePrim()
        self.prims[path] = prim
        self.defined_prims.append((path, prim_type))
        return prim


class _FakeXformable:
    def __init__(self, prim):
        self.prim = prim

    def SetWorldXformFromUsd(self):
        self.prim.set_world_xform_from_usd += 1

    def CreateFabricHierarchyWorldMatrixAttr(self):
        self.prim.created_world_matrix_attrs += 1


class _FakeFabricHierarchy:
    def __init__(self):
        self.update_world_xforms_count = 0

    def update_world_xforms(self):
        self.update_world_xforms_count += 1


class _FakeRt:
    Xformable = _FakeXformable


class _FakeValueTypeNames:
    UInt = "UInt"


class _FakeSdf:
    ValueTypeNames = _FakeValueTypeNames


class _FakeUsdrt:
    Rt = _FakeRt
    Sdf = _FakeSdf


def test_initialize_fabric_body_prims_uses_existing_fabric_prim():
    prim = _FakePrim()
    stage = _FakeStage({"/World/envs/env_0/Robot/base": prim})
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_0/Robot/base", 3)]
    )

    assert stage.defined_prims == []
    assert prim.set_world_xform_from_usd == 1
    assert prim.created_world_matrix_attrs == 0
    assert prim.GetAttribute("newton:index").value_type == "UInt"
    assert prim.GetAttribute("newton:index").custom is True
    assert prim.GetAttribute("newton:index").value == 3
    assert prim.applied_schemas == ["PhysicsRigidBodyAPI"]
    assert fabric_hierarchy.update_world_xforms_count == 1


def test_initialize_fabric_body_prims_creates_missing_body_as_xform():
    stage = _FakeStage()
    fabric_hierarchy = _FakeFabricHierarchy()

    NewtonManager._initialize_fabric_body_prims(
        stage, fabric_hierarchy, _FakeUsdrt, [("/World/envs/env_1/Robot/joints/forearm", 7)]
    )

    prim = stage.prims["/World/envs/env_1/Robot/joints/forearm"]
    assert stage.defined_prims == [("/World/envs/env_1/Robot/joints/forearm", "Xform")]
    assert prim.set_world_xform_from_usd == 0
    assert prim.created_world_matrix_attrs == 1
    assert prim.GetAttribute("newton:index").value_type == "UInt"
    assert prim.GetAttribute("newton:index").custom is True
    assert prim.GetAttribute("newton:index").value == 7
    assert prim.applied_schemas == ["PhysicsRigidBodyAPI"]
    assert fabric_hierarchy.update_world_xforms_count == 1


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_root_pose_write_is_visible_on_next_render_without_step():
    """A reset-time pose write must reach Kit/RTX on the next render.

    This reproduces the application sequence that used to render one-frame-old
    transforms: write asset state, then render without an intervening physics
    step or asset-data read. The assertion reads Fabric's world matrix, which is
    the transform consumed by Kit/RTX; USD is intentionally not written back.
    """
    device = "cuda:0"
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_RenderSceneCfg(num_envs=1, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            sim.render()

            cube = scene["cube"]
            body_path = "/World/envs/env_0/Cube"
            target_pose = torch.tensor(
                [[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )

            cube.write_root_link_pose_to_sim_index(root_pose=target_pose)

            physics_steps = sim.get_physics_step_count()
            sim.render()
            wp.synchronize_device(device)

            assert sim.get_physics_step_count() == physics_steps
            torch.testing.assert_close(
                _fabric_position(body_path),
                target_pose[0, :3].cpu(),
                rtol=0.0,
                atol=1.0e-4,
            )

            # Exercise the same application boundary with a production
            # graph-safe writer. Clearing the capture-time host flag before
            # replay ensures each render depends on replayed device-side
            # invalidation rather than Python code that ran during capture.
            env_mask = wp.ones(1, dtype=wp.bool, device=device)
            pose_buffer = target_pose.clone()
            cube.write_root_link_pose_to_sim_mask(root_pose=pose_buffer, env_mask=env_mask)
            sim.render()

            torch.cuda.synchronize(device)
            with wp.ScopedCapture(device=device) as capture:
                cube.write_root_link_pose_to_sim_mask(root_pose=pose_buffer, env_mask=env_mask)

            sim.render()

            replay_targets = (
                torch.tensor([2.5, 0.5, 1.25, 0.0, 0.0, 0.0, 1.0], device=device),
                torch.tensor([-1.25, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0], device=device),
            )
            for replay_target in replay_targets:
                pose_buffer.copy_(replay_target.unsqueeze(0))
                torch.cuda.synchronize(device)
                wp.capture_launch(capture.graph)
                wp.synchronize_device(device)

                physics_steps = sim.get_physics_step_count()
                sim.render()
                wp.synchronize_device(device)

                assert sim.get_physics_step_count() == physics_steps
                torch.testing.assert_close(
                    _fabric_position(body_path),
                    replay_target[:3].cpu(),
                    rtol=0.0,
                    atol=1.0e-4,
                )
        finally:
            sim.register_interactive_scene(None)
