# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body synchronization."""

from __future__ import annotations

import contextlib

from isaaclab.app import AppLauncher

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager, VBDSolverCfg, XPBDSolverCfg

from pxr import Gf as UsdGf
from pxr import UsdGeom
from usdrt import Gf, Rt

import isaaclab.sim as sim_utils
from isaaclab.assets import CableObjectCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg
from isaaclab.utils import math as math_utils
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


@configclass
class _CableRenderSceneCfg(InteractiveSceneCfg):
    cable: CableObjectCfg = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=((0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0), (0.0, 0.6, 1.0)),
            physics_material=CableMaterialCfg(
                thickness=0.02, density=500.0, stretch_stiffness=1.0e5, bend_stiffness=1.0e3
            ),
        ),
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


def _fabric_scale(body_path: str) -> torch.Tensor:
    """Read the world scale consumed by Kit/RTX from the real Fabric stage."""
    stage = sim_utils.get_current_stage(fabric=True)
    assert stage is not None, "The rendering-side Fabric stage is unavailable"

    prim = stage.GetPrimAtPath(body_path)
    assert prim.IsValid(), f"Fabric body prim does not exist: {body_path}"
    world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    assert world_matrix is not None, f"Fabric body prim has no world matrix: {body_path}"
    scale = Gf.Transform(world_matrix).GetScale()
    return torch.tensor([float(scale[i]) for i in range(3)])


def _fabric_curve_points_world(curve_path: str) -> torch.Tensor:
    """Read curve points from Fabric and transform them to world space."""
    stage = sim_utils.get_current_stage(fabric=True)
    prim = stage.GetPrimAtPath(curve_path)
    assert prim.IsValid(), f"Fabric curve prim does not exist: {curve_path}"
    points_attr = prim.GetAttribute("points")
    points_attr.SyncDataToCpu()
    points = points_attr.Get()
    world_matrix = Rt.Xformable(prim).GetFabricHierarchyWorldMatrixAttr().Get()
    transformed = [
        [float(value) for value in world_matrix.Transform(Gf.Vec3d(*[float(value) for value in point]))]
        for point in points
    ]
    return torch.tensor(transformed)


def _expected_cable_points_world(cable, env_id: int = 0) -> torch.Tensor:
    """Reconstruct curve points from Newton body and shape state."""
    model = NewtonManager.get_model()
    body_q = wp.to_torch(NewtonManager.get_state_0().body_q).cpu()
    shape_transform = wp.to_torch(model.shape_transform).cpu()
    shape_scale = wp.to_torch(model.shape_scale).cpu()
    root_id = int(cable.data._sim_bind_root_body_ids.numpy()[env_id])
    link_ids = [int(value) for value in cable.data._sim_bind_link_body_ids.numpy()[env_id]]
    body_ids = [root_id, *link_ids]

    negative_endpoints = []
    positive_endpoints = []
    for body_id in body_ids:
        shape_id = int(model.body_shapes[body_id][0])
        body_pose = body_q[body_id]
        shape_pose = shape_transform[shape_id]
        for sign, endpoints in ((-1.0, negative_endpoints), (1.0, positive_endpoints)):
            local_axis = torch.tensor([0.0, 0.0, sign], dtype=shape_scale.dtype)
            local_endpoint = local_axis * shape_scale[shape_id, 1]
            body_endpoint = shape_pose[:3] + math_utils.quat_apply(shape_pose[3:], local_endpoint)
            endpoints.append(body_pose[:3] + math_utils.quat_apply(body_pose[3:], body_endpoint))

    points = [negative_endpoints[0]]
    points.extend(
        0.5 * (positive_endpoints[index - 1] + negative_endpoints[index]) for index in range(1, len(body_ids))
    )
    points.append(positive_endpoints[-1])
    return torch.stack(points)


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


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_root_pose_sync_preserves_authored_scale():
    """Newton body pose synchronization must preserve authored USD scale in Kit/RTX."""
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
            body_path = "/World/envs/env_0/Cube"
            authored_scale = torch.tensor([0.25, 0.5, 0.75])
            body_prim = sim_utils.get_current_stage().GetPrimAtPath(body_path)
            body_prim.GetAttribute("xformOp:scale").Set(UsdGf.Vec3d(*authored_scale.tolist()))

            sim.reset()
            scene.reset()
            sim.render()
            wp.synchronize_device(device)

            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)

            target_pose = torch.tensor(
                [[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            scene["cube"].write_root_link_pose_to_sim_index(root_pose=target_pose)
            sim.render()
            wp.synchronize_device(device)

            torch.testing.assert_close(_fabric_position(body_path), target_pose[0, :3].cpu(), rtol=0.0, atol=1.0e-4)
            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)
        finally:
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_periodic_cable_is_skipped_by_fabric_sync():
    """Periodic cables must not abort unsupported Fabric synchronization."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cuda:0",
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        cable_cfg = _CableRenderSceneCfg(num_envs=1, env_spacing=1.0).cable.spawn
        cable_cfg.func("/World/Cable", cable_cfg)
        curve = UsdGeom.BasisCurves(sim_utils.get_current_stage().GetPrimAtPath("/World/Cable/geometry/mesh"))
        curve.GetWrapAttr().Set(UsdGeom.Tokens.periodic)

        sim.reset()

        assert NewtonManager._cable_shape_ids is None


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_cable_points_follow_newton_segments_after_step_and_reset():
    """Fabric cable points must follow Newton segments across steps and hard resets."""
    device = "cuda:0"
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device=device,
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_CableRenderSceneCfg(num_envs=2, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            scene.update(0.0)
            sim.render()
            wp.synchronize_device(device)

            cable = scene["cable"]
            curve_path = "/World/envs/env_0/Cable/geometry/mesh"
            initial_points = _fabric_curve_points_world(curve_path)
            torch.testing.assert_close(initial_points, _expected_cable_points_world(cable), rtol=0.0, atol=1.0e-4)

            for _ in range(8):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.cfg.dt)
            sim.render()
            wp.synchronize_device(device)
            moved_points = _fabric_curve_points_world(curve_path)
            torch.testing.assert_close(moved_points, _expected_cable_points_world(cable), rtol=0.0, atol=1.0e-4)
            assert not torch.allclose(moved_points, initial_points, rtol=0.0, atol=1.0e-5)
            replicated_curve_path = "/World/envs/env_1/Cable/geometry/mesh"
            torch.testing.assert_close(
                _fabric_curve_points_world(replicated_curve_path),
                _expected_cable_points_world(cable, env_id=1),
                rtol=0.0,
                atol=1.0e-4,
            )

            sim.reset()
            scene.update(0.0)
            sim.render()
            wp.synchronize_device(device)
            reset_points = _fabric_curve_points_world(curve_path)
            torch.testing.assert_close(reset_points, _expected_cable_points_world(cable), rtol=0.0, atol=1.0e-4)

            for _ in range(8):
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.cfg.dt)
            sim.render()
            wp.synchronize_device(device)
            after_reset_points = _fabric_curve_points_world(curve_path)
            torch.testing.assert_close(after_reset_points, _expected_cable_points_world(cable), rtol=0.0, atol=1.0e-4)
            assert not torch.allclose(after_reset_points, reset_points, rtol=0.0, atol=1.0e-5)
        finally:
            sim.register_interactive_scene(None)


"""FrameView (non-physics frame) synchronization.

A :class:`~isaaclab.sim.views.FrameView` prim is not a Newton body, so its *own* pose writes had no
path to Fabric (frames parented to a body already track it via the hierarchy pass). These tests check
the same boundary as the body tests: the Fabric world matrix the renderer consumes.
"""


@contextlib.contextmanager
def _frame_scene(frame_path: str, translation, device: str = "cuda:0"):
    """Yield a reset render scene with a FrameView over a freshly created Xform at ``frame_path``."""
    from isaaclab.sim.views import FrameView

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
            sim_utils.create_prim(frame_path, "Xform", translation=translation)
            view = FrameView(frame_path, device=device)
            sim.reset()
            scene.reset()
            _render(sim, device)
            yield sim, scene, view
        finally:
            sim.register_interactive_scene(None)


def _render(sim, device: str = "cuda:0") -> None:
    """Render and wait for the write to land in Fabric."""
    sim.render()
    wp.synchronize_device(device)


def _write_frame_world_position(view, position: torch.Tensor) -> None:
    """Write a world-space position (identity rotation) through the view's world-space writer."""
    positions = wp.from_torch(position.reshape(1, 3).contiguous(), dtype=wp.vec3f)
    orientations = wp.from_torch(
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=position.device), dtype=wp.vec4f
    )
    with view.xform_world_space_writer() as writer:
        writer.set_poses(positions, orientations)


def _assert_position(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-4)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_frame_view_pose_write_reaches_fabric():
    """A world-attached FrameView pose write must reach the transform Kit/RTX renders.

    Regression for camera poses set via ``Camera.set_world_poses`` under Newton: the write lands in
    the Newton site's local transform and ``get_world_poses`` reflects it, but nothing mirrors it
    onto the prim, so the renderer keeps drawing the frame at its spawn pose.
    """
    device = "cuda:0"
    frame_path = "/World/Frame"
    spawn_position = torch.tensor([0.0, 0.0, 2.0])
    target_position = torch.tensor([1.0, -0.5, 8.0])

    with _frame_scene(frame_path, tuple(spawn_position.tolist()), device) as (sim, _, view):
        _assert_position(_fabric_position(frame_path), spawn_position)

        _write_frame_world_position(view, target_position.to(device))
        _render(sim, device)

        # The view's own report and the rendered transform must agree; only the latter regresses.
        _assert_position(wp.to_torch(view.get_world_poses()[0].warp).cpu()[0], target_position)
        _assert_position(_fabric_position(frame_path), target_position)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_frame_view_pose_write_reaches_fabric_when_the_scope_raises():
    """A write that lands in Newton must still be mirrored when the scope unwinds.

    The Newton-side write is committed as soon as ``set_poses`` returns, so skipping the mirror on
    the exception path would leave the renderer showing the old pose indefinitely.
    """
    device = "cuda:0"
    frame_path = "/World/Frame"
    target_position = torch.tensor([1.0, -0.5, 8.0])

    with _frame_scene(frame_path, (0.0, 0.0, 2.0), device) as (sim, _, view):
        positions = wp.from_torch(target_position.reshape(1, 3).to(device).contiguous(), dtype=wp.vec3f)
        orientations = wp.from_torch(
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device), dtype=wp.vec4f
        )
        with pytest.raises(RuntimeError, match="scope body failed"):  # noqa: PT012 -- the raise is the scenario
            with view.xform_world_space_writer() as writer:
                writer.set_poses(positions, orientations)
                raise RuntimeError("scope body failed")
        _render(sim, device)

        _assert_position(_fabric_position(frame_path), target_position)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_frame_view_pose_write_on_body_child_survives_body_motion():
    """A pose write on a body-attached frame must render, and the frame must still track the body.

    Writing a world pose has to update the frame's transform *relative to its body*, not pin it in
    world space.  A fix that only stamps the world matrix renders correctly once and is then either
    overwritten by the next hierarchy pass or frozen away from the body.
    """
    device = "cuda:0"
    body_path = "/World/envs/env_0/Cube"
    frame_path = f"{body_path}/Frame"

    with _frame_scene(frame_path, (0.0, 0.0, 0.35), device) as (sim, scene, view):
        # Cube spawns at (0, 0, 1); place the frame 0.5 m to its +X side.
        body_start = torch.tensor([0.0, 0.0, 1.0])
        written_position = body_start + torch.tensor([0.5, 0.0, 0.0])
        _write_frame_world_position(view, written_position.to(device))
        _render(sim, device)

        _assert_position(_fabric_position(frame_path), written_position)

        # Move the body; the frame must carry the written offset with it.
        target_pose = torch.tensor([[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        scene["cube"].write_root_link_pose_to_sim_index(root_pose=target_pose)
        _render(sim, device)

        expected = target_pose[0, :3].cpu() + (written_position - body_start)
        _assert_position(wp.to_torch(view.get_world_poses()[0].warp).cpu()[0], expected)
        _assert_position(_fabric_position(frame_path), expected)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_first_frame_pose_write_after_body_move_leaves_the_body_rendered():
    """Building the mirror must not reseed its prims from USD.

    The mirror tags the frame *and its parent body*, and the body's Fabric transform is driven by
    Newton, which never writes back to USD. Seeding it from USD when the mirror is built would snap
    the rendered body back to its spawn pose -- and the body sync is a no-op at that point, so
    nothing would put it back. Ordering matters: the body has to move and render *before* the first
    frame write, which is what builds the mirror.
    """
    device = "cuda:0"
    body_path = "/World/envs/env_0/Cube"
    frame_path = f"{body_path}/Frame"

    with _frame_scene(frame_path, (0.0, 0.0, 0.35), device) as (sim, scene, view):
        body_start = torch.tensor([0.0, 0.0, 1.0])
        body_target = torch.tensor([1.5, -0.75, 2.0])
        target_pose = torch.zeros((1, 7), dtype=torch.float32, device=device)
        target_pose[0, :3] = body_target.to(device)
        target_pose[0, 6] = 1.0

        # Move and render the body first, so its Fabric transform is current and USD is stale.
        scene["cube"].write_root_link_pose_to_sim_index(root_pose=target_pose)
        _render(sim, device)
        _assert_position(_fabric_position(body_path), body_target)

        # First frame write: this is what builds the mirror.
        written_position = body_target + torch.tensor([0.5, 0.0, 0.0])
        _write_frame_world_position(view, written_position.to(device))
        _render(sim, device)

        _assert_position(_fabric_position(body_path), body_target)
        _assert_position(_fabric_position(frame_path), written_position)
        assert not torch.allclose(_fabric_position(body_path), body_start, rtol=0.0, atol=1.0e-4)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_frame_view_pose_write_after_unrendered_steps_reaches_fabric():
    """A pose write must render correctly even when the body moved since the last render.

    Bodies sync to Fabric only at render cadence, so after ``sim.step(render=False)`` the parent's
    Fabric matrix lags Newton. Deriving the frame's local matrix from that lagging parent displaces
    the rendered frame by exactly the parent's motion once the next hierarchy pass runs.
    """
    device = "cuda:0"
    body_path = "/World/envs/env_0/Cube"
    frame_path = f"{body_path}/Frame"

    with _frame_scene(frame_path, (0.0, 0.0, 0.35), device) as (sim, scene, view):
        # Move the body through physics without rendering, leaving its Fabric matrix at the old pose.
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
        velocity[0, 0] = 5.0
        scene["cube"].write_root_com_velocity_to_sim_index(root_velocity=velocity)
        for _ in range(30):
            sim.step(render=False)
        assert _fabric_position(body_path)[0].item() == pytest.approx(0.0, abs=1.0e-4)

        target_position = torch.tensor([0.0, 0.0, 1.5])
        _write_frame_world_position(view, target_position.to(device))
        _render(sim, device)

        _assert_position(wp.to_torch(view.get_world_poses()[0].warp).cpu()[0], target_position)
        _assert_position(_fabric_position(frame_path), target_position)
