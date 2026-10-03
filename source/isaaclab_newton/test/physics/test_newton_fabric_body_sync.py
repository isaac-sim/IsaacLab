# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton USD/Fabric body synchronization."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock

# Launch Isaac Sim before importing Newton modules so USD schema bindings are initialized.
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(enable_cameras=True)

import omni.kit.app

simulation_app = omni.kit.app.get_app()

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonManager,
    VBDSolverCfg,
    XPBDSolverCfg,
)
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_physx.renderers.fabric import FabricBackend, FabricBackendCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from pxr import Gf as UsdGf
from pxr import Sdf, UsdGeom, UsdPhysics
from usdrt import Gf, Rt, Vt
from usdrt import Sdf as RtSdf

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.app.settings_manager import get_settings_manager
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, CableObjectCfg, RigidObjectCfg
from isaaclab.envs.utils.video_recorder import VideoRecorder
from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.scene_data import SceneDataFormat, SceneDataProvider
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg
from isaaclab.utils import configclass, replace
from isaaclab.utils import math as math_utils


@configclass
class _RenderSceneCfg(InteractiveSceneCfg):
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        height=16,
        width=16,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(),
        renderer_cfg=IsaacRtxRendererCfg(),
    )
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=PhysxRigidBodyCfg(disable_gravity=True),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class _CableRenderSceneCfg(InteractiveSceneCfg):
    camera: CameraCfg = _RenderSceneCfg().camera
    cube: RigidObjectCfg = replace(
        _RenderSceneCfg().cube, init_state=RigidObjectCfg.InitialStateCfg(pos=(1.0, 0.0, 1.0))
    )
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


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
@pytest.mark.parametrize("capture_method", ["render", "video_recorder"])
def test_root_pose_write_is_visible_on_next_render_without_step(capture_method, monkeypatch, tmp_path):
    """A reset-time pose write must reach Kit/RTX on the next render.

    This reproduces the application sequence that used to render one-frame-old
    transforms: write asset state, then render without an intervening physics
    step or asset-data read. The assertion reads Fabric's world matrix, which is
    the transform consumed by Kit/RTX; USD is intentionally not written back.
    The synchronized pose must also keep the body's authored USD scale.
    """
    device = "cuda:0"
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
        visualizer_cfgs=[KitVisualizerCfg(headless=True, window_width=64, window_height=64)],
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(_RenderSceneCfg(num_envs=1, env_spacing=2.0))
        sim.register_interactive_scene(scene)
        previous_rtx_sensors = sim.get_setting("/isaaclab/render/rtx_sensors")
        try:
            body_path = "/World/envs/env_0/Cube"
            authored_scale = torch.tensor([0.25, 0.5, 0.75])
            body_prim = sim_utils.get_current_stage().GetPrimAtPath(body_path)
            body_prim.GetAttribute("xformOp:scale").Set(UsdGf.Vec3d(*authored_scale.tolist()))

            sim.reset()
            scene.reset()
            _render(sim, scene)
            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)

            fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
            assert sim.visualizers[0]._fabric is scene["camera"]._renderer._fabric is fabric
            assert sum(isinstance(resource, FabricBackend) for _, resource in sim._backend_registry) == 1

            recorder = None
            if capture_method == "video_recorder":
                # The suite enables cameras at startup; this case exercises capture-only rendering.
                sim.set_setting("/isaaclab/render/rtx_sensors", False)
                assert not sim.is_rendering
                monkeypatch.setattr("isaaclab.envs.utils.video_recorder.ImageSequenceClip", Mock())
                recorder = VideoRecorder(
                    VideoRecorderCfg(source="viz:kit", output_dir=str(tmp_path)), SimpleNamespace(sim=sim)
                )

            def capture_frame() -> None:
                if recorder is None:
                    _render(sim, scene)
                else:
                    assert recorder._get_frame().shape == (64, 64, 3)
                    wp.synchronize_device(device)

            cube = scene["cube"]
            target_pose = torch.tensor(
                [[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )

            cube.write_root_link_pose_to_sim_index(root_pose=target_pose)

            physics_steps = sim.get_physics_step_count()
            capture_frame()

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
            capture_frame()

            torch.cuda.synchronize(device)
            with wp.ScopedCapture(device=device) as capture:
                cube.write_root_link_pose_to_sim_mask(root_pose=pose_buffer, env_mask=env_mask)

            capture_frame()

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
                capture_frame()

                assert sim.get_physics_step_count() == physics_steps
                torch.testing.assert_close(
                    _fabric_position(body_path),
                    replay_target[:3].cpu(),
                    rtol=0.0,
                    atol=1.0e-4,
                )
            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)
        finally:
            sim.set_setting("/isaaclab/render/rtx_sensors", previous_rtx_sensors)
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
@pytest.mark.parametrize(
    ("device", "renderer_cfg"),
    [("cpu", IsaacRtxRendererCfg()), ("cuda:0", NewtonWarpRendererCfg())],
    ids=["rtx-cpu", "newton-warp"],
)
def test_root_pose_sync_preserves_authored_scale(device, renderer_cfg):
    """Newton body pose synchronization must preserve authored USD scale in Kit/RTX."""
    sim_cfg = SimulationCfg(
        device=device,
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene_cfg = _RenderSceneCfg(num_envs=1, env_spacing=2.0)
        scene_cfg.camera.renderer_cfg = renderer_cfg
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        try:
            body_path = "/World/envs/env_0/Cube"
            authored_scale = torch.tensor([0.25, 0.5, 0.75])
            body_prim = sim_utils.get_current_stage().GetPrimAtPath(body_path)
            body_prim.GetAttribute("xformOp:scale").Set(UsdGf.Vec3d(*authored_scale.tolist()))

            sim.reset()
            scene.reset()
            fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
            fabric.bind_transforms(sim.get_scene_data_provider())
            _render(sim, scene)

            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)

            target_pose = torch.tensor(
                [[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]],
                dtype=torch.float32,
                device=device,
            )
            scene["cube"].write_root_link_pose_to_sim_index(root_pose=target_pose)
            if isinstance(renderer_cfg, NewtonWarpRendererCfg):
                assert not sim.visualizers
                fabric.update_transforms(sim.get_scene_data_provider())
            _render(sim, scene)

            torch.testing.assert_close(_fabric_position(body_path), target_pose[0, :3].cpu(), rtol=0.0, atol=1.0e-4)
            torch.testing.assert_close(_fabric_scale(body_path), authored_scale, rtol=0.0, atol=1.0e-5)
        finally:
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_nested_bodies_keep_independent_world_poses():
    """A nested rigid body must not inherit its parent's independently published motion."""
    sim_cfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=XPBDSolverCfg(), use_cuda_graph=False),
    )
    scene_cfg = _RenderSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.child = replace(scene_cfg.cube, prim_path="{ENV_REGEX_NS}/Cube/Child")
    scene_cfg.cube = AssetBaseCfg(prim_path=scene_cfg.cube.prim_path, spawn=scene_cfg.cube.spawn)
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            _render(sim, scene)
            paths = ["/World/envs/env_0/Cube", "/World/envs/env_0/Cube/Child"]
            targets = torch.tensor([[1.5, -0.75, 2.0], [-0.25, 1.0, 3.0]], device=sim.device)
            state = wp.to_torch(NewtonManager.get_state_0().body_q)
            indices = [NewtonManager.get_model().body_label.index(path) for path in paths]
            state[indices, :3] = targets
            NewtonManager.invalidate_body_state()
            _render(sim, scene)
            for path, target in zip(paths, targets.cpu()):
                _assert_position(_fabric_position(path), target)
                assert not UsdGeom.Xformable(sim.stage.GetPrimAtPath(path)).GetResetXformStack()
        finally:
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_nested_articulation_follows_root_pose_writes():
    """Rendering a nested child link must apply its parent's root displacement only once."""
    sim_cfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MJWarpSolverCfg(), use_cuda_graph=False),
    )
    scene_cfg = _RenderSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.cube = None
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=None,
        actuators={"all": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        root_path = "/World/envs/env_0/Robot"
        root = UsdGeom.Xform.Define(sim.stage, root_path)
        UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
        paths = [f"{root_path}/base", f"{root_path}/base/arm"]
        for path, position in zip(paths, ((0.0, 0.0, 1.0), (0.0, 0.5, 0.0)), strict=True):
            body = UsdGeom.Xform.Define(sim.stage, path)
            body.AddTranslateOp().Set(UsdGf.Vec3d(*position))
            UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
            UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
            geometry = UsdGeom.Cube.Define(sim.stage, f"{path}/geometry")
            geometry.GetSizeAttr().Set(0.2)
            UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
        joint = UsdPhysics.RevoluteJoint.Define(sim.stage, f"{root_path}/base/arm_joint")
        joint.CreateBody0Rel().SetTargets([paths[0]])
        joint.CreateBody1Rel().SetTargets([paths[1]])
        joint.CreateAxisAttr("X")
        joint.CreateLocalPos0Attr(UsdGf.Vec3f(0.0, 0.5, 0.0))
        joint.CreateLocalPos1Attr(UsdGf.Vec3f(0.0))

        scene = InteractiveScene(scene_cfg)
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()
            _render(sim, scene)
            indices = [NewtonManager.get_model().body_label.index(path) for path in paths]
            pose = scene["robot"].data.default_root_pose.torch.clone()
            physics_steps = sim.get_physics_step_count()
            for displacement in (1.0, 2.0, 3.0):
                pose[:, 0] = displacement
                scene["robot"].write_root_link_pose_to_sim_index(root_pose=pose)
                _render(sim, scene)
                expected = wp.to_torch(NewtonManager.get_state_0().body_q)[indices, :3].cpu()
                for path, position in zip(paths, expected, strict=True):
                    _assert_position(_fabric_position(path), position)
                    _assert_position(_fabric_position(f"{path}/geometry"), position)
                assert sim.get_physics_step_count() == physics_steps
        finally:
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
@pytest.mark.parametrize("point_path", ["gpu", "legacy-kit", "streaming"])
def test_fabric_geometry_sink_uses_sdp_world_points_and_frame_cadence(
    monkeypatch, particle_render_settings, point_path
):
    """Mesh, curve and cloud sinks consume only named SDP buffers, independent of Newton internals."""
    kit_version = tuple(map(int, simulation_app.get_kit_version_short().split(".")))
    gpu_points = point_path == "gpu" and kit_version >= (110, 4)
    particle_render_settings.set("/UJITSO/geometryTypes", "Points" if point_path == "streaming" else "Mesh")
    if point_path == "legacy-kit":
        app = Mock(wraps=simulation_app)
        app.get_kit_version_short.return_value = "110.3"
        monkeypatch.setattr(omni.kit.app, "get_app", lambda: app)
    cfg = SimulationCfg(device="cuda:0", physics=NewtonCfg(), visualizer_cfgs=[])
    with build_simulation_context(sim_cfg=cfg) as sim:
        parent = UsdGeom.Xform.Define(sim.stage, "/World/Geometry")
        parent.AddTranslateOp().Set(UsdGf.Vec3d(10.0, 20.0, 30.0))
        points = {}
        for schema, name in ((UsdGeom.Mesh, "Mesh"), (UsdGeom.BasisCurves, "Curve"), (UsdGeom.Points, "Cloud")):
            path = f"/World/Geometry/{name}"
            geometry = schema.Define(sim.stage, path)
            geometry.CreatePointsAttr([(0.0, 0.0, 0.0)] * 3)
            geometry.AddTranslateOp().Set(UsdGf.Vec3d(1.0, 2.0, 3.0))
            if name == "Mesh":
                geometry.CreateFaceVertexCountsAttr([3])
                geometry.CreateFaceVertexIndicesAttr([0, 1, 2])
            elif name == "Curve":
                geometry.CreateCurveVertexCountsAttr([3])
                geometry.CreateTypeAttr("linear")
                geometry.CreateWrapAttr("nonperiodic")
            else:
                geometry.GetPrim().CreateAttribute("isaaclab:pointsUpdateFrequency", Sdf.ValueTypeNames.Int).Set(3)
            points[path] = wp.array([[1.0, 0.0, 2.0], [2.0, 0.0, 2.0], [3.0, 0.0, 2.0]], wp.vec3f, device=sim.device)
        batches = []
        for path, values in points.items():
            source = SceneDataFormat.Points()
            source.points = values
            batches.append((source, {path: (0, len(values))}))
        provider = SceneDataProvider(
            SimpleNamespace(
                native_transform_formats=(),
                geometry_timestamp=0,
                get_geometry_batches=lambda _format=SceneDataFormat.Points: batches,
            )
        )
        fabric = sim.get_or_create_backend(FabricBackendCfg(stage=sim.stage, device=sim.device))
        fabric.update_geometries(provider, 0)
        simulation_app.update()
        wp.synchronize_device(sim.device)
        for path in points:
            np.testing.assert_allclose(
                _fabric_curve_points_world(path), points[path].numpy(), atol=1.0e-6, err_msg=path
            )
        # Kit's first update can add prims; settle those structural changes before checking cadence.
        fabric.update_geometries(provider, 0)

        # Replace pointers in the same published mapping; parents must never be applied a second time.
        for (source, _), path in zip(batches, points, strict=True):
            points[path] = wp.array([[4.0, 5.0, 6.0]] * 3, wp.vec3f, device=sim.device)
            source.points = points[path]
        provider.backend.geometry_timestamp += 1
        fabric.update_geometries(provider, 0)
        wp.synchronize_device(sim.device)
        for name in ("Mesh", "Curve"):
            path = f"/World/Geometry/{name}"
            np.testing.assert_allclose(_fabric_curve_points_world(path), points[path].numpy(), atol=1.0e-6)
        cloud = UsdGeom.Points(sim.stage.GetPrimAtPath("/World/Geometry/Cloud"))
        fabric.update_geometries(provider, 1)
        fabric.update_geometries(provider, 2)
        np.testing.assert_allclose(_fabric_curve_points_world(str(cloud.GetPath()))[0], [1.0, 0.0, 2.0])
        copy = wp.copy

        def copy_without_point_readback(dest, src, *args, **kwargs):
            if gpu_points:
                assert not (src.device.is_cuda and dest.device.is_cpu), "Particle rendering copied points to CPU"
            return copy(dest, src, *args, **kwargs)

        # Only the cloud is due here; a CUDA cloud must enter Fabric without a host copy.
        with monkeypatch.context() as patch:
            patch.setattr(wp, "copy", copy_without_point_readback)
            fabric.update_geometries(provider, 3)
        np.testing.assert_allclose(
            _fabric_curve_points_world(str(cloud.GetPath())), points["/World/Geometry/Cloud"].numpy(), atol=1.0e-6
        )
        # Runtime transport must not rewrite the authored USD points.
        np.testing.assert_array_equal(cloud.GetPointsAttr().Get(), np.zeros((3, 3)))

        # Moving a prim to a new Fabric bucket must refresh the sink even with unchanged physics.
        mesh_path = "/World/Geometry/Mesh"
        mesh = fabric.stage.GetPrimAtPath(mesh_path)
        mesh.CreateAttribute("test:geometryBucket", RtSdf.ValueTypeNames.Bool, custom=True).Set(True)
        mesh.GetAttribute("points").Set(Vt.Vec3fArray([Gf.Vec3f(99.0)] * 3))
        with monkeypatch.context() as patch:
            patch.setattr(provider, "get_geometry_points", Mock(side_effect=RuntimeError("conversion failed")))
            with pytest.raises(RuntimeError, match="conversion failed"):
                fabric.update_geometries(provider, 3)
        fabric.update_geometries(provider, 3)
        wp.synchronize_device(sim.device)
        np.testing.assert_allclose(_fabric_curve_points_world(mesh_path), points[mesh_path].numpy(), atol=1.0e-6)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
@pytest.mark.parametrize("point_path", ["gpu", "streaming"])
def test_mpm_particle_writes_and_resets_are_visible_in_rtx(particle_render_settings, point_path):
    """RTX must render particle writes and resets independently for cloned MPM assets."""
    particle_render_settings.set("/UJITSO/geometryTypes", "Points" if point_path == "streaming" else "Mesh")

    @configclass
    class ParticleSceneCfg(InteractiveSceneCfg):
        red = MPMObjectCfg(
            prim_path="{ENV_REGEX_NS}/Red",
            spawn=MPMGridCfg(
                lower=(-0.1, -0.1, -0.1),
                upper=(0.1, 0.1, 0.1),
                voxel_size=0.2,
                particle_placement="cell_center",
                visual_color=(1.0, 0.0, 0.0),
            ),
            init_state=MPMObjectCfg.InitialStateCfg(pos=(-0.4, 0.0, 1.0)),
        )
        green = replace(red, prim_path="{ENV_REGEX_NS}/Green")
        green.spawn.visual_color = (0.0, 1.0, 0.0)
        green.init_state.pos = (0.4, 0.0, 1.0)
        camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Camera",
            height=128,
            width=128,
            data_types=["rgb"],
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), convention="opengl"),
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=20.0),
            renderer_cfg=IsaacRtxRendererCfg(),
        )
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

    cfg = SimulationCfg(
        device="cuda:0",
        gravity=(0.0, 0.0, 0.0),
        physics=NewtonCfg(solver_cfg=MPMSolverCfg(max_iterations=2, voxel_size=0.2), use_cuda_graph=False),
    )
    with build_simulation_context(sim_cfg=cfg) as sim:
        scene = InteractiveScene(ParticleSceneCfg(num_envs=2, env_spacing=4.0))
        sim.register_interactive_scene(scene)
        try:
            sim.reset()
            scene.reset()

            def centers():
                for _ in range(4):
                    _render(sim, scene)
                rgb = scene["camera"].data.output["rgb"].torch[..., :3].to(torch.float32)
                values = []
                for image in rgb:
                    for channel in (0, 1):
                        others = image[..., [value for value in range(3) if value != channel]].amax(-1)
                        pixels = torch.nonzero((image[..., channel] > 40) & (image[..., channel] > others * 1.5))
                        assert len(pixels) >= 5, "The colored particle cloud is absent from RTX"
                        values.append(pixels.to(torch.float32).mean(0))
                return torch.stack(values)

            before = centers()
            state = scene["red"].data.default_particle_state_w.torch[1:2].clone()
            state[..., 1] += 0.4
            scene["red"].write_particle_state_to_sim_index(state, env_ids=[1])
            moved = centers()
            assert torch.linalg.vector_norm(moved[2] - before[2]) > 10
            torch.testing.assert_close(moved[[0, 1, 3]], before[[0, 1, 3]], rtol=0.0, atol=2.0)
            scene["red"].reset(env_ids=[1])
            torch.testing.assert_close(centers(), before, rtol=0.0, atol=2.0)
        finally:
            sim.register_interactive_scene(None)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_cable_points_follow_newton_segments_after_step_and_reset():
    """Cable segments and neighboring rigid bodies must render across steps and hard resets."""
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
            cable = scene["cable"]
            paths = [f"{path}/Cable/geometry/mesh" for path in scene.env_prim_paths]
            for reset_scene in (True, False):
                sim.reset()
                if reset_scene:
                    scene.reset()
                scene.update(0.0)
                _render(sim, scene)
                initial_points = [_fabric_curve_points_world(path) for path in paths]
                for env_id, initial in enumerate(initial_points):
                    expected = _expected_cable_points_world(cable, env_id)
                    torch.testing.assert_close(initial, expected, rtol=0.0, atol=1.0e-4)
                cube_pose = scene["cube"].data.default_root_pose.torch.clone()
                cube_pose[:, :3] = scene.env_origins + torch.tensor([1.5, -0.75, 2.0], device=device)
                scene["cube"].write_root_link_pose_to_sim_index(root_pose=cube_pose)
                _render(sim, scene)
                for env_path, pose in zip(scene.env_prim_paths, cube_pose.cpu(), strict=True):
                    _assert_position(_fabric_position(f"{env_path}/Cube"), pose[:3])
                for _ in range(8):
                    scene.write_data_to_sim()
                    sim.step(render=False)
                    scene.update(sim.cfg.dt)
                _render(sim, scene)
                for env_id, path in enumerate(paths):
                    moved = _fabric_curve_points_world(path)
                    expected = _expected_cable_points_world(cable, env_id)
                    torch.testing.assert_close(moved, expected, rtol=0.0, atol=1.0e-4)
                    assert not torch.allclose(moved, initial_points[env_id], rtol=0.0, atol=1.0e-5)
        finally:
            sim.register_interactive_scene(None)


@contextlib.contextmanager
def _frame_scene(frame_path: str, translation, device: str = "cuda:0"):
    """Yield a reset render scene with a FrameView over a new Xform at ``frame_path``."""
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
            _render(sim, scene)
            yield sim, scene, view
        finally:
            sim.register_interactive_scene(None)


@pytest.fixture
def particle_render_settings():
    """Restore the process-wide geometry settings after each particle render case."""
    settings = get_settings_manager()
    original = {path: settings.get(path) for path in ("/UJITSO/geometry", "/UJITSO/geometryTypes")}
    settings.set("/UJITSO/geometry", False)
    try:
        yield settings
    finally:
        for path, value in original.items():
            settings.set(path, value)


def _render(sim, scene) -> None:
    """Render through the camera's public data path and wait for Fabric writes."""
    sim.render()
    scene["camera"].update(sim.get_rendering_dt(), force_recompute=True)
    wp.synchronize_device(sim.device)


def _world_pose(position: torch.Tensor) -> tuple[wp.array, wp.array]:
    """Return ``(positions, orientations)`` writer arguments for ``position`` with identity rotation."""
    return (
        wp.from_torch(position.reshape(1, 3).contiguous(), dtype=wp.vec3f),
        wp.from_torch(
            torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=position.device), dtype=wp.vec4f
        ),
    )


def _write_frame_world_position(view, position: torch.Tensor) -> None:
    """Write a world-space position through the view's world-space writer."""
    with view.xform_world_space_writer() as writer:
        writer.set_poses(*_world_pose(position))


def _reported_position(view) -> torch.Tensor:
    """Read the view's own world position, as opposed to the one Fabric renders."""
    return wp.to_torch(view.get_world_poses()[0].warp).cpu()[0]


def _assert_position(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-4)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_frame_view_pose_write_reaches_fabric_when_the_scope_raises():
    """A pose write already committed to Newton is mirrored even when the scope unwinds."""
    device = "cuda:0"
    frame_path = "/World/Frame"
    target_position = torch.tensor([1.0, -0.5, 8.0])

    with _frame_scene(frame_path, (0.0, 0.0, 2.0), device) as (sim, scene, view):
        with pytest.raises(RuntimeError, match="boom"):  # noqa: PT012 -- the raise is the scenario
            with view.xform_world_space_writer() as writer:
                writer.set_poses(*_world_pose(target_position.to(device)))
                raise RuntimeError("boom")
        _render(sim, scene)

        _assert_position(_reported_position(view), target_position)
        _assert_position(_fabric_position(frame_path), target_position)


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_first_frame_pose_write_after_body_move_leaves_the_body_rendered():
    """A body-attached frame renders at written poses while the body moves.

    Building the mirror on the first frame write must not reseed its prims from USD, which would
    unrender the moved body. The written frame then keeps tracking later body writes, and a frame
    write after unrendered physics steps composes against the live body pose.
    """
    device = "cuda:0"
    body_path = "/World/envs/env_0/Cube"
    frame_path = f"{body_path}/Frame"

    with _frame_scene(frame_path, (0.0, 0.0, 0.35), device) as (sim, scene, view):
        body_pose = torch.tensor([[1.5, -0.75, 2.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        body_target = body_pose[0, :3].cpu()
        scene["cube"].write_root_link_pose_to_sim_index(root_pose=body_pose)
        _render(sim, scene)
        _assert_position(_fabric_position(body_path), body_target)

        # The first frame write must be the one that builds the mirror.
        late_child = f"{frame_path}/LateChild"
        offset = torch.tensor([0.0, 0.0, 0.25])
        sim_utils.create_prim(late_child, "Xform", translation=tuple(offset.tolist()))
        written_position = body_target + torch.tensor([0.5, 0.0, 0.0])
        _write_frame_world_position(view, written_position.to(device))
        _render(sim, scene)

        _assert_position(_fabric_position(body_path), body_target)
        _assert_position(_fabric_position(frame_path), written_position)
        _assert_position(_fabric_position(late_child), written_position + offset)

        # The written frame keeps tracking its body.
        moved_pose = torch.tensor([[-1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
        scene["cube"].write_root_link_pose_to_sim_index(root_pose=moved_pose)
        _render(sim, scene)

        expected = moved_pose[0, :3].cpu() + (written_position - body_target)
        _assert_position(_reported_position(view), expected)
        _assert_position(_fabric_position(frame_path), expected)

        # Move the body without rendering, then write the frame.
        velocity = torch.zeros((1, 6), dtype=torch.float32, device=device)
        velocity[0, 0] = 5.0
        scene["cube"].write_root_com_velocity_to_sim_index(root_velocity=velocity)
        for _ in range(30):
            sim.step(render=False)
        _assert_position(_fabric_position(body_path), moved_pose[0, :3].cpu())

        target_position = torch.tensor([0.0, 0.0, 1.5])
        _write_frame_world_position(view, target_position.to(device))
        _render(sim, scene)

        _assert_position(_reported_position(view), target_position)
        _assert_position(_fabric_position(frame_path), target_position)
