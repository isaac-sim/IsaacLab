# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gc
import weakref
from types import SimpleNamespace

import pytest
import torch
import warp as wp

pytest.importorskip("newton")

from isaaclab_newton.assets import CableObject as NewtonCableObject
from isaaclab_newton.physics import NewtonCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

from pxr import UsdGeom, Vt

from isaaclab.assets import CableObjectCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.deformable import VBDSolverCfg


@configclass
class _CableSceneCfg(InteractiveSceneCfg):
    cable = CableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cable",
        spawn=CableCfg(
            positions=((0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0), (0.0, 0.6, 1.0)),
            physics_material=CableMaterialCfg(
                thickness=0.02,
                density=500.0,
                stretch_stiffness=1.0e5,
                bend_stiffness=1.0e3,
            ),
        ),
    )


def _author_curve(stage, path, points, counts) -> None:
    curves = UsdGeom.BasisCurves.Define(stage, path)
    curves.CreatePointsAttr(points)
    curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")


def _expected_segment_state(cable, state, model) -> tuple[torch.Tensor, torch.Tensor]:
    root_body_ids = wp.to_torch(cable.root_view.get_attribute("joint_parent", model)[:, 0, 0]).long()
    root_pose = wp.to_torch(state.body_q)[root_body_ids].unsqueeze(1)
    root_velocity = wp.to_torch(state.body_qd)[root_body_ids].unsqueeze(1)
    link_pose = wp.to_torch(cable.root_view.get_link_transforms(state)[:, 0])
    link_velocity = wp.to_torch(cable.root_view.get_link_velocities(state)[:, 0])
    return torch.cat((root_pose, link_pose), dim=1), torch.cat((root_velocity, link_velocity), dim=1)


def test_interactive_scene_manages_newton_cables():
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_CableSceneCfg(num_envs=2, env_spacing=1.0))
        sim.reset()
        scene.update(0.0)

        cable = scene["cable"]
        assert isinstance(cable, NewtonCableObject)
        assert scene.cable_objects["cable"] is cable
        assert "cable" in scene.keys()
        assert cable.num_instances == 2
        assert cable.num_segments == 3
        assert cable.data.segment_pose_w.torch.shape == (2, 3, 7)
        assert cable.data.segment_velocity_w.torch.shape == (2, 3, 6)

        model = SimulationManager.get_model()
        state = SimulationManager.get_state_0()
        expected_pose, expected_velocity = _expected_segment_state(cable, state, model)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)
        assert len(model.body_color_groups) > 0

        default_pose = cable.data.default_segment_pose_w.torch
        default_velocity = cable.data.default_segment_velocity_w.torch
        torch.testing.assert_close(cable.data.segment_pose_w.torch, default_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, default_velocity)
        scene_state = scene.get_state(is_relative=True)
        assert scene_state["cable_object"]["cable"]["segment_pose"].shape == (2, 3, 7)

        initial_pose = cable.data.segment_pose_w.torch.clone()
        scene.reset()
        for _ in range(3):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.cfg.dt)

        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()
        assert torch.any(cable.data.segment_pose_w.torch[..., 2] < initial_pose[..., 2])
        expected_pose, expected_velocity = _expected_segment_state(
            cable, SimulationManager.get_state_0(), SimulationManager.get_model()
        )
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)

        scene.reset_to(scene_state, is_relative=True)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, default_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, default_velocity)

        for _ in range(3):
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim.cfg.dt)
        assert torch.any(cable.data.segment_pose_w.torch[..., 2] < default_pose[..., 2])
        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()

        state_0_pose, state_0_velocity = _expected_segment_state(
            cable, SimulationManager.get_state_0(), SimulationManager.get_model()
        )
        state_1_pose, state_1_velocity = _expected_segment_state(
            cable, SimulationManager.get_state_1(), SimulationManager.get_model()
        )
        env_ids = torch.tensor([0], device=sim.device, dtype=torch.long)
        reset_scene_to_default(SimpleNamespace(scene=scene), env_ids)

        torch.testing.assert_close(cable.data.segment_pose_w.torch[0], default_pose[0])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[0], default_velocity[0])
        torch.testing.assert_close(cable.data.segment_pose_w.torch[1], state_0_pose[1])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch[1], state_0_velocity[1])
        for reset_state, other_pose, other_velocity in (
            (SimulationManager.get_state_0(), state_0_pose, state_0_velocity),
            (SimulationManager.get_state_1(), state_1_pose, state_1_velocity),
        ):
            reset_pose, reset_velocity = _expected_segment_state(cable, reset_state, model)
            torch.testing.assert_close(reset_pose[0], default_pose[0])
            torch.testing.assert_close(reset_velocity[0], default_velocity[0])
            torch.testing.assert_close(reset_pose[1], other_pose[1])
            torch.testing.assert_close(reset_velocity[1], other_velocity[1])
        assert wp.to_torch(SimulationManager._world_reset_mask).tolist() == [True, False]
        assert not wp.to_torch(SimulationManager._fk_reset_mask).any()

        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.cfg.dt)
        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()

        previous_state = state
        sim.reset()
        scene.update(0.0)
        state = SimulationManager.get_state_0()
        assert state is not previous_state

        expected_pose, expected_velocity = _expected_segment_state(cable, state, SimulationManager.get_model())
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)

        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(sim.cfg.dt)
        assert torch.isfinite(cable.data.segment_pose_w.torch).all()
        assert torch.isfinite(cable.data.segment_velocity_w.torch).all()

        model = SimulationManager.get_model()
        root_view = cable.root_view
        root_body_id = int(cable.data._sim_bind_root_body_ids.numpy()[0])
        root_label = model.body_label[root_body_id]
        model.body_label[root_body_id] = f"{root_label}_invalid"
        try:
            with pytest.raises(RuntimeError, match="standalone, unwelded"):
                cable._initialize_impl()
        finally:
            model.body_label[root_body_id] = root_label
            cable._root_view = root_view


def test_cable_mask_writes_update_selected_environments():
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_CableSceneCfg(num_envs=3, env_spacing=1.0))
        sim.reset()
        SimulationManager.forward()
        scene.update(0.0)
        cable = scene["cable"]

        original_pose = cable.data.segment_pose_w.torch.clone()
        original_velocity = cable.data.segment_velocity_w.torch.clone()
        target_pose = original_pose.clone()
        target_velocity = original_velocity.clone()
        offsets = torch.tensor([0.1, 0.2, 0.3], device=sim.device).unsqueeze(1)
        target_pose[..., 0] += offsets
        target_velocity[..., 0] += 10.0 * offsets
        env_mask = wp.array([True, False, True], dtype=wp.bool, device=sim.device)

        cable.write_segment_pose_to_sim_mask(segment_pose=target_pose, env_mask=env_mask)
        cable.write_segment_velocity_to_sim_mask(segment_velocity=target_velocity, env_mask=env_mask)

        expected_pose = target_pose.clone()
        expected_pose[1] = original_pose[1]
        expected_velocity = target_velocity.clone()
        expected_velocity[1] = original_velocity[1]
        for state in (SimulationManager.get_state_0(), SimulationManager.get_state_1()):
            state_pose, state_velocity = _expected_segment_state(cable, state, SimulationManager.get_model())
            torch.testing.assert_close(state_pose, expected_pose)
            torch.testing.assert_close(state_velocity, expected_velocity)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)
        assert torch.equal(cable.data.segment_pose_w.torch[1], original_pose[1])
        assert torch.equal(cable.data.segment_velocity_w.torch[1], original_velocity[1])
        assert wp.to_torch(SimulationManager._world_reset_mask).tolist() == [True, False, True]
        other_env_mask = wp.array([False, True, False], dtype=wp.bool, device=sim.device)
        SimulationManager.invalidate_body_state(env_mask=other_env_mask)
        assert wp.to_torch(SimulationManager._world_reset_mask).tolist() == [True, True, True]
        assert not wp.to_torch(SimulationManager._fk_reset_mask).any()


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
def test_cable_mask_writes_are_cuda_graph_capturable(device):
    sim_cfg = SimulationCfg(
        device=device,
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_CableSceneCfg(num_envs=3, env_spacing=1.0))
        sim.reset()
        scene.update(0.0)
        cable = scene["cable"]

        original_pose = cable.data.segment_pose_w.torch.clone()
        original_velocity = cable.data.segment_velocity_w.torch.clone()
        pose_buffer = original_pose.clone()
        velocity_buffer = original_velocity.clone()
        env_mask = wp.array([True, False, True], dtype=wp.bool, device=device)

        cable.write_segment_pose_to_sim_mask(segment_pose=pose_buffer, env_mask=env_mask)
        cable.write_segment_velocity_to_sim_mask(segment_velocity=velocity_buffer, env_mask=env_mask)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            cable.write_segment_pose_to_sim_mask(segment_pose=pose_buffer, env_mask=env_mask)
            cable.write_segment_velocity_to_sim_mask(segment_velocity=velocity_buffer, env_mask=env_mask)

        SimulationManager.forward()
        offsets = torch.tensor([0.4, 0.5, 0.6], device=device).unsqueeze(1)
        pose_buffer[..., 1] += offsets
        velocity_buffer[..., 1] += 10.0 * offsets
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)

        expected_pose = pose_buffer.clone()
        expected_pose[1] = original_pose[1]
        expected_velocity = velocity_buffer.clone()
        expected_velocity[1] = original_velocity[1]
        for state in (SimulationManager.get_state_0(), SimulationManager.get_state_1()):
            state_pose, state_velocity = _expected_segment_state(cable, state, SimulationManager.get_model())
            torch.testing.assert_close(state_pose, expected_pose)
            torch.testing.assert_close(state_velocity, expected_velocity)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, expected_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, expected_velocity)
        assert wp.to_torch(SimulationManager._world_reset_mask).tolist() == [True, False, True]
        assert not wp.to_torch(SimulationManager._fk_reset_mask).any()


def test_cable_rejects_non_vbd_solver():
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu",
        physics=NewtonCfg(use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg):
        with pytest.raises(RuntimeError, match="VBDSolverCfg"):
            InteractiveScene(_CableSceneCfg(num_envs=1, env_spacing=1.0))


def test_cable_rejects_external_rigid_integration():
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(integrate_with_external_rigid_solver=True),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg):
        with pytest.raises(RuntimeError, match="rigid-body integration"):
            InteractiveScene(_CableSceneCfg(num_envs=1, env_spacing=1.0))


def test_cable_callback_does_not_retain_asset():
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(_CableSceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()
        cable = scene["cable"]
        callback_id = cable._physics_ready_handle.id
        cable_ref = weakref.ref(cable)

        del cable
        del scene
        gc.collect()

        assert cable_ref() is None
        assert callback_id not in SimulationManager._callbacks


@pytest.mark.parametrize(
    ("points", "counts", "parent_scale", "message"),
    [
        (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.4, 0.0)), (3,), None, "finite"),
        (((0.0, 0.0, 0.0), (1.0e-8, 0.0, 0.0), (0.0, 0.4, 0.0)), (3,), None, "longer than"),
        (((0.0, 0.0, 0.0), (0.0, float("nan"), 0.0), (0.0, 0.4, 0.0)), (3,), None, "finite"),
        (((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.4, 0.0, 0.0)), (3,), (0.0, 1.0, 1.0), "finite"),
        (tuple((0.0, 0.1 * index, 0.0) for index in range(6)), (3, 3), None, "one open"),
    ],
)
def test_cable_rejects_invalid_imported_curve(points, counts, parent_scale, message):
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=2),
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        path = "/World/Cable"
        if parent_scale is not None:
            parent = UsdGeom.Xform.Define(sim.stage, "/World/Parent")
            parent.AddScaleOp().Set(parent_scale)
            path = "/World/Parent/Cable"
        _author_curve(sim.stage, path, points, counts)
        with pytest.raises(RuntimeError, match=message):
            NewtonCableObject(CableObjectCfg(prim_path=path))
