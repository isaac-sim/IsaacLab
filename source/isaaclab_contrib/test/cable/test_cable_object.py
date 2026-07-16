# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import NewtonCfg, XPBDSolverCfg
from isaaclab_newton.physics import NewtonManager as SimulationManager

from pxr import UsdGeom

from isaaclab.assets import CableObjectCfg
from isaaclab.cloner import ReplicateSession
from isaaclab.sim import SimulationCfg, SimulationContext, build_simulation_context
from isaaclab.sim.spawners.materials import CableMaterialCfg
from isaaclab.sim.spawners.shapes import CableCfg

from isaaclab_contrib.cable import CableData, CableObject
from isaaclab_contrib.deformable import VBDSolverCfg


def _make_cable(sim, num_envs: int = 2) -> CableObject:
    cfg = CableObjectCfg(
        prim_path="/World/envs/env_.*/Cable",
        spawn=CableCfg(
            positions=[
                (-0.3, 0.0, 1.0),
                (-0.1, 0.0, 1.0),
                (0.1, 0.0, 1.0),
                (0.3, 0.0, 1.0),
            ],
            physics_material=CableMaterialCfg(
                thickness=0.02,
                density=1000.0,
                stretch_stiffness=1.0e6,
                bend_stiffness=1.0e4,
            ),
        ),
    )
    with ReplicateSession(
        [cfg],
        num_clones=num_envs,
        env_spacing=1.0,
        device=sim.device,
        stage=sim.stage,
    ):
        cable = CableObject(cfg)
    return cable


def test_cable_runtime_gathers_current_state_and_handles_resets():
    sim_cfg = SimulationCfg(
        dt=1.0 / 120.0,
        device="cpu",
        gravity=(0.0, 0.0, -9.81),
        physics=NewtonCfg(
            solver_cfg=VBDSolverCfg(iterations=3),
            num_substeps=1,
            use_cuda_graph=False,
        ),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        cable = _make_cable(sim)
        sim.reset()

        assert cable.num_instances == 2
        assert cable.num_segments == 3
        assert cable.data.segment_pose_w.torch.shape == (2, 3, 7)
        assert cable.data.segment_velocity_w.torch.shape == (2, 3, 6)

        model = SimulationManager.get_model()
        cable_labels = [label for label in model.body_label if "Cable/geometry/mesh_edge_body_" in label]
        assert len(cable_labels) == 6
        assert sum(label.endswith("_edge_body_0") for label in cable_labels) == 2
        assert not hasattr(SimulationManager, "_cable_registry")

        body_world = model.body_world.numpy()
        expected_body_indices = []
        for world in range(2):
            world_indices = []
            for segment in range(3):
                matches = [
                    body_id
                    for body_id, label in enumerate(model.body_label)
                    if int(body_world[body_id]) == world and label.endswith(f"_edge_body_{segment}")
                ]
                assert len(matches) == 1
                world_indices.append(matches[0])
            expected_body_indices.append(world_indices)
        expected_body_indices = torch.tensor(expected_body_indices, dtype=torch.long)

        state = SimulationManager.get_state_0()
        cable.update(0.0)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, wp.to_torch(state.body_q)[expected_body_indices])
        torch.testing.assert_close(
            cable.data.segment_velocity_w.torch,
            wp.to_torch(state.body_qd)[expected_body_indices],
        )

        initial_pose = cable.data.segment_pose_w.torch.clone()
        sim.step(render=False)
        cable.update(sim.cfg.dt)
        stepped_pose = cable.data.segment_pose_w.torch.clone()
        stepped_velocity = cable.data.segment_velocity_w.torch.clone()
        assert torch.all(torch.isfinite(stepped_pose))
        assert torch.any(stepped_pose[..., 2] < initial_pose[..., 2])

        cable.reset(env_ids=[0])
        cable.update(0.0)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, stepped_pose)
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, stepped_velocity)

        previous_body_indices = cable._body_indices
        sim.reset()
        assert SimulationManager.get_model() is not model
        assert cable._body_indices is not previous_body_indices
        assert cable.data._body_indices is cable._body_indices

        state = SimulationManager.get_state_0()
        body_indices = wp.to_torch(cable._body_indices).long()
        cable.update(0.0)
        torch.testing.assert_close(cable.data.segment_pose_w.torch, wp.to_torch(state.body_q)[body_indices])
        torch.testing.assert_close(cable.data.segment_velocity_w.torch, wp.to_torch(state.body_qd)[body_indices])

        sim.step(render=False)
        cable.update(sim.cfg.dt)
        assert torch.all(torch.isfinite(cable.data.segment_pose_w.torch))
        assert torch.all(torch.isfinite(cable.data.segment_velocity_w.torch))


def test_cable_supports_general_prim_regex():
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=2), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim.stage.DefinePrim("/World/Table_0", "Xform")
        cfg = CableObjectCfg(
            prim_path="/World/Table_[0-9]+/Cable",
            spawn=CableCfg(
                positions=[(0.0, 0.0, 1.0), (0.0, 0.2, 1.0), (0.0, 0.4, 1.0)],
                physics_material=CableMaterialCfg(),
            ),
        )
        cable = CableObject(cfg)
        sim.reset()

        assert cable.num_instances == 1
        assert cable.data.segment_pose_w.torch.shape == (1, 2, 7)


def test_cable_data_reports_missing_state(monkeypatch):
    """Test the error reported when Newton state is unavailable."""
    body_indices = wp.array([[0]], dtype=wp.int32, device="cpu")
    data = CableData(body_indices, "cpu")
    monkeypatch.setattr(SimulationManager, "get_state_0", staticmethod(lambda: None))

    with pytest.raises(RuntimeError, match="body state is unavailable"):
        data.update(0.0)


def test_cable_rejects_periodic_curve():
    """Test the runtime topology guard for an unsupported periodic curve."""
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(solver_cfg=VBDSolverCfg(iterations=2), use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        curves = UsdGeom.BasisCurves.Define(sim.stage, "/World/Cable")
        curves.CreatePointsAttr([(0.0, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.4, 0.0)])
        curves.CreateCurveVertexCountsAttr([3])
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.periodic)
        curves.GetPrim().AddAppliedSchema("PhysicsCurvesDeformableSimAPI")

        with pytest.raises(ValueError, match="one open, linear"):
            CableObject(CableObjectCfg(prim_path="/World/Cable"))


@pytest.mark.parametrize(
    "solver_cfg",
    [
        XPBDSolverCfg(),
        VBDSolverCfg(integrate_with_external_rigid_solver=True),
        VBDSolverCfg(class_type="isaaclab_newton.physics.xpbd_manager:NewtonXPBDManager"),
    ],
)
def test_cable_rejects_unsupported_solver_cfg(solver_cfg):
    sim_cfg = SimulationCfg(
        device="cpu",
        physics=NewtonCfg(solver_cfg=solver_cfg, use_cuda_graph=False),
    )

    with build_simulation_context(sim_cfg=sim_cfg):
        cfg = CableObjectCfg(prim_path="/World/Cable")
        with pytest.raises(RuntimeError, match="uncoupled NewtonVBDManager"):
            CableObject(cfg)


def test_cable_rejects_backend_without_solver_cfg(monkeypatch):
    class _OtherManager:
        pass

    sim = SimpleNamespace(
        cfg=SimpleNamespace(physics=object()),
        physics_manager=_OtherManager,
        vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda owner: None),
    )
    monkeypatch.setattr(SimulationContext, "instance", staticmethod(lambda: sim))

    with pytest.raises(RuntimeError, match="uncoupled NewtonVBDManager"):
        CableObject(CableObjectCfg(prim_path="/World/Cable"))
