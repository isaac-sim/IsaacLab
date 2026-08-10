# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab_newton.ik.newton_ik_objectives as objectives_module
import isaaclab_newton.ik.newton_ik_solver as ik_solver_module
import newton
import pytest
import torch
import warp as wp
from isaaclab_newton.ik.newton_ik_objectives import (
    NewtonIKBuildContext,
    NewtonIKJointPostureObjective,
    NewtonIKObjective,
    NewtonIKPoseObjective,
)
from isaaclab_newton.ik.newton_ik_objectives_cfg import (
    NewtonIKJointLimitObjectiveCfg,
    NewtonIKJointPostureObjectiveCfg,
    NewtonIKObjectiveCfg,
    NewtonIKPoseObjectiveCfg,
)
from isaaclab_newton.ik.newton_ik_solver import NewtonIKSolver
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.utils.configclass import configclass

# Maps the stub body names used across these tests to Newton link indices.
_LINKS = {"ee": 0, "torso": 1, "custom": 0}
_JOINTS = {"shoulder": (0, 0), "elbow": (1, 1)}


def _resolver(body_name: str) -> int:
    return _LINKS[body_name]


def _joint_resolver(joint_name: str) -> tuple[int, int]:
    return _JOINTS[joint_name]


class _Model:
    joint_coord_count = 2
    joint_dof_count = 2
    joint_q = wp.array([0.25, -0.5], dtype=wp.float32, device="cpu")
    joint_limit_lower = None
    joint_limit_upper = None
    body_label = None


class _PoseObjective:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.target = kwargs.get("target_positions", kwargs.get("target_rotations"))

    def set_target_positions(self, target):
        self.target = target

    def set_target_rotations(self, target):
        self.target = target


class _JointLimitObjective:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _Solver:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.joint_q = wp.zeros((1, 2), dtype=wp.float32, device="cpu")
        self.costs = wp.zeros((1,), dtype=wp.float32, device="cpu")
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def step(self, joint_q_in, joint_q_out, *, iterations, step_size):
        del iterations, step_size
        wp.to_torch(joint_q_out).copy_(wp.to_torch(joint_q_in) + 1.0)


def _patch_newton_ik(monkeypatch):
    # ``newton.ik`` is the same module object in both the solver and objective
    # modules, so patching its attributes affects objective construction too.
    monkeypatch.setattr(objectives_module.ik, "IKObjectivePosition", _PoseObjective)
    monkeypatch.setattr(objectives_module.ik, "IKObjectiveRotation", _PoseObjective)
    monkeypatch.setattr(objectives_module.ik, "IKObjectiveJointLimit", _JointLimitObjective)
    monkeypatch.setattr(ik_solver_module.ik, "IKSolver", _Solver)
    monkeypatch.setattr(ik_solver_module.ik, "IKOptimizer", lambda value: value)
    monkeypatch.setattr(ik_solver_module.ik, "IKJacobianType", lambda value: value)
    monkeypatch.setattr(ik_solver_module.ik, "IKSampler", lambda value: value)


def _pose_solver(cfg: NewtonIKSolverCfg | None = None, num_envs: int = 2, objectives=None) -> NewtonIKSolver:
    return NewtonIKSolver(
        NewtonIKSolverCfg() if cfg is None else cfg,
        model=_Model(),
        num_envs=num_envs,
        device="cpu",
        objectives=[NewtonIKPoseObjectiveCfg(body_name="ee")] if objectives is None else objectives,
        link_resolver=_resolver,
        joint_resolver=_joint_resolver,
    )


def test_solve_writes_output_buffer(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = _pose_solver()
    seed = wp.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dtype=wp.float32)

    result = solver.solve(seed)

    assert torch.allclose(wp.to_torch(result), torch.tensor([[2.0, 3.0], [4.0, 5.0]]))


def test_constraint_objectives_carry_no_target_or_action(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = _pose_solver(
        objectives=[
            NewtonIKPoseObjectiveCfg(body_name="ee"),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ]
    )
    # Only the pose objective is named and command-driven; the joint limit is a
    # pure constraint (no name, no action dimensions).
    assert list(solver.objectives_by_name) == ["ee"]
    assert [obj.action_dim for obj in solver.objectives] == [6, 0]


def test_joint_posture_objective_is_constraint_and_uses_named_reference(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = _pose_solver(
        objectives=[
            NewtonIKPoseObjectiveCfg(body_name="ee"),
            NewtonIKJointPostureObjectiveCfg(
                joint_names=["shoulder", "elbow"],
                target_positions=(0.3, -0.4),
                weight=0.05,
            ),
        ]
    )

    posture = solver.objectives[1]
    assert list(solver.objectives_by_name) == ["ee"]
    assert posture.action_dim == 0
    assert torch.equal(wp.to_torch(posture.objective.coordinate_indices), torch.tensor([0, 1], dtype=torch.int32))
    assert torch.equal(wp.to_torch(posture.objective.dof_indices), torch.tensor([0, 1], dtype=torch.int32))
    assert torch.allclose(wp.to_torch(posture.objective.target_positions), torch.tensor([0.3, -0.4]))
    assert torch.allclose(wp.to_torch(posture.objective.weights), torch.full((2,), 0.05))


def test_joint_posture_objective_defaults_to_model_posture(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = _pose_solver(
        objectives=[
            NewtonIKPoseObjectiveCfg(body_name="ee"),
            NewtonIKJointPostureObjectiveCfg(joint_names=["shoulder", "elbow"]),
        ]
    )

    posture = solver.objectives[1]
    assert torch.allclose(wp.to_torch(posture.objective.target_positions), wp.to_torch(_Model.joint_q))


def test_joint_posture_objective_residual_and_analytic_jacobian():
    ctx = NewtonIKBuildContext(
        model=_Model(),
        num_envs=2,
        device="cpu",
        resolve_link=_resolver,
        resolve_joint=_joint_resolver,
    )
    posture = NewtonIKJointPostureObjective(
        NewtonIKJointPostureObjectiveCfg(joint_names=["shoulder", "elbow"], target_positions=(0.25, -0.5), weight=0.2),
        ctx,
    ).objective
    posture.set_batch_layout(total_residuals=3, residual_offset=1, n_batch=2)
    posture.bind_device("cpu")
    posture.init_buffers(_Model(), objectives_module.ik.IKJacobianType.ANALYTIC)

    joint_q = wp.array([[0.5, -1.0], [0.1, -0.3]], dtype=wp.float32, device="cpu")
    residuals = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    problem_idx = wp.array([0, 1], dtype=wp.int32, device="cpu")
    posture.compute_residuals(None, joint_q, _Model(), residuals, 1, problem_idx)

    assert torch.allclose(
        wp.to_torch(residuals),
        torch.tensor([[0.0, 0.05, -0.1], [0.0, -0.03, 0.04]]),
    )

    jacobian = wp.zeros((2, 3, 2), dtype=wp.float32, device="cpu")
    posture.compute_jacobian_analytic(None, joint_q, _Model(), jacobian, None, 1)
    expected = torch.zeros((2, 3, 2))
    expected[:, 1, 0] = 0.2
    expected[:, 2, 1] = 0.2
    assert torch.allclose(wp.to_torch(jacobian), expected)


@pytest.mark.parametrize(
    "jacobian_mode",
    [objectives_module.ik.IKJacobianType.AUTODIFF, objectives_module.ik.IKJacobianType.ANALYTIC],
)
def test_joint_posture_objective_converges_with_newton_solver(jacobian_mode):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    unit_inertia = wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    link_1 = builder.add_link(mass=1.0, inertia=unit_inertia)
    joint_1 = builder.add_joint_revolute(parent=-1, child=link_1, axis=newton.Axis.Z)
    link_2 = builder.add_link(mass=1.0, inertia=unit_inertia)
    joint_2 = builder.add_joint_revolute(parent=link_1, child=link_2, axis=newton.Axis.Z)
    builder.add_articulation([joint_1, joint_2])
    model = builder.finalize(device="cpu", requires_grad=True)

    ctx = NewtonIKBuildContext(
        model=model,
        num_envs=2,
        device="cpu",
        resolve_link=_resolver,
        resolve_joint=_joint_resolver,
    )
    posture = NewtonIKJointPostureObjective(
        NewtonIKJointPostureObjectiveCfg(joint_names=["shoulder", "elbow"], target_positions=(0.25, -0.5), weight=0.2),
        ctx,
    ).objective
    solver = objectives_module.ik.IKSolver(
        model,
        2,
        [posture],
        lambda_initial=1.0e-3,
        jacobian_mode=jacobian_mode,
    )
    requires_grad = jacobian_mode == objectives_module.ik.IKJacobianType.AUTODIFF
    seed = wp.array([[1.0, -1.0], [-0.75, 0.9]], dtype=wp.float32, device="cpu", requires_grad=requires_grad)
    solved = wp.zeros((2, 2), dtype=wp.float32, device="cpu", requires_grad=requires_grad)

    solver.step(seed, solved, iterations=20)

    assert torch.allclose(wp.to_torch(solved), torch.tensor([[0.25, -0.5], [0.25, -0.5]]), atol=1.0e-4)


def test_multiple_pose_objectives_register_distinct_targets(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = _pose_solver(
        objectives=[
            NewtonIKPoseObjectiveCfg(body_name="ee"),
            NewtonIKPoseObjectiveCfg(body_name="torso"),
            NewtonIKJointLimitObjectiveCfg(weight=0.1),
        ]
    )
    assert list(solver.objectives_by_name) == ["ee", "torso"]
    assert solver.objectives_by_name["ee"].link_index == 0
    assert solver.objectives_by_name["torso"].link_index == 1


def _build_pose_objective(cfg: NewtonIKPoseObjectiveCfg, num_envs: int = 2) -> NewtonIKPoseObjective:
    ctx = NewtonIKBuildContext(model=_Model(), num_envs=num_envs, device="cpu", resolve_link=_resolver)
    return NewtonIKPoseObjective(cfg, ctx)


def test_pose_objective_action_dim_and_coordinate_names(monkeypatch):
    _patch_newton_ik(monkeypatch)
    rel_pose = _build_pose_objective(
        NewtonIKPoseObjectiveCfg(body_name="ee", command_type="pose", use_relative_mode=True)
    )
    abs_pose = _build_pose_objective(
        NewtonIKPoseObjectiveCfg(body_name="ee", command_type="pose", use_relative_mode=False)
    )
    position = _build_pose_objective(NewtonIKPoseObjectiveCfg(body_name="ee", command_type="position"))

    assert rel_pose.action_dim == 6
    assert abs_pose.action_dim == 7
    assert position.action_dim == 3
    assert position.command_coordinate_names() == ["x", "y", "z"]
    assert rel_pose.command_coordinate_names() == ["x", "y", "z", "roll", "pitch", "yaw"]


def test_pose_objective_exposes_warp_command_data(monkeypatch):
    _patch_newton_ik(monkeypatch)
    rel_pose = _build_pose_objective(
        NewtonIKPoseObjectiveCfg(body_name="ee", command_type="pose", use_relative_mode=True, scale=0.5)
    )
    position = _build_pose_objective(
        NewtonIKPoseObjectiveCfg(body_name="ee", command_type="position", use_relative_mode=False)
    )

    # Command convention is exposed to the action's Warp kernel as plain data.
    assert (rel_pose.command_code, rel_pose.use_relative) == (1, 1)
    assert (position.command_code, position.use_relative) == (0, 0)
    assert torch.allclose(wp.to_torch(rel_pose.scale), torch.full((6,), 0.5))


class _CustomObjective(NewtonIKObjective):
    SENTINEL = object()

    def __init__(self, cfg, ctx):
        del cfg, ctx
        self.name = "custom"
        self.solver_objectives = [_CustomObjective.SENTINEL]


def test_custom_objective_cfg_is_built_and_wired(monkeypatch):
    _patch_newton_ik(monkeypatch)

    @configclass
    class _CustomObjectiveCfg(NewtonIKObjectiveCfg):
        class_type: type | str = _CustomObjective

    solver = _pose_solver(objectives=[NewtonIKPoseObjectiveCfg(body_name="ee"), _CustomObjectiveCfg()])

    assert "custom" in solver.objectives_by_name
    assert _CustomObjective.SENTINEL in solver.solver.kwargs["objectives"]
