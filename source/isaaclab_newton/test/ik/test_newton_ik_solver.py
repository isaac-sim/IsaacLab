# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab_newton.ik.newton_ik_solver as ik_solver_module
import torch
import warp as wp
from isaaclab_newton.ik.newton_ik_solver import NewtonIKPoseObjective, NewtonIKSolver
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg


class _Model:
    joint_coord_count = 2
    joint_limit_lower = None
    joint_limit_upper = None


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
    monkeypatch.setattr(ik_solver_module.ik, "IKObjectivePosition", _PoseObjective)
    monkeypatch.setattr(ik_solver_module.ik, "IKObjectiveRotation", _PoseObjective)
    monkeypatch.setattr(ik_solver_module.ik, "IKObjectiveJointLimit", _JointLimitObjective)
    monkeypatch.setattr(ik_solver_module.ik, "IKSolver", _Solver)
    monkeypatch.setattr(ik_solver_module.ik, "IKOptimizer", lambda value: value)
    monkeypatch.setattr(ik_solver_module.ik, "IKJacobianType", lambda value: value)
    monkeypatch.setattr(ik_solver_module.ik, "IKSampler", lambda value: value)


def _cfg() -> NewtonIKSolverCfg:
    cfg = NewtonIKSolverCfg()
    cfg.use_persistent_seed = True
    cfg.joint_limit_weight = None
    return cfg


def test_persistent_seed_is_reused_between_solves(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = NewtonIKSolver(
        _cfg(),
        model=_Model(),
        num_envs=2,
        device="cpu",
        pose_objectives=[NewtonIKPoseObjective(name="ee", link_index=0)],
    )

    seed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    solver.set_joint_seed(seed)

    first = solver.solve().clone()
    second = solver.solve().clone()

    assert torch.allclose(first, seed + 1.0)
    assert torch.allclose(second, seed + 2.0)


def test_solve_result_is_independent_from_next_solve(monkeypatch):
    _patch_newton_ik(monkeypatch)
    cfg = _cfg()
    cfg.use_persistent_seed = False
    solver = NewtonIKSolver(
        cfg,
        model=_Model(),
        num_envs=2,
        device="cpu",
        pose_objectives=[NewtonIKPoseObjective(name="ee", link_index=0)],
    )
    seed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    first = solver.solve(seed)
    expected_first = first.clone()
    solver.solve(seed + 10.0)

    assert torch.allclose(first, expected_first)


def test_set_target_pose_updates_named_objective(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = NewtonIKSolver(
        _cfg(),
        model=_Model(),
        num_envs=2,
        device="cpu",
        pose_objectives=[NewtonIKPoseObjective(name="ee", link_index=0)],
    )
    target_pos = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]])
    target_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]])

    solver.set_target_pose("ee", target_pos, target_quat)

    assert torch.allclose(wp.to_torch(solver.position_objectives["ee"].target), target_pos)
    assert torch.allclose(wp.to_torch(solver.rotation_objectives["ee"].target), target_quat)


def test_pose_targets_can_be_initialized_from_body_transforms(monkeypatch):
    _patch_newton_ik(monkeypatch)
    solver = NewtonIKSolver(
        _cfg(),
        model=_Model(),
        num_envs=2,
        device="cpu",
        pose_objectives=[
            NewtonIKPoseObjective(name="ee", link_index=0),
            NewtonIKPoseObjective(name="torso", link_index=1, position_weight=50.0, rotation_weight=50.0),
        ],
    )
    body_q = torch.tensor(
        [
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    solver.set_pose_targets_from_body_q(body_q, names=["torso"], env_origins=env_origins)

    target_pos = wp.to_torch(solver.position_objectives["torso"].target)
    target_quat = wp.to_torch(solver.rotation_objectives["torso"].target)
    assert torch.allclose(target_pos, torch.tensor([[4.0, 5.0, 6.0], [3.0, 4.0, 5.0]]))
    assert torch.allclose(target_quat, torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]))
