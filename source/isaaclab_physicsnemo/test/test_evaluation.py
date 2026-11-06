# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Open-loop comparison of learned and teacher joint rollouts."""

import math

import pytest

pytest.importorskip("physicsnemo_robotics")

import torch
from isaaclab_physicsnemo import JointRollout, compare_rollouts

# One free root (position, then an (x, y, z, w) quaternion) followed by one revolute joint.
_REVOLUTE_Q = torch.tensor([False] * 7 + [True])
_QUATERNION_Q_STARTS = (3,)


def _rollout(joint_q: torch.Tensor, active: torch.Tensor) -> JointRollout:
    return JointRollout(
        joint_q=joint_q,
        joint_qd=torch.zeros(*joint_q.shape[:2], 7),
        active=active,
        revolute_q=_REVOLUTE_Q,
        quaternion_q_starts=_QUATERNION_Q_STARTS,
    )


def test_equivalent_configurations_have_no_error():
    """A full revolute turn and a negated quaternion describe the same configuration."""
    teacher_q = torch.tensor([[[0.0, 0.0, 0.5, 0.0, 0.0, 0.6, 0.8, 3.0]]])
    learned_q = torch.tensor([[[0.0, 0.0, 0.5, -0.0, -0.0, -0.6, -0.8, 3.0 - 2.0 * math.pi]]])
    active = torch.ones(1, 1, dtype=torch.bool)

    error = compare_rollouts(_rollout(teacher_q, active), _rollout(learned_q, active))

    torch.testing.assert_close(error.joint_q_rmse, torch.zeros(1), atol=1e-6, rtol=0.0)


def test_error_counts_only_environments_active_in_both_rollouts():
    teacher_q = torch.zeros(2, 3, 8)
    teacher_q[..., 6] = 1.0
    learned_q = teacher_q.clone()
    learned_q[1, 0, 0] = 0.8  # active environment, 0.8 m root error
    learned_q[1, 1, 0] = 100.0  # reset in the learned rollout
    learned_q[1, 2, 0] = 100.0  # reset in the teacher rollout
    teacher_active = torch.tensor([[True, True, True], [True, True, False]])
    learned_active = torch.tensor([[True, True, True], [True, False, True]])

    error = compare_rollouts(_rollout(teacher_q, teacher_active), _rollout(learned_q, learned_active))

    torch.testing.assert_close(error.active_envs, torch.tensor([3, 1]))
    torch.testing.assert_close(error.joint_q_rmse, torch.tensor([0.0, math.sqrt(0.8**2 / 8)]))
