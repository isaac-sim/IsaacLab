# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for DiffIK Jacobian aliasing (NVBug 6043099).

``DifferentialInverseKinematicsAction._compute_frame_jacobian`` historically
aliased the parent Jacobian and applied the body-offset correction in place.
When the parent Jacobian was a view onto the engine's mutable buffer, repeated
calls within a single simulation step accumulated the correction. The fix
copies the Jacobian into an owned buffer before mutating, making the method
idempotent regardless of whether ``jacobian_b`` returns a view or a copy.
"""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils


class _Stub:
    """Minimal stand-in for ``DifferentialInverseKinematicsAction`` that exposes only
    what ``_compute_frame_jacobian`` reads. ``jacobian_b`` returns the backing buffer
    **without copying**, mirroring the worst case where the data layer hands out a view
    onto engine memory. The owned ``_jacobian_b`` buffer is what the fixed method must
    write into.
    """

    def __init__(self, num_envs: int, num_joints: int, body_offset_pos, body_offset_rot, backing_buffer):
        self.cfg = SimpleNamespace(body_offset=SimpleNamespace(pos=body_offset_pos, rot=body_offset_rot))
        self._offset_pos = torch.tensor(body_offset_pos, dtype=torch.float32).repeat(num_envs, 1)
        self._offset_rot = torch.tensor(body_offset_rot, dtype=torch.float32).repeat(num_envs, 1)
        self._jacobian_b = torch.zeros(num_envs, 6, num_joints)
        self._backing_buffer = backing_buffer

    @property
    def jacobian_b(self):
        return self._backing_buffer


def _make_stub(num_envs: int, num_joints: int, body_offset_pos, body_offset_rot, backing_buffer: torch.Tensor):
    return _Stub(num_envs, num_joints, body_offset_pos, body_offset_rot, backing_buffer)


def test_compute_frame_jacobian_is_idempotent_within_step():
    """Two consecutive calls under the same state must return identical Jacobians.

    Regression for NVBug 6043099. With the buggy alias-and-mutate pattern, the
    second call returned the first-call result with the body-offset correction
    applied a second time.
    """
    num_envs, num_joints = 4, 7
    backing = torch.randn(num_envs, 6, num_joints)
    backing_snapshot = backing.clone()

    stub = _make_stub(
        num_envs,
        num_joints,
        body_offset_pos=[0.0, 0.0, 0.05],
        body_offset_rot=[1.0, 0.0, 0.0, 0.0],
        backing_buffer=backing,
    )

    compute = DifferentialInverseKinematicsAction._compute_frame_jacobian

    j1 = compute(stub).clone()
    j2 = compute(stub).clone()
    j3 = compute(stub).clone()

    torch.testing.assert_close(j1, j2)
    torch.testing.assert_close(j1, j3)
    # The backing buffer must be untouched: the fix may not corrupt the source.
    torch.testing.assert_close(backing, backing_snapshot)


def test_compute_frame_jacobian_applies_offset_once():
    """The body-offset correction is applied exactly once.

    Compares the method output against an out-of-place reference computation.
    """
    num_envs, num_joints = 2, 5
    backing = torch.randn(num_envs, 6, num_joints)
    offset_pos = [0.1, -0.02, 0.03]
    offset_rot = [0.7071, 0.0, 0.7071, 0.0]

    stub = _make_stub(num_envs, num_joints, offset_pos, offset_rot, backing)

    # Reference: out-of-place computation, no aliasing.
    skew = math_utils.skew_symmetric_matrix(stub._offset_pos)
    rot = math_utils.matrix_from_quat(stub._offset_rot)
    ref_trans = backing[:, 0:3, :] + torch.bmm(-skew, backing[:, 3:, :])
    ref_rot = torch.bmm(rot, backing[:, 3:, :])
    reference = torch.cat([ref_trans, ref_rot], dim=1)

    actual = DifferentialInverseKinematicsAction._compute_frame_jacobian(stub)
    torch.testing.assert_close(actual, reference)


def test_compute_frame_jacobian_returns_owned_buffer():
    """The returned tensor must be the owned buffer, not the data-layer source.

    Guards against future regressions where ``return self.jacobian_b`` slips back in.
    """
    num_envs, num_joints = 1, 3
    backing = torch.randn(num_envs, 6, num_joints)
    stub = _make_stub(num_envs, num_joints, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], backing)

    out = DifferentialInverseKinematicsAction._compute_frame_jacobian(stub)
    assert out.data_ptr() == stub._jacobian_b.data_ptr()
    assert out.data_ptr() != backing.data_ptr()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
