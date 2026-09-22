# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for DiffIK Jacobian aliasing (NVBug 6043099).

``DifferentialInverseKinematicsAction._compute_frame_jacobian`` historically aliased the parent Jacobian and
applied the body-offset correction in place. When the parent Jacobian was a view onto the engine's mutable
buffer, repeated calls within a single simulation step accumulated the correction. The fix copies the Jacobian
into an owned buffer before mutating, making the method idempotent regardless of whether ``jacobian_b`` returns
a view or a copy.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import math as math_utils

pytestmark = pytest.mark.unit


class _Stub:
    """Exposes only what ``_compute_frame_jacobian`` reads.

    ``jacobian_b`` hands out the backing buffer without copying, mirroring the worst case where the data layer
    returns a view onto engine memory. The owned ``_jacobian_b`` buffer is what the fixed method must write into.
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


def test_compute_frame_jacobian_is_idempotent_and_owns_its_buffer():
    """Repeated calls return identical Jacobians written into the owned buffer without touching the source."""
    num_envs, num_joints = 4, 7
    backing = torch.randn(num_envs, 6, num_joints)
    backing_snapshot = backing.clone()
    stub = _Stub(num_envs, num_joints, [0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0], backing)
    compute = DifferentialInverseKinematicsAction._compute_frame_jacobian

    outputs = [compute(stub) for _ in range(3)]
    for out in outputs:
        assert out.data_ptr() == stub._jacobian_b.data_ptr()
        assert out.data_ptr() != backing.data_ptr()
    results = [out.clone() for out in outputs]
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(results[0], results[2])
    torch.testing.assert_close(backing, backing_snapshot)


def test_compute_frame_jacobian_applies_offset_once():
    """The body-offset correction matches an out-of-place reference computation."""
    num_envs, num_joints = 2, 5
    backing = torch.randn(num_envs, 6, num_joints)
    stub = _Stub(num_envs, num_joints, [0.1, -0.02, 0.03], [0.7071, 0.0, 0.7071, 0.0], backing)

    skew = math_utils.skew_symmetric_matrix(stub._offset_pos)
    rot = math_utils.matrix_from_quat(stub._offset_rot)
    reference = torch.cat(
        [backing[:, 0:3, :] + torch.bmm(-skew, backing[:, 3:, :]), torch.bmm(rot, backing[:, 3:, :])], dim=1
    )

    torch.testing.assert_close(DifferentialInverseKinematicsAction._compute_frame_jacobian(stub), reference)
