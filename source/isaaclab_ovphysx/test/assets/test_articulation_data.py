# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for ovphysx articulation data helpers."""

import numpy as np

import warp as wp

from isaaclab_ovphysx import tensor_types as TT
from isaaclab_ovphysx.assets.articulation.articulation_data import ArticulationData
from isaaclab_ovphysx.test.mock_interfaces.views import MockOvPhysxBindingSet

wp.init()


class TestArticulationData:
    """Unit tests for deterministic ArticulationData behavior."""

    def test_joint_acc_uses_inverse_dt(self):
        """Finite-difference joint acceleration should divide by ``dt``."""
        mock_bindings = MockOvPhysxBindingSet(num_instances=1, num_joints=2, num_bodies=1)
        data = ArticulationData(mock_bindings.bindings, device="cpu")
        data._create_buffers()

        mock_bindings.bindings[TT.DOF_VELOCITY]._data[...] = np.array([[1.0, -2.0]], dtype=np.float32)

        data.update(dt=0.25)

        np.testing.assert_allclose(
            data.joint_acc.numpy(),
            np.array([[4.0, -8.0]], dtype=np.float32),
            atol=1e-6,
            err_msg="Joint acceleration should be computed as delta_velocity / dt.",
        )
