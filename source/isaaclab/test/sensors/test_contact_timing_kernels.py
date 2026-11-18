# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact timing and compatibility of the backend-specific Warp kernel entry points."""

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.sensors.contact_sensor import contact_sensor_kernels as newton_kernels
from isaaclab_ov.sensors.contact_sensor import kernels as ov_kernels
from isaaclab_physx.sensors.contact_sensor import kernels as physx_kernels

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


@pytest.mark.parametrize("device", test_devices(DeviceScope.CPU))
@pytest.mark.parametrize("backend", ["physx", "ovphysx", "newton"])
def test_contact_kernel_entry_points_preserve_timing_and_mask(backend, device):
    """Old timestamp callers and new elapsed-time callers update forces and transitions correctly."""
    for use_elapsed in (False, True):
        timestamp = wp.full(2, 8.25, dtype=wp.float32, device=device)
        timing = (
            [wp.full(2, 0.02, dtype=wp.float64, device=device)]
            if use_elapsed
            else [timestamp, wp.full(2, 8.0, dtype=wp.float32, device=device)]
        )
        mask = wp.array([True, False], dtype=wp.bool, device=device)
        normal = wp.array([[(0.0, 0.0, 10.0)], [(0.0, 0.0, 20.0)]], dtype=wp.vec3f, device=device)
        friction = wp.zeros((2, 1), dtype=wp.vec3f, device=device)
        normal_history = wp.zeros((2, 2, 1), dtype=wp.vec3f, device=device)
        total_history = wp.zeros_like(normal_history)
        normal_output = wp.zeros_like(normal)
        total_output = wp.zeros_like(normal)
        air = wp.full((2, 1), 0.5, dtype=wp.float32, device=device)
        contact = wp.zeros_like(air)
        last_air = wp.full_like(air, -1.0)
        last_contact = wp.full_like(air, -1.0)

        if backend == "newton":
            kernel = (
                newton_kernels.update_contact_sensor_elapsed_kernel
                if use_elapsed
                else newton_kernels.update_contact_sensor_kernel
            )
            inputs = [
                2,
                0,
                1.0,
                mask,
                normal,
                None,
                normal,
                None,
                friction,
                None,
                *timing,
                total_history,
                None,
                normal_history,
                None,
                None,
                None,
                air,
                contact,
                last_air,
                last_contact,
            ]
        else:
            if backend == "physx":
                kernel = (
                    physx_kernels.update_net_forces_elapsed_kernel
                    if use_elapsed
                    else physx_kernels.update_net_forces_kernel
                )
            else:
                kernel = (
                    ov_kernels.update_net_forces_ovphysx_elapsed_kernel
                    if use_elapsed
                    else ov_kernels.update_net_forces_ovphysx_kernel
                )
            inputs = [
                normal.flatten(),
                friction.flatten(),
                None,
                None,
                mask,
                2,
                0,
                2,
                1.0,
                *([timestamp, *timing] if use_elapsed else timing),
                normal_output,
                normal_history,
                None,
                None,
                air,
                contact,
                last_air,
                last_contact,
                total_output,
                total_history,
                None,
                None,
                None,
                None,
                None,
                None,
            ]
        wp.launch(kernel, dim=(2, 1), inputs=inputs, device=device)

        dt = 0.02 if use_elapsed else 0.25
        np.testing.assert_allclose(contact.numpy(), [[dt], [0.0]], atol=1e-7, rtol=0)
        np.testing.assert_allclose(last_air.numpy(), [[0.5 + dt], [-1.0]], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(air.numpy(), [[0.0], [0.5]])
        np.testing.assert_array_equal(last_contact.numpy(), [[-1.0], [-1.0]])
        expected_history = np.zeros((2, 2, 1, 3), dtype=np.float32)
        expected_history[0, 0, 0, 2] = 10.0
        np.testing.assert_array_equal(normal_history.numpy(), expected_history)
        np.testing.assert_array_equal(total_history.numpy(), expected_history)
