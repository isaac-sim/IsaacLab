# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysx contact-sensor Warp kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.sensors.contact_sensor.kernels import unpack_contact_buffer_data


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("track_points,track_friction", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("capacity", [None, 5])
def test_unpack_contact_buffer_data_pattern_major(
    device: str, track_points: bool, track_friction: bool, use_mask: bool, capacity: int | None
):
    """Aggregate independent contact and friction layouts, preserving unselected environments."""
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is not available")
    num_envs, num_sensors, num_filters = 3, 4, 2
    pair_shape = (num_envs * num_sensors, num_filters)
    inputs, outputs, references = [], [], []
    for avg, enabled in ((True, track_points), (False, track_friction)):
        if not enabled:
            inputs.extend([None, None, None])
            outputs.append(None)
            references.append(None)
            continue
        counts = (np.arange(num_envs * num_sensors * num_filters, dtype=np.uint32) + int(avg)) % 3
        starts = np.cumsum(counts, dtype=np.uint32) - counts
        flat = np.arange(int(counts.sum()) * 3, dtype=np.float32).reshape(-1, 3) * (1.0 if avg else -1.0)
        flat = flat[:capacity]
        counts, starts = counts.reshape(pair_shape), starts.reshape(pair_shape)
        default = float("nan") if avg else 0.0
        aggregate = np.mean if avg else np.sum
        expected = np.full((num_envs, num_sensors, num_filters, 3), -1.0, dtype=np.float32)
        for env in (0, 2) if use_mask else range(num_envs):
            for sensor in range(num_sensors):
                for partner in range(num_filters):
                    row = sensor * num_envs + env
                    start, count = int(starts[row, partner]), int(counts[row, partner])
                    values = flat[start : start + count]
                    expected[env, sensor, partner] = aggregate(values, axis=0) if len(values) else default
        inputs.extend(
            [
                wp.array(flat, dtype=wp.vec3f, device=device),
                wp.array(counts, dtype=wp.uint32, device=device),
                wp.array(starts, dtype=wp.uint32, device=device),
            ]
        )
        outputs.append(
            wp.full(
                (num_envs, num_sensors, num_filters), value=wp.vec3f(wp.float32(-1.0)), dtype=wp.vec3f, device=device
            )
        )
        references.append(expected)
    mask = wp.array([True, False, True], dtype=wp.bool, device=device) if use_mask else None
    wp.launch(
        unpack_contact_buffer_data,
        dim=(num_envs, num_sensors, num_filters),
        inputs=[*inputs, mask, num_envs],
        outputs=outputs,
        device=device,
    )
    for output, expected in zip(outputs, references):
        if output is not None:
            np.testing.assert_allclose(output.numpy(), expected, equal_nan=True)
