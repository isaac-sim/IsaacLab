# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

import numpy as np
import pytest
from gymnasium.spaces import Box, Discrete, MultiDiscrete

from isaaclab.envs.utils.spaces import deserialize_space, serialize_space

pytestmark = pytest.mark.unit


def test_gym_space_serialization_preserves_metadata():
    """Round-tripping Gymnasium spaces should preserve bounds metadata and dtypes."""
    box = Box(
        low=np.array([-1.0, 0.0], dtype=np.float64),
        high=np.array([1.0, 2.0], dtype=np.float64),
        dtype=np.float64,
    )
    box_output = deserialize_space(serialize_space(box))
    assert box_output.dtype == box.dtype
    assert np.array_equal(box_output.low, box.low)
    assert np.array_equal(box_output.high, box.high)

    discrete = Discrete(5, start=2)
    discrete_output = deserialize_space(serialize_space(discrete))
    assert discrete_output.n == discrete.n
    assert discrete_output.start == discrete.start

    multi_discrete = MultiDiscrete(
        np.array([3, 4], dtype=np.int32), start=np.array([1, 2], dtype=np.int32), dtype=np.int32
    )
    multi_discrete_output = deserialize_space(serialize_space(multi_discrete))
    assert multi_discrete_output.dtype == multi_discrete.dtype
    assert np.array_equal(multi_discrete_output.nvec, multi_discrete.nvec)
    assert np.array_equal(multi_discrete_output.start, multi_discrete.start)


def test_gym_space_deserialization_remains_backward_compatible():
    """Serialized payloads created before metadata fields were added should keep previous defaults."""
    box_payload = json.dumps(
        {"type": "gymnasium", "space": "Box", "low": [-1.0], "high": [1.0], "shape": [1]}
    )
    box_output = deserialize_space(box_payload)
    assert box_output.dtype == np.dtype("float32")

    discrete_payload = json.dumps({"type": "gymnasium", "space": "Discrete", "n": 3})
    discrete_output = deserialize_space(discrete_payload)
    assert discrete_output.start == 0

    multi_discrete_payload = json.dumps({"type": "gymnasium", "space": "MultiDiscrete", "nvec": [2, 3]})
    multi_discrete_output = deserialize_space(multi_discrete_payload)
    assert multi_discrete_output.dtype == np.dtype("int64")
    assert np.array_equal(multi_discrete_output.start, np.zeros(2, dtype=np.int64))
