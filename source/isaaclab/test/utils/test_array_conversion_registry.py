# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import warp as wp

from isaaclab.utils.dict import convert_dict_to_backend

pytestmark = pytest.mark.unit


def test_convert_numpy_array_to_warp_backend():
    data = {"values": np.array([1.0, 2.0, 3.0], dtype=np.float32)}

    converted = convert_dict_to_backend(data, backend="warp", array_types=("numpy",))

    assert isinstance(converted["values"], wp.array)
    np.testing.assert_array_equal(converted["values"].numpy(), data["values"])
