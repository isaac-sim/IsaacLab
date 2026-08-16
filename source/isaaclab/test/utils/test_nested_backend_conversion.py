# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from isaaclab.utils.dict import convert_dict_to_backend

pytestmark = pytest.mark.unit


def test_nested_conversion_preserves_requested_backend():
    data = {"outer": {"values": np.array([1.0, 2.0, 3.0], dtype=np.float32)}}

    converted = convert_dict_to_backend(data, backend="torch", array_types=("numpy",))

    assert isinstance(converted["outer"]["values"], torch.Tensor)
    torch.testing.assert_close(converted["outer"]["values"], torch.tensor([1.0, 2.0, 3.0]))
