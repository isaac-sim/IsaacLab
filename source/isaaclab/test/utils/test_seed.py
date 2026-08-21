# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import pytest
import torch

from isaaclab.utils.seed import configure_seed

pytestmark = pytest.mark.unit


def test_configure_seed_disables_deterministic_algorithms_when_requested():
    """A non-deterministic call must undo deterministic-algorithm mode enabled by an earlier call."""
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cublas_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

    try:
        configure_seed(123, torch_deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()

        configure_seed(123, torch_deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_algorithms)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        if previous_cublas_config is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_cublas_config
