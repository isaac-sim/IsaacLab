# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random

import numpy as np
import pytest
import torch

import isaaclab.utils.seed as seed_utils

pytestmark = pytest.mark.unit


def test_configure_seed_disables_deterministic_algorithms_when_requested(monkeypatch):
    """A non-deterministic call must undo deterministic-algorithm mode enabled by an earlier call."""
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_env = {key: os.environ.get(key) for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")}
    previous_random_state = random.getstate()
    previous_numpy_state = np.random.get_state()
    previous_torch_state = torch.get_rng_state()
    previous_cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    monkeypatch.setattr(seed_utils.wp, "rand_init", lambda _seed: None)

    try:
        seed_utils.configure_seed(123, torch_deterministic=True)
        assert torch.are_deterministic_algorithms_enabled()
        seed_utils.configure_seed(123, torch_deterministic=False)
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(previous_algorithms, warn_only=previous_warn_only)
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        random.setstate(previous_random_state)
        np.random.set_state(previous_numpy_state)
        torch.set_rng_state(previous_torch_state)
        if previous_cuda_states is not None:
            torch.cuda.set_rng_state_all(previous_cuda_states)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
