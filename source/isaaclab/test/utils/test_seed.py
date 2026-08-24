# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

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

    # Keep the regression focused on deterministic-mode state rather than RNG state.
    monkeypatch.setattr(seed_utils.random, "seed", lambda _seed: None)
    monkeypatch.setattr(seed_utils.np.random, "seed", lambda _seed: None)
    monkeypatch.setattr(seed_utils.torch, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(seed_utils.torch.cuda, "manual_seed", lambda _seed: None)
    monkeypatch.setattr(seed_utils.torch.cuda, "manual_seed_all", lambda _seed: None)
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
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
