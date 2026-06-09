# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import importlib.util
from pathlib import Path

import pytest
import torch


def _load_soft_dtw_module():
    module_path = (
        Path(__file__).parents[2]
        / "isaaclab_tasks"
        / "contrib"
        / "automate"
        / "soft_dtw_cuda.py"
    )
    spec = importlib.util.spec_from_file_location("automate_soft_dtw_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_soft_dtw_use_cuda_does_not_require_numba():
    soft_dtw = _load_soft_dtw_module()
    criterion = soft_dtw.SoftDTW(use_cuda=True, device="cuda", gamma=0.01)

    x = torch.zeros((1, 2, 3), dtype=torch.float32)
    y = torch.zeros((1, 2, 3), dtype=torch.float32)

    assert criterion(x, y).shape == (1,)


def test_soft_dtw_hard_dtw_value():
    soft_dtw = _load_soft_dtw_module()
    criterion = soft_dtw.SoftDTW(use_cuda=False, device="cpu", gamma=0.0)

    x = torch.tensor([[[0.0], [1.0]]])
    y = torch.tensor([[[0.0], [2.0]]])

    assert criterion(x, y) == pytest.approx(torch.tensor([1.0]))


def test_normalized_soft_dtw_identical_sequences_are_zero():
    soft_dtw = _load_soft_dtw_module()
    criterion = soft_dtw.SoftDTW(use_cuda=True, device="cuda", gamma=0.1, normalize=True)

    x = torch.tensor([[[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]])

    assert criterion(x, x) == pytest.approx(torch.zeros(1), abs=1e-6)
