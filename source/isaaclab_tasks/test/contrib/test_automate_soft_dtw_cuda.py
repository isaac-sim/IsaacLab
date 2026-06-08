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
    spec = importlib.util.spec_from_file_location("automate_soft_dtw_cuda_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_soft_dtw_falls_back_to_cpu_when_numba_cuda_import_fails(monkeypatch):
    soft_dtw_cuda = _load_soft_dtw_module()
    monkeypatch.setattr(soft_dtw_cuda, "_CUDA_IMPORT_ERROR", AttributeError("missing NPDatetime"))
    monkeypatch.setattr(soft_dtw_cuda, "_CUDA_FALLBACK_WARNED", False)

    criterion = soft_dtw_cuda.SoftDTW(use_cuda=True, device="cpu", gamma=0.01)
    x = torch.zeros((1, 2, 3), dtype=torch.float32)
    y = torch.zeros((1, 2, 3), dtype=torch.float32)

    with pytest.warns(RuntimeWarning, match="numba.cuda failed to import"):
        func_dtw = criterion._get_func_dtw(x, y)

    assert getattr(func_dtw, "__self__", None) is soft_dtw_cuda._SoftDTW
