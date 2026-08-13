# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for selecting the renderer device by CUDA index."""

import pytest

from isaaclab.app.app_launcher import AppLauncher


def _resolve(launcher_args: dict) -> dict:
    """Run device resolution without constructing an ``AppLauncher``."""
    launcher = AppLauncher.__new__(AppLauncher)
    launcher.device_id = 0
    launcher._deferred_cuda_device_id = None
    launcher._xr = False
    AppLauncher._resolve_device_settings(launcher, launcher_args)
    return launcher_args


def test_active_gpu_is_not_set():
    """Leave ``activeGpu`` unset so the renderer applies the CUDA index translation."""
    args = _resolve({"device": "cuda:0"})

    assert args["active_gpu"] is None


def test_renderer_selected_by_cuda_index():
    """Select the renderer device through the CUDA-indexed setting."""
    args = _resolve({"device": "cuda:1"})

    assert "--/renderer/multiGpu/activeCudaGpus=1," in args["extra_args"]


def test_physics_keeps_the_cuda_index():
    """Keep the masked index for physics, which CUDA resolves itself."""
    args = _resolve({"device": "cuda:1"})

    assert args["physics_gpu"] == 1


@pytest.mark.parametrize("device", ["cuda:0", "cuda:3"])
def test_cuda_index_setting_is_comma_terminated(device):
    """Terminate the value with a comma: a bare integer is silently ignored by the renderer."""
    args = _resolve({"device": device})

    cuda_gpu_args = [arg for arg in args["extra_args"] if "activeCudaGpus" in arg]
    assert len(cuda_gpu_args) == 1
    assert cuda_gpu_args[0].endswith(",")


def test_user_extra_args_are_preserved():
    """Append to caller-provided ``extra_args`` rather than replacing them."""
    args = _resolve({"device": "cuda:0", "extra_args": ["--/app/fastShutdown=False"]})

    assert "--/app/fastShutdown=False" in args["extra_args"]
    assert any("activeCudaGpus" in arg for arg in args["extra_args"])
